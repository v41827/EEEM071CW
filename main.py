# Copyright (c) EEEM071, University of Surrey

import datetime
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src import models
from src.data_manager import ImageDataManager
from src.eval_metrics import evaluate
from src.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer
from src.utils.avgmeter import AverageMeter
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from src.utils.loggers import Logger, RankLogger
from src.utils.torchtools import (
    count_num_param,
    accuracy,
    load_pretrained_weights,
    save_checkpoint,
    resume_from_checkpoint,
)
from src.utils.visualtools import visualize_ranked_results
import uuid

# global variables
parser = argument_parser()
args = parser.parse_args()

wandb.init(
    project="eeem071-veri",
    name=f"{args.arch}-{time.strftime('%Y%m%d-%H%M%S')}",
    config=vars(args)
)

def main():
    global args

    set_random_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU via MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Using CPU only — GPU is not available or not selected.")

    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========")
    student_id = os.environ.get('STUDENT_ID', '<your id>')
    student_name = os.environ.get('STUDENT_NAME', '<your name>')
    print("Student ID:{}".format(student_id))
    print("Student name:{}".format(student_name))
    print("UUID:{}".format(uuid.uuid4()))
    print("Experiment time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("==========")
    print(f"==========\nArgs:{args}\n==========")

    print("Initializing image data manager")
    dm = ImageDataManager(True, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print(f"Initializing model: {args.arch}")
    model = models.init_model(
        name=args.arch,
        num_classes=dm.num_train_pids,
        loss={"xent", "htri"},
        pretrained=not args.no_pretrained,
        use_gpu=True,
    )
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = model.to(device)
    wandb.watch(model, log="all", log_freq=100)

    criterion_xent = CrossEntropyLoss(
        num_classes=dm.num_train_pids, label_smooth=args.label_smooth, device=device
    )
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print(f"Evaluating {name} ...")
            queryloader = testloader_dict[name]["query"]
            galleryloader = testloader_dict[name]["gallery"]
            distmat = test(
                model, queryloader, galleryloader, device, return_distmat=True
            )

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat,
                    dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, "ranked_results", name),
                    topk=20,
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    best_rank1 = -1
    print("=> Start training")

    for epoch in range(args.start_epoch, args.max_epoch):
        train(
            epoch,
            model,
            criterion_xent,
            criterion_htri,
            optimizer,
            trainloader,
            device,
        )

        scheduler.step()

        if (
            (epoch + 1) > args.start_eval
            and args.eval_freq > 0
            and (epoch + 1) % args.eval_freq == 0
            or (epoch + 1) == args.max_epoch
        ):
            print("=> Test")

            for name in args.target_names:
                print(f"Evaluating {name} ...")
                queryloader = testloader_dict[name]["query"]
                galleryloader = testloader_dict[name]["gallery"]
                rank1 = test(model, queryloader, galleryloader, device)
                ranklogger.write(name, epoch + 1, rank1)

                if rank1 > best_rank1:
                    best_rank1 = rank1
                    print(f"✅ Saving new best model (Rank-1: {rank1:.2f}%)")
                    save_path = os.path.join(wandb.run.dir, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
                    wandb.save(save_path)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")
    ranklogger.show_summary()

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, device):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        imgs, pids = imgs.to(device), pids.to(device)

        outputs, features = model(imgs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]	"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})	"
                "Data {data_time.val:.4f} ({data_time.avg:.4f})	"
                "Xent {xent.val:.4f} ({xent.avg:.4f})	"
                "Htri {htri.val:.4f} ({htri.avg:.4f})	"
                "Acc {acc.val:.2f} ({acc.avg:.2f})	".format(
                    epoch + 1,
                    batch_idx + 1,
                    len(trainloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    xent=xent_losses,
                    htri=htri_losses,
                    acc=accs,
                )
            )

        end = time.time()

def test(model, queryloader, galleryloader, device, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            imgs = imgs.to(device)
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            imgs = imgs.to(device)
            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print(f"=> BatchTime(s)/BatchSize(img): {batch_time.avg:.3f}/{args.test_batch_size}")

    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]
# Optional: upload saved best model to W&B
best_model_path = os.path.join(args.save_dir, "best_model.pth")
if os.path.exists(best_model_path):
    artifact = wandb.Artifact(name="mobilenet_v3_small-best-model", type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)
    print("✅ Uploaded best_model.pth to W&B as an artifact.")

if __name__ == "__main__":
    main()
    wandb.finish()



