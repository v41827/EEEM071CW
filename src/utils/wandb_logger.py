import wandb
import os
from typing import Optional
import torch.nn as nn

class WandbLogger:
    def __init__(self, args, run_name: Optional[str] = None): #“You can give me a string, or nothing — and I’ll handle both cases.
        """
        Initializes the Weights & Biases logger.
        """
        self.run = wandb.init(
            project="CV coursework",
            name=run_name or f"{args.arch}-{wandb.util.generate_id()}", #run_name taken from train.sh
            config=vars(args))

        self.save_dir = args.save_dir

    def log_metrics(self, epoch: int, xent_loss: float, htri_loss: float, total_loss: float, acc: float):
        """
        Logs training metrics to wandb.
        """
        self.run.log({
            "epoch": epoch,   #optional for the coursework
            "loss/xent": xent_loss,
            "loss/htri": htri_loss,
            "loss/total": total_loss,
            "train/accuracy": acc  #optional for the coursework
            "learning_rate": lr #optional for the coursework
        }, step=epoch)

    def log_eval_metrics(self, epoch: int, cmc, mAP: float):
        """
        Logs evaluation metrics (CMC ranks and mAP) to wandb.
        """
        self.run.log({
            "rank-1": cmc[0],
            "rank-5": cmc[4],
            "rank-10": cmc[9],
            "mAP": mAP
        }, step=epoch)

    def watch_model(self, model: nn.Module):
        """
        Monitors model weights and gradients.
        """
        self.run.watch(model, log="all", log_freq=100)

    def save_model_artifact(self, model_path: str, name: str = "best_model"):
        """
        Uploads the saved model file as a wandb artifact.
        """
        if os.path.exists(model_path):
            artifact = wandb.Artifact(name=name, type="model")
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """
        Finishes the wandb run.
        """
        self.run.finish()
