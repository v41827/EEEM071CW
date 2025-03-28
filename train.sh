#!/bin/bash

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet50 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50 \


STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet50_fc512 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50_fc512 \

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a mobilenet_v3_small \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/mobilenet_v3_small \


STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a vgg16 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/vgg16 \