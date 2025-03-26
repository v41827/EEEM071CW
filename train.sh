#!/bin/bash

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a mobilenet_v3_small \
--root /Users/yiwenchan/Desktop/Workspace \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/mobilenet_v3_small-veri-veri_ot1 \
