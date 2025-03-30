#!/bin/bash

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet50 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0001 \
--max-epoch 10 \
--lr-scheduler multi_step
--stepsize 4 8 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50_1e-4 \

