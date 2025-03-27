#!/bin/bash

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--random-erase \
--save-dir logs/mobilenet_v3_small_random-erase \

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--random-erase \
--color-jitter \
--save-dir logs/mobilenet_v3_small_random-erase_color-jitter \

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--color-aug \
--save-dir logs/mobilenet_v3_small_color_aug \

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--color-jitter \
--color-aug \
--save-dir logs/mobilenet_v3_small_color-jitter_color-aug \

STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--random-erase \
--color-aug \
--save-dir logs/mobilenet_v3_small_random-erase_color-aug \


STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--color-jitter \
--save-dir logs/mobilenet_v3_small_color-jitter \


STUDENT_ID=yc01847 STUDENT_NAME="Yiwen Chan" python main.py \
-s veri \
-t veri \
-a resnet18 \
--root /user/HS400/yc01847/Desktop \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--color-jitter \
--random-erase \
--color-aug \
--save-dir logs/mobilenet_v3_small_color-jitter_random-erase_color-aug \
