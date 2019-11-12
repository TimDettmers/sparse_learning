#!/bin/bash
IMAGENET_PATH=/home/tim/data/imagenet/

# Pick a density
#DENSITY=0.25626087
#DENSITY=0.15626087
DENSITY=0.4

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python main.py --model imagenet_resnet50 --schedule-file ./learning_schedules/resnet_schedule.yaml --initial-sparsity-fc 0.0 --initial-sparsity-conv 0.0 --batch-size 256 --widen-factor 1 --weight-decay 1.0e-4 --no-validate-train --sub-kernel-granularity 4 --job-idx 15 --sparse-momentum --data $IMAGENET_PATH --verbose --density $DENSITY --prune-rate 0.2 --workers 32 --fp16 -p 300 --seed 14
