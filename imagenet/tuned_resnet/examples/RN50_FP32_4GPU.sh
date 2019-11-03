# This script launches ResNet50 training in FP32 on 4 GPUs using 512 batch size (128 per GPU)
# Usage ./RN50_FP32_4GPU.sh <path to this repository> <additional flags>

python $1/multiproc.py --nproc_per_node 4 $1/main.py -j5 -p 300 --arch resnet50 -c fanin --label-smoothing 0.1 -b 64 --lr 0.2 --warmup 5 --epochs 100 --density 0.2 --gather-checkpoints $2
