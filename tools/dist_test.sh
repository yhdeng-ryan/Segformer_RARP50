#!/usr/bin/env bash

CONFIG=${1:-local_configs/segformer/B0/segformer.b0.512x512.rarp50.40k.py}
CHECKPOINT=${2:-pretrained/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth}
GPUS=${2:-1}
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
