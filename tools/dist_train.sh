#!/usr/bin/env bash

CONFIG=${1:-local_configs/segformer/B0/segformer.b0.512x512.rarp50.40k.py}
GPUS=${2:-1}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
