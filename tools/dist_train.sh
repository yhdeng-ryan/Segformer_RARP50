#!/usr/bin/env bash

CONFIG=${1:-local_configs/segformer/B1/segformer.b1.512x512.rarp50.aug.160k.py}
GPUS=${2:-1}
PORT=${PORT:-29500}
CHECKPOINT=${3:-work_dirs/segformer.b1.512x512.endvis2018.80k/iter_44000.pth}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --load-from $CHECKPOINT --launcher pytorch ${@:3}
