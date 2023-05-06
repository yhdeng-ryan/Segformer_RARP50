#!/usr/bin/env bash

CONFIG=work_dirs/test_val_cfg.py
GPUS=1
PORT=${PORT:-29500}

for CHECKPOINT in \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_4000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_8000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_12000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_16000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_20000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_24000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_28000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_32000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_36000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_40000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_44000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_48000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_52000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_56000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_60000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_64000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_68000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_72000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_76000.pth \
work_dirs/segformer.b1.512x512.rarp50.80k/iter_80000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_84000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_88000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_92000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_96000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_100000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_104000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_108000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_112000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_116000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_120000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_124000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_128000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_132000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_136000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_140000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_144000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_148000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_152000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_156000.pth \
work_dirs/segformer.b1.512x512.rarp50.160k/iter_160000.pth
do

echo $CHECKPOINT
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --out work_dirs/res.pkl --show

done

