#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
--training-dataset GLDv2 \
--imsize 512 \
--batch-size 128 \
--num-workers 10 \
--device cuda \
--num-epochs 30 \
--val-epoch 1 \
--warmup-epochs 5 \
--warmup-lr 0.0001 \
--base-lr 0.01 \
--final-lr 0 \
--momentum 0.9 \
--weight-decay 0.0001 \
--clip_max_norm -1.0 \
--update-every 1 \
--seed 11 \












