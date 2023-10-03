#!/bin/bash
cd AnimateDiff || exit 2

torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/training.yaml
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/image_finetune.yaml

