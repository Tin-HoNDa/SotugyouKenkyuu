#!/bin/bash

cd AnimateDiff || exit 2

pip3 install accelerate triton

torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/training.yaml
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/image_finetune.yaml

python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/2-Lyriel.yaml
python -m scripts.animate --config configs/prompts/3-RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/4-MajicMix.yaml
python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml
python -m scripts.animate --config configs/prompts/6-Tusun.yaml
python -m scripts.animate --config configs/prompts/7-FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/8-GhibliBackground.yaml
