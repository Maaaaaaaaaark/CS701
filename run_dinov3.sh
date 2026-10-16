#!/usr/bin/env bash
# DINOv3-B/16 hybrid PEFT recipe inspired by the team's DINOv2 run:
# rank-32 LoRA on every attention block, plus full training of the last two blocks.
set -e
python train.py \
  --data cs701-sar-course-data \
  --backbone dinov3 \
  --init pretrained \
  --adapt lora \
  --lora-rank 32 \
  --lora-alpha 64 \
  --unfreeze-last-blocks 2 \
  --backbone-lr 2e-5 \
  --enforce-budget \
  "$@"
