#!/usr/bin/env bash
# Full fine-tuning of ImageNet-pretrained CSPNeXt-m + RTMDet + image classification head.
set -e
python train_rtmdet.py \
  --data cs701-sar-course-data \
  --epochs 48 \
  --backbone-lr 1e-4 \
  --lr 2e-4 \
  --keep-last-checkpoints 3 \
  "$@"
