#!/usr/bin/env bash
# One-epoch, 32-image check for RF-DETR Medium at 512 x 512.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant medium \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --eval-every 1 \
  --workers 2 \
  --limit 32 \
  --keep-last-checkpoints 1 \
  --out runs/smoke \
  "$@"
