#!/usr/bin/env bash
# One-epoch, 32-image end-to-end check before allocating a full RF-DETR run.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --eval-every 1 \
  --workers 2 \
  --limit 32 \
  --keep-last-checkpoints 1 \
  --out runs/smoke \
  "$@"
