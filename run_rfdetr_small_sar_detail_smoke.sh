#!/usr/bin/env bash
# One-epoch, 32-image check for the RF-DETR SAR detail branch.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --detail-fusion \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum-steps 1 \
  --classification-weight 1.0 \
  --box-loss-scale 1.0 \
  --eval-every 1 \
  --workers 2 \
  --limit 32 \
  --keep-last-checkpoints 1 \
  --out runs/smoke \
  --run-name rfdetr_small_sar_detail_smoke \
  "$@"
