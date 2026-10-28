#!/usr/bin/env bash
# One-epoch, 32-image check for the stride-4 small-object head.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --small-object-head \
  --small-object-max-size 32 \
  --small-object-loss-weight 0.5 \
  --small-object-score-threshold 0.01 \
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
  --run-name rfdetr_small_smallobj_s4_smoke \
  "$@"
