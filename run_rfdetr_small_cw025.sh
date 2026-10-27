#!/usr/bin/env bash
# Detection-prioritized RF-DETR Small ablation: only the image-classification loss weight changes.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --classification-weight 0.25 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  --run-name rfdetr_small_coco_full_e48_cw025 \
  "$@"
