#!/usr/bin/env bash
# Full fine-tuning of COCO-pretrained RF-DETR Small plus a shared image-classification head.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  "$@"
