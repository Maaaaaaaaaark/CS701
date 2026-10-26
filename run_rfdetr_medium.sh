#!/usr/bin/env bash
# Full fine-tuning of COCO-pretrained RF-DETR Medium at the course-legal 512 resolution.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant medium \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --classification-weight 1.0 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  "$@"
