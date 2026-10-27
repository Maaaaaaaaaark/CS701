#!/usr/bin/env bash
# RF-DETR Small localization ablation: scale every bbox and GIoU loss by 1.5.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --classification-weight 1.0 \
  --box-loss-scale 1.5 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  --run-name rfdetr_small_coco_full_e48_box15 \
  "$@"
