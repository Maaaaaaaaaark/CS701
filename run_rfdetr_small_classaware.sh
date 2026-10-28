#!/usr/bin/env bash
# RF-DETR Small baseline with 1.5x sampling weights for bridge and playground images.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --class-aware-sampling \
  --class-aware-classes bridge playground \
  --class-aware-factor 1.5 \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --classification-weight 1.0 \
  --box-loss-scale 1.0 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  --run-name rfdetr_small_coco_full_e48_classaware_b15_p15 \
  "$@"
