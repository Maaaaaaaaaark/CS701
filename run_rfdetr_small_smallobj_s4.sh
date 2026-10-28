#!/usr/bin/env bash
# RF-DETR Small baseline plus a stride-4 anchor-free head for <=32 px objects.
set -e
python train_rfdetr.py \
  --data cs701-sar-course-data \
  --variant small \
  --small-object-head \
  --small-object-max-size 32 \
  --small-object-loss-weight 0.5 \
  --small-object-score-threshold 0.01 \
  --epochs 48 \
  --batch-size "${BATCH_SIZE:-4}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-2}" \
  --encoder-lr 1.5e-4 \
  --lr 1e-4 \
  --classification-weight 1.0 \
  --box-loss-scale 1.0 \
  --eval-every 4 \
  --keep-last-checkpoints 3 \
  --run-name rfdetr_small_coco_full_e48_smallobj_s4 \
  "$@"
