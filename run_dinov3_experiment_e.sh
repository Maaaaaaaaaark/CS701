#!/usr/bin/env bash
# Experiment E: widen each Cascade box head from 384 to 448 while staying below 40 M trainable parameters.
set -e
exec bash run_dinov3_detail.sh \
  --cascade \
  --cascade-hidden-dim 448 \
  --epochs 36 \
  --keep-last-checkpoints 3 \
  "$@"
