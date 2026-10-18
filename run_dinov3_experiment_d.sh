#!/usr/bin/env bash
# Experiment D: add right-angle rotation augmentation to the 36-epoch Cascade configuration.
set -e
exec bash run_dinov3_detail.sh \
  --cascade \
  --rotation90 \
  --epochs 36 \
  --keep-last-checkpoints 3 \
  "$@"
