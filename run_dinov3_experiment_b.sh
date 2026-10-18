#!/usr/bin/env bash
# Experiment B: isolate a budget-aware compact Cascade R-CNN head at the original 24-epoch schedule.
set -e
exec bash run_dinov3_detail.sh \
  --cascade \
  --epochs 24 \
  --keep-last-checkpoints 3 \
  "$@"
