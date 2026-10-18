#!/usr/bin/env bash
# Experiment C: keep Experiment B unchanged except for extending Cascade training to 36 epochs.
set -e
exec bash run_dinov3_detail.sh \
  --cascade \
  --epochs 36 \
  --keep-last-checkpoints 3 \
  "$@"
