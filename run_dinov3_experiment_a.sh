#!/usr/bin/env bash
# Experiment A: DINOv3 detail stem + dihedral orientation augmentation + longer training.
set -e
exec bash run_dinov3_detail.sh \
  --rotation90 \
  --epochs 36 \
  --keep-last-checkpoints 3 \
  "$@"
