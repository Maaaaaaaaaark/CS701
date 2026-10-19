#!/usr/bin/env bash
# End-to-end GPU check on 32 images before spending hours on the full experiment.
set -e
exec bash run_rtmdet_m.sh \
  --epochs 1 \
  --eval-every 1 \
  --limit 32 \
  --keep-last-checkpoints 1 \
  --out runs/smoke \
  "$@"
