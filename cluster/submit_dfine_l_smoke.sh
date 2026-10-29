#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs runs
EPOCHS=1 BATCH_SIZE=2 VAL_BATCH_SIZE=2 WORKERS=2 \
OUT="$PROJECT_ROOT/runs/dfine_l_smoke" \
sbatch --export=ALL,EPOCHS,BATCH_SIZE,VAL_BATCH_SIZE,WORKERS,OUT \
  --job-name=cs701-dfine-smoke "$@" cluster/train_dfine_l.slurm
