#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs runs/smoke
export RUN_SCRIPT=run_rfdetr_small_smoke.sh
export WORKERS="${WORKERS:-2}"
sbatch --job-name=cs701-rfdetr-smoke "$@" cluster/train_rfdetr_small.slurm
