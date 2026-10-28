#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs runs
export RUN_SCRIPT=run_rfdetr_small_classaware.sh
sbatch --job-name=cs701-rfdetr-classaware "$@" cluster/train_rfdetr_small.slurm
