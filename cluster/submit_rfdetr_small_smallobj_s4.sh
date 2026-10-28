#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs runs
export RUN_SCRIPT=run_rfdetr_small_smallobj_s4.sh
sbatch --job-name=cs701-rfdetr-smallobj "$@" cluster/train_rfdetr_small.slurm
