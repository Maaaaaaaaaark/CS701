#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs runs
export RUN_SCRIPT=run_rfdetr_small_sar_detail_det_only.sh
sbatch --job-name=cs701-rfdetr-detail-det "$@" cluster/train_rfdetr_small.slurm
