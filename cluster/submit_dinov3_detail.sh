#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

export RUN_SCRIPT=run_dinov3_detail.sh
sbatch --job-name=cs701-dino3-detail "$@" cluster/train_dinov3.slurm
