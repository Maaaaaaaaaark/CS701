#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

export RUN_SCRIPT=run_dinov3_experiment_c.sh
sbatch --job-name=cs701-dino3-exp-c "$@" cluster/train_dinov3.slurm
