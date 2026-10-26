#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs submissions
export VARIANT=medium
export OUTPUT="${OUTPUT:-submissions/rfdetr_medium_val.zip}"
sbatch --job-name=cs701-submit-rfdetr-medium "$@" cluster/predict_rfdetr_small.slurm
