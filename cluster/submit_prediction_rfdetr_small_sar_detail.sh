#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs submissions
export DETAIL_FUSION=1
export OUTPUT="${OUTPUT:-submissions/rfdetr_small_coco_full_e48_sar_detail_val.zip}"
sbatch --job-name=cs701-submit-rfdetr-detail "$@" cluster/predict_rfdetr_small.slurm
