#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs submissions
export SMALL_OBJECT_HEAD=1
export SMALL_OBJECT_MAX_SIZE="${SMALL_OBJECT_MAX_SIZE:-32}"
export SMALL_OBJECT_SCORE_THRESHOLD="${SMALL_OBJECT_SCORE_THRESHOLD:-0.01}"
export OUTPUT="${OUTPUT:-submissions/rfdetr_small_coco_full_e48_smallobj_s4_val.zip}"
sbatch --job-name=cs701-submit-rfdetr-smallobj "$@" cluster/predict_rfdetr_small.slurm
