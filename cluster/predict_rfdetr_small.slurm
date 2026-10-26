#!/usr/bin/env bash
#SBATCH --job-name=cs701-submit-rfdetr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load Python/3.12.8-GCCcore-13.3.0
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the project directory}"
cd "$PROJECT_ROOT"
source cluster/project_env.sh
source .venv-rfdetr/bin/activate

: "${CHECKPOINT:?Set CHECKPOINT to an RF-DETR model checkpoint}"
EXTRA_ARGS=()
if [[ -n "${CLASS_FUSION_ALPHA:-}" ]]; then
  EXTRA_ARGS+=(--class-fusion-alpha "$CLASS_FUSION_ALPHA")
fi
if [[ "${CLASS_GATE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--class-gate)
fi
if [[ -n "${TOPK:-}" ]]; then
  EXTRA_ARGS+=(--topk "$TOPK")
fi
python make_submission_rfdetr.py \
  --data cs701-sar-course-data/val \
  --checkpoint "$CHECKPOINT" \
  --variant "${VARIANT:-small}" \
  --output "${OUTPUT:-submissions/rfdetr_small_val.zip}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" \
  "${EXTRA_ARGS[@]}"
