#!/usr/bin/env bash
#SBATCH --job-name=cs701-rfdetr-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load Python/3.12.8-GCCcore-13.3.0
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the project directory}"
cd "$PROJECT_ROOT"
source cluster/project_env.sh
source .venv-rfdetr/bin/activate

: "${CHECKPOINT:?Set CHECKPOINT to an RF-DETR model_best_map.pt}"
python diagnose_rfdetr.py \
  --checkpoint "$CHECKPOINT" \
  --variant "${VARIANT:-small}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" \
  --output "${OUTPUT:-runs/rfdetr_postprocess_diagnostic.json}"
