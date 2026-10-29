#!/usr/bin/env bash
#SBATCH --job-name=cs701-submit-dfine
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
source .venv-dfine/bin/activate

: "${CHECKPOINT:?Set CHECKPOINT to best_stg1.pth or best_stg2.pth}"
python make_submission_dfine.py \
  --checkpoint "$CHECKPOINT" \
  --output "${OUTPUT:-submissions/dfine_l_sar512_val.zip}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --workers "${WORKERS:-4}" \
  --score-threshold "${SCORE_THRESHOLD:-0.01}" \
  --topk "${TOPK:-100}" \
  --class-pool "${CLASS_POOL:-top5-sum}"
