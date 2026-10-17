#!/usr/bin/env bash
#SBATCH --job-name=cs701-submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load Python/3.12.8-GCCcore-13.3.0
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
cd "$PROJECT_ROOT"
source cluster/project_env.sh
source .venv/bin/activate

python make_submission.py \
  --data cs701-sar-course-data/val \
  --checkpoint "${CHECKPOINT:-runs/dinov3_pretrained_lora_r32_u2_detail/model.pt}" \
  --output "${OUTPUT:-submissions/dinov3_detail_val.zip}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --workers "${WORKERS:-8}"
