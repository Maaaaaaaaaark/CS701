#!/usr/bin/env bash
#SBATCH --job-name=cs701-dinov3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Batch jobs must not fall back to Origami's system Python 3.9.
module load Python/3.12.8-GCCcore-13.3.0

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
cd "$PROJECT_ROOT"
source cluster/project_env.sh

source .venv/bin/activate

echo "project=$PROJECT_ROOT"
echo "job_id=$SLURM_JOB_ID node=$(hostname)"
nvidia-smi
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError('PyTorch cannot see the allocated CUDA GPU')
print('gpu:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
PY

# Override these at submission time. RUN_SCRIPT lets the detail-stem wrapper reuse the same Slurm job.
bash "${RUN_SCRIPT:-run_dinov3.sh}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --workers "${WORKERS:-8}"
