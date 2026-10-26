#!/usr/bin/env bash
#SBATCH --job-name=cs701-rfdetr-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load Python/3.12.8-GCCcore-13.3.0
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the project directory}"
cd "$PROJECT_ROOT"
source cluster/project_env.sh
source .venv-rfdetr/bin/activate

echo "project=$PROJECT_ROOT"
echo "job_id=$SLURM_JOB_ID node=$(hostname)"
nvidia-smi
python - <<'PY'
from importlib.metadata import version
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the allocated GPU")
print("gpu:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "rfdetr:", version("rfdetr"))
PY

EXTRA_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi

bash "${RUN_SCRIPT:-run_rfdetr_small.sh}" \
  --workers "${WORKERS:-8}" \
  "${EXTRA_ARGS[@]}"
