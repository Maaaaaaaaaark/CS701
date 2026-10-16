#!/usr/bin/env bash
set -euo pipefail

# Origami's login default is Python 3.9; use the supported project version explicitly.
module load Python/3.12.8-GCCcore-13.3.0

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
cd "$PROJECT_ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dinov3.txt

python - <<'PY'
import torch
import timm

print(f'PyTorch {torch.__version__}')
print(f'timm {timm.__version__}')
print(f'CUDA build: {torch.version.cuda}')
print('Environment is ready.')
PY

echo "Next run: source cluster/project_env.sh && .venv/bin/hf auth login"
