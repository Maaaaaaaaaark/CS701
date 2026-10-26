#!/usr/bin/env bash
set -euo pipefail

module load Python/3.12.8-GCCcore-13.3.0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
cd "$PROJECT_ROOT"

python3 -m venv .venv-rfdetr
source .venv-rfdetr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rfdetr.txt

python - <<'PY'
from importlib.metadata import version
import torch
from rfdetr import RFDETRSmall

print(f"PyTorch {torch.__version__}")
print(f"RF-DETR {version('rfdetr')}")
print(f"CUDA build: {torch.version.cuda}")
model = RFDETRSmall(num_classes=8, num_channels=1, resolution=512)
print("Official RF-DETR Small weights downloaded successfully")
print(f"Detection parameters: {sum(p.numel() for p in model.model.model.parameters()):,}")
PY

echo "RF-DETR environment and weights are ready."
