#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
module load Python/3.12.8-GCCcore-13.3.0
source cluster/project_env.sh

DFINE_ROOT="$PROJECT_ROOT/.dfine/source"
DFINE_COMMIT="956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHT="$PROJECT_ROOT/.dfine/weights/dfine_l_obj2coco_e25.pth"

mkdir -p .dfine/weights logs runs
if [[ ! -d "$DFINE_ROOT/.git" ]]; then
  git clone https://github.com/Peterande/D-FINE.git "$DFINE_ROOT"
fi
git -C "$DFINE_ROOT" fetch origin "$DFINE_COMMIT"
git -C "$DFINE_ROOT" checkout --detach "$DFINE_COMMIT"

if [[ ! -x .venv-dfine/bin/python ]]; then
  python3 -m venv .venv-dfine
fi
source .venv-dfine/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.0 torchvision==0.24.0 \
  faster-coco-eval PyYAML tensorboard scipy calflops transformers loguru \
  pillow tqdm pycocotools

if [[ ! -s "$WEIGHT" ]]; then
  curl -L --fail --retry 3 \
    https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth \
    -o "$WEIGHT"
fi

mkdir -p "$DFINE_ROOT/configs/cs701"
cp configs/dfine_l_sar512.yml "$DFINE_ROOT/configs/cs701/dfine_l_sar512.yml"
python prepare_dfine_data.py --data cs701-sar-course-data --out .dfine/data

echo "D-FINE environment is ready"
echo "source .venv-dfine/bin/activate"
echo "checkpoint: $WEIGHT"
