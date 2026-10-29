#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
DFINE_ROOT="$PROJECT_ROOT/.dfine/source"
CONFIG="$DFINE_ROOT/configs/cs701/dfine_l_sar512.yml"
WEIGHT="$PROJECT_ROOT/.dfine/weights/dfine_l_obj2coco_e25.pth"
OUT="${OUT:-$PROJECT_ROOT/runs/dfine_l_obj2coco_sar512_e36}"

[[ -x .venv-dfine/bin/python ]] || { echo "Run cluster/setup_dfine.sh first" >&2; exit 1; }
[[ -f "$WEIGHT" ]] || { echo "Missing $WEIGHT; run cluster/setup_dfine.sh" >&2; exit 1; }
mkdir -p "$OUT" "$DFINE_ROOT/configs/cs701"
cp configs/dfine_l_sar512.yml "$CONFIG"
.venv-dfine/bin/python prepare_dfine_data.py --data cs701-sar-course-data --out .dfine/data
.venv-dfine/bin/python check_dfine_budget.py --source "$DFINE_ROOT" --config "$CONFIG"

EXTRA=()
if [[ -n "${RESUME:-}" ]]; then
  EXTRA=(-r "$RESUME")
else
  EXTRA=(-t "$WEIGHT")
fi

exec .venv-dfine/bin/python "$DFINE_ROOT/train.py" \
  -c "$CONFIG" \
  "${EXTRA[@]}" \
  --use-amp \
  --seed "${SEED:-0}" \
  --output-dir "$OUT" \
  -u \
  "epochs=${EPOCHS:-36}" \
  "train_dataloader.total_batch_size=${BATCH_SIZE:-8}" \
  "train_dataloader.num_workers=${WORKERS:-8}" \
  "val_dataloader.total_batch_size=${VAL_BATCH_SIZE:-8}" \
  "val_dataloader.num_workers=${WORKERS:-8}"
