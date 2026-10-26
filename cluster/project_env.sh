#!/usr/bin/env bash
# Source this file before setup, Hugging Face login, or training to keep all generated files in the repository.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/nvidia"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export RF_HOME="$PROJECT_ROOT/.cache/rfdetr"
export TMPDIR="$PROJECT_ROOT/.tmp"

mkdir -p "$PIP_CACHE_DIR" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$RF_HOME" "$TMPDIR" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/runs"
