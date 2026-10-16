#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# All arguments are passed to sbatch, so cluster-specific options stay outside the tracked job script.
sbatch "$@" cluster/train_dinov3.slurm
