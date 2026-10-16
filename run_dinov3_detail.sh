#!/usr/bin/env bash
# DINOv3 hybrid PEFT plus a pixel-level stride-4 feature for sub-patch objects.
set -e
exec bash run_dinov3.sh --detail-stem "$@"
