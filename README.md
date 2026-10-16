# CS701 SAR Multi-Task Benchmark

Reference code for the CS701 team assignment. One ViT-B/16 backbone, two tasks on SAR images: **classify** the
image (9 classes) and **detect** its objects.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Train Baseline](#22-train-baseline)
   - [2.3 Try DINOv3](#23-try-dinov3)
   - [2.4 Use Course Data](#24-use-course-data)
   - [2.5 Make Submission](#25-make-submission)
3. [Baseline Results (for reference)](#3-baseline-results-for-reference)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 106-107 M) and at most 40 M
  trainable (heads included: LoRA 21.0 M; full fine-tuning and from scratch, 106 M, are over); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs a reference model (ViT + LoRA, the example command
  in 2.2; mean of 3 seeds):
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound).

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Train Baseline

```bash
python train.py --backbone vit --init pretrained --adapt lora   # the Δm reference; one run, about 50 min
bash run_all.sh                                                  # all eight runs, about 7 h
```

- Options: `--backbone vit|terramind|dinov3`, `--init pretrained|scratch`, `--adapt full|lora|moelora`
  (`python train.py -h`).
- `--adapt full` and `--init scratch` train 106 M parameters: over the 40 M budget, for reference only.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Times for one RTX PRO 6000 Blackwell GPU. Memory: [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.3 Try DINOv3

The included experiment mirrors the team's DINOv2 recipe as closely as the available information permits:
DINOv3 ViT-B/16 LVD-1689M, LoRA rank 32 on every attention QKV/projection, plus the final two transformer blocks
fully trainable. It uses a fixed class-stratified 90/10 split and stays under both course parameter limits:
107.13 M total and 36.06 M trainable, including both task heads.

```bash
# The first download may require accepting Meta's DINOv3 terms on Hugging Face, then:
hf auth login
bash run_dinov3.sh
```

Local validation metrics are printed during training and saved in
`runs/dinov3_pretrained_lora_r32_u2/metrics.json`. The exact split is saved beside it in `split.json`.

For the small-object follow-up, `bash run_dinov3_detail.sh` adds a lightweight CNN path from the original image
to the stride-4 detector feature. Its separate output is `runs/dinov3_pretrained_lora_r32_u2_detail/`.

For a self-contained Slurm installation (environment, caches, logs, data and results all below this repository),
see `cluster/setup_dinov3.sh`, `cluster/submit_dinov3.sh` and the cluster instructions below.

### 2.4 Use Course Data

The code auto-detects the released
[course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data) when it is placed at
`cs701-sar-course-data/`. It reads `train/labels.csv` and `train/instances.json`, then creates the reproducible
local holdout controlled by `--val-fraction` and `--split-seed`.

The released validation and test labels remain hidden, so local metrics come from the training holdout; official
validation scores still require a Codabench submission.

### 2.5 Make Submission

`make_submission.py` has `write_submission` (the function in the briefing video): maps your boxes back to
original image pixels, writes `submission.zip` for Codabench.

## 3. Baseline Results (for reference)

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
train.py              entry: train + evaluate one configuration
run_all.sh            the eight reference runs (calls train.py)
make_submission.py    write_submission(): predictions -> submission.zip
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base, DINOv3 ViT-B/16
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
└── metrics.py        accuracy, macro-F1, COCO box AP
tests/                checks of sarbench/ (python -m pytest tests)
docs/DETAILS.md       model, training protocol, design notes, time and memory
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
