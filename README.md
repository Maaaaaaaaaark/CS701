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

Experiment A runs that model for 36 epochs with random right-angle rotations and keeps the best-mAP model plus
the final three validation checkpoints: `bash run_dinov3_experiment_a.sh`. Its output is
`runs/dinov3_pretrained_lora_r32_u2_detail_rot90_e36/`.

Experiment B isolates a compact three-stage Cascade R-CNN head at the original 24-epoch schedule without the
rotation augmentation: `bash run_dinov3_experiment_b.sh`. Its output is
`runs/dinov3_pretrained_lora_r32_u2_detail_cascade/`.

Experiment C keeps Experiment B unchanged except for extending the schedule to 36 epochs. It still does not use
rotation augmentation: `bash run_dinov3_experiment_c.sh`. Its separate output is
`runs/dinov3_pretrained_lora_r32_u2_detail_cascade_e36/`.

Experiment D adds random right-angle rotation augmentation to Experiment C while keeping its 36-epoch Cascade
configuration unchanged: `bash run_dinov3_experiment_d.sh`. Its separate output is
`runs/dinov3_pretrained_lora_r32_u2_detail_cascade_rot90_e36/`.

Experiment E returns to the unrotated Experiment C recipe and widens each Cascade box head from 384 to 448. It
uses 110.82 M total and 39.75 M trainable parameters: `bash run_dinov3_experiment_e.sh`. Its output is
`runs/dinov3_pretrained_lora_r32_u2_detail_cascade_h448_e36/`.

For a self-contained Slurm installation (environment, caches, logs, data and results all below this repository),
see `cluster/setup_dinov3.sh`, `cluster/submit_dinov3.sh` and the cluster instructions below.

### 2.4 Full-finetune RTMDet-m

`bash run_rtmdet_m.sh` trains one shared OpenMMLab ImageNet-pretrained CSPNeXt-m backbone end-to-end with a global image
classification branch and an RTMDet-style CSPNeXt PAFPN/SepBN detection branch. It uses Dynamic Soft Label
assignment, Quality Focal Loss and GIoU box loss. CSPNeXt-m is implemented locally and loads the official
OpenMMLab ImageNet-1k checkpoint, so it does not require MMCV. It enforces the 130 M total / 40 M trainable course
limits and writes to `runs/rtmdet_m_imagenet_full_e48/`.

On Slurm, use `cluster/submit_rtmdet_m.sh`. Every validation checkpoint also updates
`training_state_latest.pt`, which includes model, optimizer and scheduler state for recovery after preemption.
Set the `RESUME` environment variable to that path when resubmitting. Only ImageNet backbone weights are loaded;
the PAFPN, detector and image classifier are trained from scratch on the course data.
Run `cluster/submit_rtmdet_m_smoke.sh` first for a 32-image, one-epoch end-to-end check.

### 2.5 Full-finetune RF-DETR Small and Medium

`bash run_rfdetr_small.sh` fully fine-tunes the official COCO-pretrained RF-DETR Small detector and a new 9-class
image-classification head. The classifier globally pools RF-DETR's P4 backbone feature through a forward hook, so
classification and detection share one DINOv2 encoder invocation and each image enters the model exactly once. Input
is fixed at 512 x 512, multi-scale training and test-time augmentation are disabled, and each grayscale SAR image is
replicated across three channels with RF-DETR's native ImageNet normalization so every pretrained weight loads without
an input-channel mismatch. The run uses batch size 4 with two-step gradient
accumulation (effective batch 8), validates every four epochs, and writes to
`runs/rfdetr_small_coco_full_e48/`.

RF-DETR has an isolated environment because its Transformers/Lightning requirements differ from the DINOv3 stack:

```bash
bash cluster/setup_rfdetr.sh

# Always run this first. It uses 32 images for one epoch.
./cluster/submit_rfdetr_small_smoke.sh \
  --partition=researchshort --gres=gpu:a5000:1 --mem=16G --time=01:00:00

# Submit only after the smoke job exits successfully.
./cluster/submit_rfdetr_small.sh \
  --partition=researchshort --gres=gpu:a5000:1 --mem=32G --time=24:00:00
```

Every validation point updates `training_state_latest.pt`. If Slurm preempts a run, resubmit with
`RESUME=runs/rfdetr_small_coco_full_e48/training_state_latest.pt`. RF-DETR 1.11.2 is pinned in
`requirements-rfdetr.txt`; the official RF-DETR Small checkpoint is generic COCO pretraining, not SAR training, and
must be named in the final report.

`diagnose_rfdetr.py` reuses the Small checkpoint to compare classification-aware detection policies on the same
local holdout without retraining. It evaluates the unchanged detector, top-100 pruning, soft image-class score
fusion, fusion plus top-100, and a hard image-class gate. Run it on Slurm before spending a leaderboard submission:

```bash
CHECKPOINT=runs/rfdetr_small_coco_full_e48/model_best_map.pt \
  ./cluster/submit_rfdetr_diagnostic.sh \
  --partition=researchshort --gres=gpu:3090:1 --mem=32G --time=02:00:00
```

RF-DETR Medium is the next capacity ablation. Its native 576 input is overridden to the course-legal 512 x 512;
the program checks the actual model remains within both parameter budgets before training. Keep all other Small
hyperparameters unchanged so the comparison isolates the variant:

```bash
./cluster/submit_rfdetr_medium_smoke.sh \
  --partition=researchshort --gres=gpu:3090:1 --mem=16G --time=01:00:00

./cluster/submit_rfdetr_medium.sh \
  --partition=researchshort --gres=gpu:3090:1 --mem=32G --time=24:00:00
```

After comparing model capacity, the detection-prioritized Small ablation keeps the stronger Small architecture and
changes only the image-classification loss weight from 1.0 to 0.25. It writes to
`runs/rfdetr_small_coco_full_e48_cw025/`, leaving the original Small checkpoint untouched:

```bash
BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 WORKERS=8 \
  ./cluster/submit_rfdetr_small_cw025.sh \
  --partition=researchshort --gres=gpu:3090:1 --mem=40G --time=12:00:00
```

### 2.6 Use Course Data

The code auto-detects the released
[course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data) when it is placed at
`cs701-sar-course-data/`. It reads `train/labels.csv` and `train/instances.json`, then creates the reproducible
local holdout controlled by `--val-fraction` and `--split-seed`.

The released validation and test labels remain hidden, so local metrics come from the training holdout; official
validation scores require submission to the course leaderboard.

### 2.7 Make Submission

Extract `val.tar.gz`, then run `make_submission.py` with the trained checkpoint. It normalizes the validation
images exactly as in training, maps predicted boxes back to original pixels, writes the two required files into a
ZIP, and validates the archive. On Slurm, submit `cluster/submit_prediction.sh`; the default output is
`submissions/dinov3_detail_val.zip`.

For RTMDet-m, use `make_submission_rtmdet.py` or `cluster/submit_prediction_rtmdet_m.sh` with `CHECKPOINT` set
to its `model_best_map.pt`.

For RF-DETR, use `make_submission_rfdetr.py`, `cluster/submit_prediction_rfdetr_small.sh`, or
`cluster/submit_prediction_rfdetr_medium.sh`. The optional `CLASS_FUSION_ALPHA`, `CLASS_GATE=1`, and `TOPK`
environment variables reproduce a policy selected by the local diagnostic:

```bash
CHECKPOINT=runs/rfdetr_small_coco_full_e48/model_best_map.pt \
  ./cluster/submit_prediction_rfdetr_small.sh \
  --partition=researchshort --gres=gpu:a5000:1 --mem=16G --time=02:00:00
```

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
train_rtmdet.py       full-finetune CSPNeXt-m/RTMDet + evaluate + resumable checkpoints
train_rfdetr.py       full-finetune RF-DETR Small + shared classifier + resumable checkpoints
run_all.sh            the eight reference runs (calls train.py)
make_submission.py    write_submission(): predictions -> submission.zip
make_submission_rtmdet.py  RTMDet checkpoint -> submission.zip
make_submission_rfdetr.py  RF-DETR checkpoint -> submission.zip
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base, DINOv3 ViT-B/16
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
├── rtmdet.py         CSPNeXt-m + PAFPN + shared SepBN RTMDet head + image classifier
├── rfdetr.py         official RF-DETR Small + shared P4 image classifier
└── metrics.py        accuracy, macro-F1, COCO box AP
tests/                checks of sarbench/ (python -m pytest tests)
docs/DETAILS.md       model, training protocol, design notes, time and memory
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
