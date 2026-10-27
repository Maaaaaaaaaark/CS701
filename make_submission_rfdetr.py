"""Create a CS701 submission ZIP from a full-finetuned RF-DETR checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from make_submission import SubmissionImages, validate_submission, write_submission
from sarbench.rfdetr import (
    RFDETR_IMAGE_MEAN,
    RFDETR_IMAGE_STD,
    RFDETRMultiTask,
    apply_detection_policy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data/val"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "medium"), default="small")
    parser.add_argument(
        "--detail-fusion",
        action="store_true",
        help="construct the RF-DETR SAR detail branch used by the checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path("submissions/rfdetr_small_val.zip"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--class-fusion-alpha", type=float, default=0.0)
    parser.add_argument("--class-gate", action="store_true")
    parser.add_argument("--topk", type=int)
    args = parser.parse_args()
    if args.class_fusion_alpha < 0:
        parser.error("--class-fusion-alpha must be non-negative")
    if args.topk is not None and args.topk < 1:
        parser.error("--topk must be positive")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use the Slurm prediction script")
    metadata_path = args.data / "images.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing {metadata_path}; extract val.tar.gz first")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")

    images = json.loads(metadata_path.read_text())["images"]
    loader = DataLoader(
        SubmissionImages(
            args.data,
            images,
            mean=RFDETR_IMAGE_MEAN,
            std=RFDETR_IMAGE_STD,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    # setup_rfdetr.sh caches the official starter checkpoint inside the project;
    # constructing the identical architecture this way also avoids a separate
    # DINOv2-backbone download before the trained state is loaded.
    model = RFDETRMultiTask(
        pretrained=True,
        variant=args.variant,
        resolution=512,
        detail_fusion=args.detail_fusion,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()

    predictions = {}
    for batch, file_names in tqdm(loader, desc=f"Predicting validation images with RF-DETR {args.variant}"):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, outputs = model(batch.to(device, non_blocking=True))
        detections = model.detections(outputs)
        image_class_probs = logits.float().softmax(1)
        image_labels = logits.argmax(1).cpu().tolist()
        for file_name, image_label, image_probs, detection in zip(
            file_names, image_labels, image_class_probs, detections
        ):
            detection = apply_detection_policy(
                detection,
                image_probs,
                class_fusion_alpha=args.class_fusion_alpha,
                class_gate=args.class_gate,
                topk=args.topk,
            )
            predictions[file_name] = (
                image_label,
                detection["boxes"].float().cpu().tolist(),
                detection["labels"].cpu().tolist(),
                detection["scores"].float().cpu().tolist(),
            )

    write_submission(images, predictions, out=args.output)
    rows, detections = validate_submission(args.output, images)
    print(f"Created {args.output.resolve()}")
    print(f"Classification rows: {rows}; detections: {detections}")


if __name__ == "__main__":
    main()
