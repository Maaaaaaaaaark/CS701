"""Create a CS701 submission ZIP from a full-finetuned RF-DETR Small checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from make_submission import SubmissionImages, validate_submission, write_submission
from sarbench.rfdetr import RFDETR_IMAGE_MEAN, RFDETR_IMAGE_STD, RFDETRMultiTask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data/val"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/rfdetr_small_val.zip"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    model = RFDETRMultiTask(pretrained=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()

    predictions = {}
    for batch, file_names in tqdm(loader, desc="Predicting validation images with RF-DETR Small"):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, outputs = model(batch.to(device, non_blocking=True))
        detections = model.detections(outputs)
        image_labels = logits.argmax(1).cpu().tolist()
        for file_name, image_label, detection in zip(file_names, image_labels, detections):
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
