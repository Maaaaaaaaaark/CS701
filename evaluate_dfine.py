#!/usr/bin/env python3
"""Evaluate D-FINE detection and detector-derived image classification on the fixed holdout."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from make_submission_dfine import Images, image_class, load_model
from sarbench.metrics import classification_metrics, detection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data"))
    parser.add_argument("--source", type=Path, default=Path(".dfine/source"))
    parser.add_argument(
        "--config", type=Path, default=Path(".dfine/source/configs/cs701/dfine_l_sar512.yml")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path(".dfine/data/split.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    val_ids = set(split["val_image_ids"])
    coco_path = args.data / "train" / "instances.json"
    coco = json.loads(coco_path.read_text())
    images = [item for item in coco["images"] if int(item["id"]) in val_ids]
    with (args.data / "train" / "labels.csv").open(newline="") as handle:
        true_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(handle)}

    loader = DataLoader(
        Images(args.data / "train", images), batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, persistent_workers=args.workers > 0, pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, postprocessor = load_model(args.source, args.config, args.checkpoint, device)
    classifications = {"max": [], "top5-sum": []}
    y_true, predictions, ordered_ids = [], [], []
    with torch.inference_mode():
        for batch, names, widths, heights in tqdm(loader, desc="Evaluating D-FINE-L"):
            batch = batch.to(device, non_blocking=True)
            sizes = torch.stack((widths, heights), dim=1).to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(batch)
            results = postprocessor(outputs, sizes)
            for name, result in zip(names, results):
                image_id = int(Path(name).stem)
                ordered_ids.append(image_id)
                y_true.append(true_by_id[image_id])
                labels, boxes, scores = result["labels"], result["boxes"], result["scores"]
                for pool in classifications:
                    classifications[pool].append(image_class(labels, scores, pool))
                keep = scores.topk(min(100, scores.numel())).indices
                for label, box, score in zip(labels[keep], boxes[keep], scores[keep]):
                    x1, y1, x2, y2 = box.float().cpu().tolist()
                    if x2 > x1 and y2 > y1:
                        predictions.append({
                            "image_id": image_id,
                            "category_id": int(label) + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        })

    result = {"classification": {}}
    for pool, y_pred in classifications.items():
        result["classification"][pool] = classification_metrics(y_true, y_pred)
        metrics = result["classification"][pool]
        print(
            f"{pool:10s} accuracy {metrics['accuracy'] * 100:6.2f} "
            f"macro-F1 {metrics['macro_f1'] * 100:6.2f}"
        )
    result["detection"] = detection_metrics(coco_path, predictions, ordered_ids)
    det = result["detection"]
    print(f"detection  mAP {det['mAP'] * 100:.2f} AP50 {det['AP50'] * 100:.2f} AP75 {det['AP75'] * 100:.2f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print("Saved", args.output.resolve())


if __name__ == "__main__":
    main()
