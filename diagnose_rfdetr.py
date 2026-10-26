"""Evaluate RF-DETR classification-aware detection policies on the local holdout."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.rfdetr import (
    RFDETR_IMAGE_MEAN,
    RFDETR_IMAGE_STD,
    RFDETRMultiTask,
    apply_detection_policy,
)


F1_REFERENCE = 0.9331
MAP_REFERENCE = 0.2613
POLICIES = (
    {"name": "baseline"},
    {"name": "top100", "topk": 100},
    {"name": "fusion_a025", "class_fusion_alpha": 0.25},
    {"name": "fusion_a050", "class_fusion_alpha": 0.5},
    {"name": "fusion_a100", "class_fusion_alpha": 1.0},
    {"name": "fusion_a050_top100", "class_fusion_alpha": 0.5, "topk": 100},
    {"name": "class_gate", "class_gate": True},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "medium"), default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rfdetr_postprocess_diagnostic.json"),
    )
    return parser.parse_args()


def delta_m(macro_f1, mean_ap):
    return 50 * ((macro_f1 - F1_REFERENCE) / F1_REFERENCE + (mean_ap - MAP_REFERENCE) / MAP_REFERENCE)


@torch.inference_mode()
def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires a CUDA GPU")

    dataset = SARMultiTask(
        args.data,
        "val",
        train=False,
        val_fraction=0.1,
        split_seed=0,
        mean=RFDETR_IMAGE_MEAN,
        std=RFDETR_IMAGE_STD,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        collate_fn=collate,
    )
    model = RFDETRMultiTask(pretrained=True, variant=args.variant, resolution=512)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.cuda().eval()

    expected_labels = []
    predicted_labels = []
    records = []
    for images, expected, targets in tqdm(loader, desc="RF-DETR diagnostic inference"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, outputs = model(images.cuda(non_blocking=True))
        probabilities = logits.float().softmax(1)
        detections = model.detections(outputs)
        expected_labels.extend(expected.tolist())
        predicted_labels.extend(logits.argmax(1).cpu().tolist())
        for target, probs, detection in zip(targets, probabilities, detections):
            records.append(
                {
                    "image_id": target["image_id"],
                    "orig_size": target["orig_size"],
                    "class_probs": probs.cpu(),
                    "detection": {
                        name: value.float().cpu() if name != "labels" else value.cpu()
                        for name, value in detection.items()
                    },
                }
            )

    classification = classification_metrics(expected_labels, predicted_labels)
    results = []
    image_ids = [record["image_id"] for record in records]
    for specification in POLICIES:
        policy = {key: value for key, value in specification.items() if key != "name"}
        predictions = []
        for record in records:
            detection = apply_detection_policy(
                record["detection"], record["class_probs"], **policy
            )
            boxes = to_original_xywh(detection["boxes"], record["orig_size"]).tolist()
            predictions.extend(
                {
                    "image_id": record["image_id"],
                    "category_id": int(category) + 1,
                    "bbox": box,
                    "score": float(score),
                }
                for box, category, score in zip(
                    boxes, detection["labels"].tolist(), detection["scores"].tolist()
                )
            )
        detection_result = detection_metrics(dataset.gt_json, predictions, image_ids)
        result = {
            "name": specification["name"],
            "policy": policy,
            "detections": len(predictions),
            **classification,
            **detection_result,
        }
        result["delta_m"] = delta_m(result["macro_f1"], result["mAP"])
        results.append(result)
        print(
            f"{result['name']:22s} detections {result['detections']:6d} "
            f"F1 {result['macro_f1'] * 100:6.2f} mAP {result['mAP'] * 100:6.2f} "
            f"AP50 {result['AP50'] * 100:6.2f} AP75 {result['AP75'] * 100:6.2f} "
            f"delta_m {result['delta_m']:6.2f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "variant": args.variant,
                "results": results,
                "best_policy": max(results, key=lambda result: result["mAP"])["name"],
            },
            indent=2,
        )
    )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
