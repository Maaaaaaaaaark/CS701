#!/usr/bin/env python3
"""Run a trained official D-FINE model and create a validated CS701 ZIP."""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from make_submission import validate_submission, write_submission


class Images(Dataset):
    def __init__(self, root, images):
        self.root, self.images = Path(root), images
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((512, 512)),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        return self.transform(image), info["file_name"], info["width"], info["height"]


def image_class(labels, scores, pool):
    class_scores = []
    for category in range(9):
        selected = scores[labels == category]
        if selected.numel() == 0:
            class_scores.append(scores.new_tensor(0.0))
        elif pool == "max":
            class_scores.append(selected.max())
        else:
            class_scores.append(selected.topk(min(5, selected.numel())).values.sum())
    return int(torch.stack(class_scores).argmax())


def load_model(source, config, checkpoint, device):
    sys.path.insert(0, str(source.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config.resolve()), resume=str(checkpoint.resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(weights)
    return cfg.model.to(device).eval(), cfg.postprocessor.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data/val"))
    parser.add_argument("--source", type=Path, default=Path(".dfine/source"))
    parser.add_argument(
        "--config", type=Path, default=Path(".dfine/source/configs/cs701/dfine_l_sar512.yml")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/dfine_l_sar512_val.zip"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--class-pool", choices=("max", "top5-sum"), default="top5-sum")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    metadata = args.data / "images.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"missing {metadata}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    images = json.loads(metadata.read_text())["images"]
    loader = DataLoader(
        Images(args.data, images), batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, persistent_workers=args.workers > 0, pin_memory=True,
    )
    device = torch.device(args.device)
    model, postprocessor = load_model(args.source, args.config, args.checkpoint, device)
    predictions = {}
    for batch, names, widths, heights in tqdm(loader, desc="D-FINE-L inference"):
        batch = batch.to(device, non_blocking=True)
        sizes = torch.stack((widths, heights), dim=1).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch)
        results = postprocessor(outputs, sizes)
        for name, width, height, result in zip(names, widths, heights, results):
            labels, boxes, scores = result["labels"], result["boxes"], result["scores"]
            class_label = image_class(labels, scores, args.class_pool)
            keep = scores >= args.score_threshold
            labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
            if args.topk and scores.numel() > args.topk:
                order = scores.topk(args.topk).indices
                labels, boxes, scores = labels[order], boxes[order], scores[order]
            # write_submission expects boxes on a 512x512 canvas and rescales them.
            scale = boxes.new_tensor([512 / int(width), 512 / int(height)]).repeat(2)
            predictions[name] = (
                class_label,
                (boxes * scale).float().cpu().tolist(),
                labels.cpu().tolist(),
                scores.float().cpu().tolist(),
            )
    write_submission(images, predictions, out=args.output)
    rows, detections = validate_submission(args.output, images)
    print(f"Created {args.output.resolve()}")
    print(f"Classification rows: {rows}; detections: {detections}")


if __name__ == "__main__":
    main()
