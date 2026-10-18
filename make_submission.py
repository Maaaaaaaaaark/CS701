"""Run a trained model on the released validation images and create a Codabench submission ZIP."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import MEAN, STD
from sarbench.model import MultiTaskModel


def write_submission(images, preds, size=512, out="submission.zip"):
    """Write zero-based class predictions and input-sized XYXY boxes to Codabench format."""
    rows, dets = ["image_id,label"], []
    for im in images:
        label, boxes, classes, scores = preds[im["file_name"]]
        rows.append(f"{im['id']},{int(label)}")
        sx, sy = im["width"] / size, im["height"] / size
        for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores):
            box = [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy]
            dets.append({"image_id": im["id"], "category_id": int(c) + 1,
                         "bbox": [float(v) for v in box], "score": float(s)})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("classification.csv", "\n".join(rows) + "\n")
        z.writestr("detection.json", json.dumps(dets))


def validate_submission(path, images):
    """Check the archive against the released validation set and submission schema."""
    expected_ids = {int(image["id"]) for image in images}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"classification.csv", "detection.json"}:
            raise ValueError("submission ZIP must contain only classification.csv and detection.json at its root")
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open("classification.csv"), encoding="utf-8")))
        detections = json.loads(archive.read("detection.json"))
    row_ids = [int(row["image_id"]) for row in rows]
    if len(row_ids) != len(expected_ids) or set(row_ids) != expected_ids:
        raise ValueError("classification.csv does not contain exactly one row for every validation image")
    if any(not 0 <= int(row["label"]) <= 8 for row in rows):
        raise ValueError("classification labels must be in 0..8")
    for detection in detections:
        if (int(detection["image_id"]) not in expected_ids
                or not 1 <= int(detection["category_id"]) <= 9
                or len(detection["bbox"]) != 4
                or not 0 <= float(detection["score"]) <= 1):
            raise ValueError(f"invalid detection: {detection}")
    return len(rows), len(detections)


class SubmissionImages(Dataset):
    def __init__(self, root, images, size=512):
        self.root, self.images = Path(root), images
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((size, size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file_name = self.images[index]["file_name"]
        image = Image.open(self.root / file_name).convert("L")
        return self.transform(image), file_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data/val"),
                        help="extracted split containing images.json and images/")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/dinov3_pretrained_lora_r32_u2_detail/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("submissions/dinov3_detail_val.zip"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cascade", action="store_true",
                        help="load the compact Cascade R-CNN detection head")
    parser.add_argument("--cascade-hidden-dim", type=int, default=384,
                        help="hidden width used by the trained compact Cascade head")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run this through the Slurm GPU script")
    metadata_path = args.data / "images.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing {metadata_path}; extract val.tar.gz first")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    images = json.loads(metadata_path.read_text())["images"]
    loader = DataLoader(SubmissionImages(args.data, images), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        persistent_workers=args.workers > 0)

    backbone = build_backbone("dinov3", pretrained=False)
    add_adapters(backbone, "lora", rank=32, alpha=64, unfreeze_last_blocks=2)
    model = MultiTaskModel(backbone, task_routing=False, detail_stem=True, cascade=args.cascade,
                           cascade_hidden_dim=args.cascade_hidden_dim)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    predictions = {}
    for batch, file_names in tqdm(loader, desc="Predicting validation images"):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, detections, _ = model(batch.to(device, non_blocking=True))
        class_labels = logits.argmax(dim=1).cpu().tolist()
        for file_name, class_label, detection in zip(file_names, class_labels, detections):
            predictions[file_name] = (
                class_label,
                detection["boxes"].cpu().tolist(),
                (detection["labels"] - 1).cpu().tolist(),
                detection["scores"].cpu().tolist(),
            )

    write_submission(images, predictions, out=args.output)
    rows, detections = validate_submission(args.output, images)
    print(f"Created {args.output.resolve()}")
    print(f"Classification rows: {rows}; detections: {detections}")


if __name__ == "__main__":
    main()
