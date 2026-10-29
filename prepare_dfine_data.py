#!/usr/bin/env python3
"""Create a reproducible COCO train/holdout layout for the official D-FINE code."""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def stratified_val_ids(rows, fraction, seed):
    per_class = defaultdict(list)
    for row in rows:
        per_class[int(row["label"])].append(row)
    selected = set()
    for label, class_rows in per_class.items():
        shuffled = class_rows.copy()
        random.Random(seed + label).shuffle(shuffled)
        count = max(1, round(len(shuffled) * fraction))
        selected.update(int(row["image_id"]) for row in shuffled[:count])
    return selected


def link_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink() and os.readlink(destination) == target:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data"))
    parser.add_argument("--out", type=Path, default=Path(".dfine/data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    train_root = args.data / "train"
    with (train_root / "labels.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    coco = json.loads((train_root / "instances.json").read_text())
    images = {int(item["id"]): item for item in coco["images"]}
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[int(ann["image_id"])].append(ann)

    val_ids = stratified_val_ids(rows, args.val_fraction, args.split_seed)
    if args.limit is not None:
        rows = rows[: args.limit]
    row_by_id = {int(row["image_id"]): row for row in rows}
    missing = set(row_by_id) - set(images)
    if missing:
        raise ValueError(f"{len(missing)} labels have no COCO image record")

    categories = [{"id": i, "name": item["name"]} for i, item in enumerate(coco["categories"])]
    manifests = {}
    for split in ("train", "val"):
        want_val = split == "val"
        split_ids = [image_id for image_id in row_by_id if (image_id in val_ids) == want_val]
        split_images, split_annotations = [], []
        next_ann_id = 1
        for image_id in split_ids:
            source_info = images[image_id]
            row = row_by_id[image_id]
            expected_category = int(row["label"])
            file_name = Path(source_info["file_name"]).name
            source = train_root / source_info["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            link_image(source, args.out / split / "images" / file_name)
            split_images.append({**source_info, "file_name": file_name})
            image_anns = annotations[image_id]
            if not image_anns:
                raise ValueError(f"image {image_id} has no boxes")
            for ann in image_anns:
                category = int(ann["category_id"]) - 1
                if category != expected_category:
                    raise ValueError(
                        f"image {image_id}: label={expected_category}, annotation={category}"
                    )
                split_annotations.append({
                    **ann,
                    "id": next_ann_id,
                    "category_id": category,
                })
                next_ann_id += 1
        manifest = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": categories,
        }
        destination = args.out / split / "instances.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, separators=(",", ":")))
        manifests[split] = (len(split_images), len(split_annotations))

    split_record = {
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "train_image_ids": [i for i in row_by_id if i not in val_ids],
        "val_image_ids": [i for i in row_by_id if i in val_ids],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(split_record, indent=2))
    print(f"Prepared {args.out.resolve()}")
    for split, (image_count, box_count) in manifests.items():
        print(f"{split}: {image_count} images, {box_count} boxes")


if __name__ == "__main__":
    main()
