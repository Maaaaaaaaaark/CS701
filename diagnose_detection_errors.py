"""Diagnose per-class RF-DETR localization, confidence, and miss patterns from saved predictions."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cs701-sar-course-data"))
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="named run directory containing split.json and predictions_val.json; repeat to compare runs",
    )
    parser.add_argument("--classes", nargs="+", default=("bridge", "playground"))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--localization-iou", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rfdetr_bridge_playground_diagnostic.json"),
    )
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if not 0 <= args.localization_iou < 0.5:
        parser.error("--localization-iou must be in [0, 0.5)")
    return args


def parse_run(value):
    if "=" not in value:
        raise ValueError(f"run must use NAME=RUN_DIR syntax: {value}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"run must use NAME=RUN_DIR syntax: {value}")
    return name, Path(path)


def box_iou_xywh(first, second):
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    intersection_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_w * intersection_h
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - intersection
    return intersection / union if union > 0 else 0.0


def summary(values):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return {"count": 0, "mean": None, "q10": None, "median": None, "q90": None}

    def percentile(fraction):
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        return values[lower] * (upper - position) + values[upper] * (position - lower)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "q10": percentile(0.10),
        "median": percentile(0.50),
        "q90": percentile(0.90),
    }


def annotation_statistics(annotations, images):
    widths, heights, areas, aspects = [], [], [], []
    boxes_per_image = Counter()
    border_touches = 0
    for annotation in annotations:
        image = images[annotation["image_id"]]
        x, y, width, height = map(float, annotation["bbox"])
        scaled_width = width * 512 / image["width"]
        scaled_height = height * 512 / image["height"]
        widths.append(scaled_width)
        heights.append(scaled_height)
        areas.append(scaled_width * scaled_height)
        aspects.append(max(scaled_width / max(scaled_height, 1e-9), scaled_height / max(scaled_width, 1e-9)))
        boxes_per_image[annotation["image_id"]] += 1
        tolerance_x = image["width"] / 512
        tolerance_y = image["height"] / 512
        if (
            x <= tolerance_x
            or y <= tolerance_y
            or x + width >= image["width"] - tolerance_x
            or y + height >= image["height"] - tolerance_y
        ):
            border_touches += 1
    return {
        "boxes": len(annotations),
        "images": len(boxes_per_image),
        "width_px_at_512": summary(widths),
        "height_px_at_512": summary(heights),
        "area_px2_at_512": summary(areas),
        "symmetric_aspect_ratio": summary(aspects),
        "boxes_per_image": summary(boxes_per_image.values()),
        "border_touch_rate": border_touches / len(annotations) if annotations else 0.0,
    }


def analyze_class(
    ground_truth,
    predictions,
    images,
    category_id,
    category_names,
    score_threshold=0.05,
    localization_iou=0.10,
):
    gt_by_image = defaultdict(list)
    predictions_by_image = defaultdict(list)
    for annotation in ground_truth:
        gt_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        predictions_by_image[prediction["image_id"]].append(prediction)

    outcomes = Counter()
    wrong_categories = Counter()
    best_same_ious, best_same_scores = [], []
    matched_ious, center_errors, edge_errors, width_ratios, height_ratios, area_ratios = (
        [] for _ in range(6)
    )
    localization_center_errors, localization_edge_errors = [], []
    localization_width_ratios, localization_height_ratios, localization_area_ratios = [], [], []
    examples = []

    for annotation in ground_truth:
        image_id = annotation["image_id"]
        same = [
            prediction
            for prediction in predictions_by_image[image_id]
            if int(prediction["category_id"]) == category_id
        ]
        wrong = [
            prediction
            for prediction in predictions_by_image[image_id]
            if int(prediction["category_id"]) != category_id
            and float(prediction["score"]) >= score_threshold
        ]
        same_with_iou = [(box_iou_xywh(annotation["bbox"], prediction["bbox"]), prediction) for prediction in same]
        scored_same = [item for item in same_with_iou if float(item[1]["score"]) >= score_threshold]
        wrong_with_iou = [(box_iou_xywh(annotation["bbox"], prediction["bbox"]), prediction) for prediction in wrong]
        best_any = max(same_with_iou, key=lambda item: item[0], default=(0.0, None))
        best_scored = max(scored_same, key=lambda item: item[0], default=(0.0, None))
        best_wrong = max(wrong_with_iou, key=lambda item: item[0], default=(0.0, None))
        if best_any[1] is not None:
            best_same_ious.append(best_any[0])
            best_same_scores.append(float(best_any[1]["score"]))

        selected_iou, selected_prediction = best_scored
        if selected_iou >= 0.5:
            outcome = "matched"
        elif best_wrong[0] >= 0.5:
            outcome = "classification_error"
            selected_iou, selected_prediction = best_wrong
            wrong_categories[category_names.get(int(selected_prediction["category_id"]), str(selected_prediction["category_id"]))] += 1
        elif best_any[0] >= 0.5:
            outcome = "low_confidence"
            selected_iou, selected_prediction = best_any
        elif selected_iou >= localization_iou:
            outcome = "localization_error"
        else:
            outcome = "missed"
            selected_iou, selected_prediction = best_any
        outcomes[outcome] += 1

        if outcome in {"matched", "localization_error"}:
            gx, gy, gw, gh = map(float, annotation["bbox"])
            px, py, pw, ph = map(float, selected_prediction["bbox"])
            image = images[image_id]
            scale_x, scale_y = 512 / image["width"], 512 / image["height"]
            center_error = math.hypot(
                ((px + pw / 2) - (gx + gw / 2)) * scale_x,
                ((py + ph / 2) - (gy + gh / 2)) * scale_y,
            )
            edge_error = (
                abs(px - gx) * scale_x
                + abs(py - gy) * scale_y
                + abs((px + pw) - (gx + gw)) * scale_x
                + abs((py + ph) - (gy + gh)) * scale_y
            ) / 4
            width_ratio = pw / max(gw, 1e-9)
            height_ratio = ph / max(gh, 1e-9)
            area_ratio = (pw * ph) / max(gw * gh, 1e-9)
            if outcome == "matched":
                matched_ious.append(selected_iou)
                center_errors.append(center_error)
                edge_errors.append(edge_error)
                width_ratios.append(width_ratio)
                height_ratios.append(height_ratio)
                area_ratios.append(area_ratio)
            else:
                localization_center_errors.append(center_error)
                localization_edge_errors.append(edge_error)
                localization_width_ratios.append(width_ratio)
                localization_height_ratios.append(height_ratio)
                localization_area_ratios.append(area_ratio)

        examples.append(
            {
                "image_id": image_id,
                "file_name": images[image_id]["file_name"],
                "outcome": outcome,
                "gt_bbox": annotation["bbox"],
                "prediction_bbox": selected_prediction["bbox"] if selected_prediction else None,
                "prediction_category": (
                    category_names.get(int(selected_prediction["category_id"]), str(selected_prediction["category_id"]))
                    if selected_prediction
                    else None
                ),
                "score": float(selected_prediction["score"]) if selected_prediction else None,
                "iou": selected_iou,
            }
        )

    prediction_overlap = Counter()
    target_predictions = [
        prediction
        for prediction in predictions
        if int(prediction["category_id"]) == category_id and float(prediction["score"]) >= score_threshold
    ]
    for prediction in target_predictions:
        same_gt = gt_by_image[prediction["image_id"]]
        maximum_iou = max(
            (box_iou_xywh(annotation["bbox"], prediction["bbox"]) for annotation in same_gt),
            default=0.0,
        )
        if maximum_iou >= 0.5:
            prediction_overlap["good_overlap"] += 1
        elif maximum_iou >= localization_iou:
            prediction_overlap["partial_overlap"] += 1
        else:
            prediction_overlap["background_or_wrong_image"] += 1

    total = len(ground_truth)
    rates = {name: count / total if total else 0.0 for name, count in outcomes.items()}
    examples.sort(key=lambda item: (item["outcome"] == "matched", item["iou"]))
    return {
        "score_threshold": score_threshold,
        "localization_iou": localization_iou,
        "outcomes": dict(outcomes),
        "outcome_rates": rates,
        "wrong_categories": dict(wrong_categories.most_common()),
        "best_same_class_iou": summary(best_same_ious),
        "best_same_class_score": summary(best_same_scores),
        "matched_iou": summary(matched_ious),
        "matched_center_error_px_at_512": summary(center_errors),
        "matched_mean_edge_error_px_at_512": summary(edge_errors),
        "matched_pred_to_gt_width_ratio": summary(width_ratios),
        "matched_pred_to_gt_height_ratio": summary(height_ratios),
        "matched_pred_to_gt_area_ratio": summary(area_ratios),
        "localization_error_center_px_at_512": summary(localization_center_errors),
        "localization_error_mean_edge_px_at_512": summary(localization_edge_errors),
        "localization_error_pred_to_gt_width_ratio": summary(localization_width_ratios),
        "localization_error_pred_to_gt_height_ratio": summary(localization_height_ratios),
        "localization_error_pred_to_gt_area_ratio": summary(localization_area_ratios),
        "predictions_above_threshold": len(target_predictions),
        "prediction_overlap": dict(prediction_overlap),
        "worst_examples": examples[:20],
    }


def print_summary(run_name, class_name, result):
    counts = result["outcomes"]
    total = sum(counts.values())
    rate = lambda name: 100 * counts.get(name, 0) / total if total else 0
    print(
        f"{run_name:14s} {class_name:12s} GT {total:4d} "
        f"matched {rate('matched'):6.2f}% loc {rate('localization_error'):6.2f}% "
        f"class {rate('classification_error'):6.2f}% lowconf {rate('low_confidence'):6.2f}% "
        f"missed {rate('missed'):6.2f}%"
    )
    print(
        f"{'':27s} median IoU {result['matched_iou']['median'] or 0:.3f} "
        f"center err {result['matched_center_error_px_at_512']['median'] or 0:.2f}px "
        f"edge err {result['matched_mean_edge_error_px_at_512']['median'] or 0:.2f}px "
        f"area ratio {result['matched_pred_to_gt_area_ratio']['median'] or 0:.2f}"
    )
    print(
        f"{'':27s} localization failures: center "
        f"{result['localization_error_center_px_at_512']['median'] or 0:.2f}px, "
        f"edge {result['localization_error_mean_edge_px_at_512']['median'] or 0:.2f}px, "
        f"area ratio {result['localization_error_pred_to_gt_area_ratio']['median'] or 0:.2f}"
    )


def main():
    args = parse_args()
    annotation_path = args.data / "train" / "instances.json"
    coco = json.loads(annotation_path.read_text())
    images = {int(image["id"]): image for image in coco["images"]}
    categories = {int(category["id"]): category["name"] for category in coco["categories"]}
    ids_by_name = {name: category_id for category_id, name in categories.items()}
    missing_classes = sorted(set(args.classes) - ids_by_name.keys())
    if missing_classes:
        raise ValueError(f"unknown classes: {', '.join(missing_classes)}")

    parsed_runs = [parse_run(run) for run in args.run]
    split_ids = None
    run_payloads = {}
    for name, run_dir in parsed_runs:
        predictions_path = run_dir / "predictions_val.json"
        split_path = run_dir / "split.json"
        if not predictions_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(f"{run_dir} must contain predictions_val.json and split.json")
        current_ids = {int(image_id) for image_id in json.loads(split_path.read_text())["val_image_ids"]}
        if split_ids is None:
            split_ids = current_ids
        elif current_ids != split_ids:
            raise ValueError("all runs must use the same validation image ids")
        run_payloads[name] = json.loads(predictions_path.read_text())

    output = {
        "data": str(args.data),
        "score_threshold": args.score_threshold,
        "classes": {},
        "runs": {},
    }
    for class_name in args.classes:
        category_id = ids_by_name[class_name]
        class_gt = [
            annotation
            for annotation in coco["annotations"]
            if int(annotation["image_id"]) in split_ids and int(annotation["category_id"]) == category_id
        ]
        stats = annotation_statistics(class_gt, images)
        output["classes"][class_name] = {"category_id": category_id, "annotation_statistics": stats}
        print(
            f"{class_name:12s} annotations: {stats['boxes']} boxes in {stats['images']} images; "
            f"median size {stats['width_px_at_512']['median']:.2f}x{stats['height_px_at_512']['median']:.2f}px; "
            f"median aspect {stats['symmetric_aspect_ratio']['median']:.2f}; "
            f"border touch {stats['border_touch_rate'] * 100:.2f}%"
        )

    for run_name, predictions in run_payloads.items():
        output["runs"][run_name] = {}
        for class_name in args.classes:
            category_id = ids_by_name[class_name]
            class_gt = [
                annotation
                for annotation in coco["annotations"]
                if int(annotation["image_id"]) in split_ids and int(annotation["category_id"]) == category_id
            ]
            result = analyze_class(
                class_gt,
                predictions,
                images,
                category_id,
                categories,
                args.score_threshold,
                args.localization_iou,
            )
            output["runs"][run_name][class_name] = result
            print_summary(run_name, class_name, result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
