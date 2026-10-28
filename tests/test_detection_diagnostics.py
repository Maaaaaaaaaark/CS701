from diagnose_detection_errors import analyze_class, box_iou_xywh


def test_box_iou_xywh():
    assert box_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]) == 0
    assert round(box_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]), 4) == 0.3333


def test_analyze_class_separates_detection_failure_modes():
    images = {
        image_id: {"id": image_id, "file_name": f"{image_id}.png", "width": 100, "height": 100}
        for image_id in range(1, 6)
    }
    ground_truth = [
        {"id": image_id, "image_id": image_id, "category_id": 3, "bbox": [10, 10, 10, 10]}
        for image_id in range(1, 6)
    ]
    predictions = [
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 10, 10], "score": 0.9},
        {"image_id": 2, "category_id": 3, "bbox": [15, 10, 10, 10], "score": 0.9},
        {"image_id": 3, "category_id": 5, "bbox": [10, 10, 10, 10], "score": 0.9},
        {"image_id": 4, "category_id": 3, "bbox": [10, 10, 10, 10], "score": 0.01},
    ]

    result = analyze_class(
        ground_truth,
        predictions,
        images,
        category_id=3,
        category_names={3: "bridge", 5: "harbor"},
        score_threshold=0.05,
        localization_iou=0.10,
    )

    assert result["outcomes"] == {
        "matched": 1,
        "localization_error": 1,
        "classification_error": 1,
        "low_confidence": 1,
        "missed": 1,
    }
    assert result["wrong_categories"] == {"harbor": 1}
    assert result["matched_center_error_px_at_512"]["median"] == 0
    assert result["matched_pred_to_gt_area_ratio"]["median"] == 1
    assert round(result["localization_error_center_px_at_512"]["median"], 1) == 25.6
    assert result["localization_error_pred_to_gt_area_ratio"]["median"] == 1
