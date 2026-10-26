import json
import zipfile

import pytest

from make_submission import validate_submission, write_submission


def test_submission_schema_and_label_numbering(tmp_path):
    images = [{"id": 7, "file_name": "images/a.png", "width": 256, "height": 128}]
    predictions = {"images/a.png": (2, [[10, 20, 30, 60]], [4], [0.75])}
    output = tmp_path / "submission.zip"
    write_submission(images, predictions, out=output)

    assert validate_submission(output, images) == (1, 1)
    with zipfile.ZipFile(output) as archive:
        classification = archive.read("classification.csv").decode()
        detections = json.loads(archive.read("detection.json"))
    assert classification == "image_id,label\n7,2\n"
    assert detections == [{"image_id": 7, "category_id": 5,
                           "bbox": [5.0, 5.0, 10.0, 10.0], "score": 0.75}]


def test_submission_clips_boxes_and_drops_degenerate_detections(tmp_path):
    images = [{"id": 7, "file_name": "images/a.png", "width": 256, "height": 128}]
    predictions = {
        "images/a.png": (
            2,
            [[-10, -20, 520, 600], [498, 512, 509, 512], [float("nan"), 1, 2, 3]],
            [4, 4, 4],
            [0.9, 0.8, 0.7],
        )
    }
    output = tmp_path / "submission.zip"

    write_submission(images, predictions, out=output)

    assert validate_submission(output, images) == (1, 1)
    with zipfile.ZipFile(output) as archive:
        detections = json.loads(archive.read("detection.json"))
    assert detections == [
        {"image_id": 7, "category_id": 5, "bbox": [0.0, 0.0, 256.0, 128.0], "score": 0.9}
    ]


def test_validation_rejects_nonpositive_box_size(tmp_path):
    images = [{"id": 7, "file_name": "images/a.png", "width": 256, "height": 128}]
    output = tmp_path / "submission.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("classification.csv", "image_id,label\n7,2\n")
        archive.writestr(
            "detection.json",
            json.dumps([{"image_id": 7, "category_id": 5, "bbox": [1, 2, 3, 0], "score": 0.9}]),
        )

    with pytest.raises(ValueError, match="invalid detection"):
        validate_submission(output, images)
