import json
import zipfile

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
