import torch

from sarbench.rfdetr import apply_detection_policy


def test_detection_policy_filters_fuses_sorts_and_prunes():
    detection = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 2.0],
            ]
        ),
        "labels": torch.tensor([0, 1, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.99]),
    }
    class_probs = torch.tensor([0.1, 0.9])

    result = apply_detection_policy(
        detection,
        class_probs,
        class_fusion_alpha=1.0,
        topk=1,
    )

    assert result["labels"].tolist() == [1]
    torch.testing.assert_close(result["scores"], torch.tensor([0.72]))
    torch.testing.assert_close(result["boxes"], detection["boxes"][[1]])


def test_detection_policy_hard_gate_keeps_only_image_class():
    detection = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 3.0, 3.0]]),
        "labels": torch.tensor([0, 1]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    result = apply_detection_policy(
        detection,
        torch.tensor([0.1, 0.9]),
        class_gate=True,
    )

    assert result["labels"].tolist() == [1]
    torch.testing.assert_close(result["scores"], torch.tensor([0.8]))
