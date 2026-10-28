import torch

from sarbench.rfdetr import RFDETR_IMAGE_MEAN, RFDETR_IMAGE_STD, SmallObjectHead


def normalized_rgb(batch=2, size=64):
    gray = torch.rand(batch, 1, size, size)
    mean = torch.tensor(RFDETR_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(RFDETR_IMAGE_STD).view(1, 3, 1, 1)
    return (gray.repeat(1, 3, 1, 1) - mean) / std


def test_small_object_head_shapes_losses_and_detections():
    head = SmallObjectHead(
        num_classes=3,
        input_size=64,
        max_object_size=16,
        score_threshold=0.005,
        detections_per_image=20,
    )
    outputs = head(normalized_rgb())

    assert outputs["class_logits"].shape == (2, 3, 16, 16)
    assert outputs["box_ltrb"].shape == (2, 4, 16, 16)

    targets = [
        {
            "boxes": torch.tensor([[0.50, 0.50, 0.125, 0.125], [0.25, 0.25, 0.50, 0.50]]),
            "labels": torch.tensor([1, 2]),
        },
        {
            "boxes": torch.tensor([[0.30, 0.60, 0.10, 0.15]]),
            "labels": torch.tensor([0]),
        },
    ]
    losses = head.losses(outputs, targets)

    assert set(losses) == {"loss_small_cls", "loss_small_l1", "loss_small_giou"}
    assert all(torch.isfinite(loss) and loss >= 0 for loss in losses.values())
    sum(losses.values()).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())

    detections = head.detections(outputs, size=64)
    assert len(detections) == 2
    for detection in detections:
        assert set(detection) == {"boxes", "labels", "scores"}
        assert len(detection["boxes"]) <= 20
        assert (detection["boxes"] >= 0).all() and (detection["boxes"] <= 64).all()


def test_small_object_head_has_small_parameter_cost():
    head = SmallObjectHead()
    assert sum(parameter.numel() for parameter in head.parameters()) < 500_000
