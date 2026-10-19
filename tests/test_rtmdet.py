import torch

from sarbench.rtmdet import RTMDetMultiTask, dynamic_soft_label_assign


def test_dynamic_soft_label_assign_finds_a_positive():
    logits = torch.zeros(4, 2)
    boxes = torch.tensor([
        [0.0, 0.0, 20.0, 20.0],
        [5.0, 5.0, 15.0, 15.0],
        [30.0, 30.0, 50.0, 50.0],
        [40.0, 40.0, 60.0, 60.0],
    ])
    points = torch.tensor([[8.0, 8.0], [12.0, 12.0], [40.0, 40.0], [48.0, 48.0]])
    strides = torch.full((4,), 8.0)
    target = {'boxes': torch.tensor([[2.0, 2.0, 18.0, 18.0]]), 'labels': torch.tensor([1])}
    positive, matched, quality = dynamic_soft_label_assign(logits, boxes, points, strides, target)
    assert len(positive) >= 1
    assert (matched == 0).all()
    assert (quality > 0).all()


def test_rtmdet_m_is_fully_trainable_and_within_budget():
    model = RTMDetMultiTask(pretrained=False)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total == trainable
    assert total <= 40_000_000


def test_rtmdet_m_training_and_eval_forward():
    model = RTMDetMultiTask(pretrained=False)
    image = torch.zeros(1, 1, 128, 128)
    target = [{'boxes': torch.tensor([[16.0, 16.0, 48.0, 48.0]]),
               'labels': torch.tensor([1])}]
    logits, detections, losses = model.train()(image, target)
    assert logits.shape == (1, 9)
    assert detections == []
    assert set(losses) == {'loss_quality_focal', 'loss_giou'}
    assert all(loss.isfinite() for loss in losses.values())
    sum(losses.values()).backward()

    with torch.inference_mode():
        logits, detections, losses = model.eval()(image)
    assert logits.shape == (1, 9)
    assert len(detections) == 1
    assert set(detections[0]) == {'boxes', 'scores', 'labels'}
    assert losses == {}
