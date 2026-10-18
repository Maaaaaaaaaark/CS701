import torch

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.model import CompactCascadeRoIHeads, MultiTaskModel


def build_cascade_model():
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora', rank=32, alpha=64, unfreeze_last_blocks=2)
    return MultiTaskModel(backbone, task_routing=False, detail_stem=True, cascade=True)


def test_compact_cascade_stays_within_course_budget():
    model = build_cascade_model()
    assert isinstance(model.roi_heads, CompactCascadeRoIHeads)
    assert len(model.roi_heads.stages) == 3

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total == 108_246_481
    assert trainable == 37_174_225
    assert total <= 130_000_000
    assert trainable <= 40_000_000


def test_wide_compact_cascade_uses_remaining_trainable_budget():
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora', rank=32, alpha=64, unfreeze_last_blocks=2)
    model = MultiTaskModel(
        backbone, task_routing=False, detail_stem=True, cascade=True, cascade_hidden_dim=448)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total == 110_817_745
    assert trainable == 39_745_489
    assert total <= 130_000_000
    assert trainable <= 40_000_000


def test_compact_cascade_eval_forward():
    model = build_cascade_model().eval()
    with torch.inference_mode():
        logits, detections, losses = model(torch.zeros(1, 1, 512, 512))
    assert logits.shape == (1, 9)
    assert len(detections) == 1
    assert set(detections[0]) == {'boxes', 'labels', 'scores'}
    assert losses == {}


def test_compact_cascade_training_losses_backpropagate():
    cascade = CompactCascadeRoIHeads(num_classes=2).train()
    features = {
        '0': torch.randn(1, 256, 16, 16, requires_grad=True),
        '1': torch.randn(1, 256, 8, 8, requires_grad=True),
        '2': torch.randn(1, 256, 4, 4, requires_grad=True),
        '3': torch.randn(1, 256, 2, 2, requires_grad=True),
    }
    proposals = [torch.tensor([[5.0, 5.0, 24.0, 24.0], [30.0, 30.0, 50.0, 50.0]])]
    targets = [{'boxes': torch.tensor([[6.0, 6.0, 25.0, 25.0]]),
                'labels': torch.tensor([1], dtype=torch.int64)}]

    detections, losses = cascade(features, proposals, [(64, 64)], targets)

    assert detections == []
    assert set(losses) == {
        'loss_classifier_s1', 'loss_box_reg_s1',
        'loss_classifier_s2', 'loss_box_reg_s2',
        'loss_classifier_s3', 'loss_box_reg_s3',
    }
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
