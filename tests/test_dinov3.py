"""Offline checks for the DINOv3 configuration used by run_dinov3.sh."""
import torch

from sarbench.adapters import LoRA, add_adapters
from sarbench.backbones import build_backbone
from sarbench.model import MultiTaskModel


def test_dinov3_tokens_and_attention_layout():
    backbone = build_backbone('dinov3', pretrained=False).eval()
    assert len(backbone.blocks) == 12
    assert backbone.vit.num_prefix_tokens == 5
    with torch.inference_mode():
        tokens = backbone(torch.zeros(1, 1, 512, 512))
    assert tokens.shape == (1, 1024, 768)


def test_dinov3_hybrid_lora_stays_within_course_budget():
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora', rank=32, alpha=64, unfreeze_last_blocks=2)
    model = MultiTaskModel(backbone, task_routing=False)

    assert isinstance(backbone.blocks[0].attn.qkv, LoRA)
    assert not backbone.blocks[0].attn.qkv.base.weight.requires_grad
    assert backbone.blocks[-1].attn.qkv.base.weight.requires_grad

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total == 107_130_649
    assert trainable == 36_058_393
    assert total <= 130_000_000
    assert trainable <= 40_000_000


def test_dinov3_detail_stem_stays_within_course_budget_and_runs_end_to_end():
    backbone = build_backbone('dinov3', pretrained=False)
    add_adapters(backbone, 'lora', rank=32, alpha=64, unfreeze_last_blocks=2)
    model = MultiTaskModel(backbone, task_routing=False, detail_stem=True).eval()

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert total == 107_281_881
    assert trainable == 36_209_625
    assert total <= 130_000_000
    assert trainable <= 40_000_000

    with torch.inference_mode():
        logits, detections, losses = model(torch.zeros(1, 1, 512, 512))
    assert logits.shape == (1, 9)
    assert len(detections) == 1
    assert set(detections[0]) == {'boxes', 'labels', 'scores'}
    assert losses == {}
