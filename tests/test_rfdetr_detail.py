import torch
from torch import nn

from sarbench.rfdetr import (
    RFDETR_IMAGE_MEAN,
    RFDETR_IMAGE_STD,
    RFDETRMultiTask,
    SARDetailFusion,
)


def test_sar_detail_fusion_shapes_budget_and_gradients():
    module = SARDetailFusion(hidden_dim=64, num_levels=2)
    gray = torch.rand(2, 1, 64, 64)
    mean = torch.tensor(RFDETR_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(RFDETR_IMAGE_STD).view(1, 3, 1, 1)
    normalized_rgb = (gray.repeat(1, 3, 1, 1) - mean) / std

    outputs = module(normalized_rgb, [(4, 4), (2, 2)])

    assert [tuple(output.shape) for output in outputs] == [(2, 64, 4, 4), (2, 64, 2, 2)]
    assert sum(parameter.numel() for parameter in module.parameters()) < 1_000_000
    sum(output.mean() for output in outputs).backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


class _NestedFeature:
    def __init__(self, tensors, mask=None, no_padding=True):
        self.tensors = tensors
        self.mask = mask
        self.no_padding = no_padding


class _UnitDetail(nn.Module):
    def forward(self, _images, target_shapes):
        return [torch.ones(1, 4, *shape) for shape in target_shapes]


def _routing_model(classifier_fusion):
    model = RFDETRMultiTask.__new__(RFDETRMultiTask)
    nn.Module.__init__(model)
    model.detail_fusion = _UnitDetail()
    model.detail_classifier_fusion = classifier_fusion
    model._detail_input = torch.zeros(1, 3, 8, 8)
    return model


def test_detection_only_routing_keeps_original_classifier_feature():
    original = _NestedFeature(torch.zeros(1, 4, 2, 2))
    model = _routing_model(classifier_fusion=False)

    fused, _, _ = model._fuse_and_capture_backbone_feature(None, None, ([original], [None], None))

    assert model._shared_feature is original
    assert torch.equal(fused[0].tensors, torch.ones_like(original.tensors))


def test_shared_routing_classifies_from_fused_feature():
    original = _NestedFeature(torch.zeros(1, 4, 2, 2))
    model = _routing_model(classifier_fusion=True)

    fused, _, _ = model._fuse_and_capture_backbone_feature(None, None, ([original], [None], None))

    assert model._shared_feature is fused[0]
    assert torch.equal(model._shared_feature.tensors, torch.ones_like(original.tensors))
