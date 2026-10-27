import torch

from sarbench.rfdetr import RFDETR_IMAGE_MEAN, RFDETR_IMAGE_STD, SARDetailFusion


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
