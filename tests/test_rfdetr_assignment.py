import pytest
import torch


pytest.importorskip("rfdetr")
pytest.importorskip("torch_linear_assignment")

from rfdetr.models import _assignment

from sarbench.rfdetr import _install_exact_size_assignment_workaround


def test_exact_size_assignment_handles_uneven_target_counts():
    _install_exact_size_assignment_workaround()

    sizes = [4, 29, 1, 2]
    group_detr = 13
    group_width = 30
    total_targets = sum(sizes)
    costs = [
        torch.rand(group_width * group_detr, total_targets),
        torch.rand(group_width * group_detr, total_targets),
    ]

    assignments = _assignment.assign_many_bucketed(costs, sizes, group_detr)

    assert len(assignments) == len(costs)
    for layer_assignments in assignments:
        assert len(layer_assignments) == len(sizes)
        for (rows, columns), size in zip(layer_assignments, sizes):
            assert rows.shape == (group_detr * size,)
            assert columns.shape == (group_detr * size,)
            assert columns.min().item() == 0
            assert columns.max().item() == size - 1
