import pytest
import torch

from sarbench.rfdetr import RFDETRMultiTask


class _Criterion:
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou_0": 2.0,
    }

    def __call__(self, _outputs, _targets):
        return {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(3.0),
            "loss_giou_0": torch.tensor(4.0),
            "cardinality_error": torch.tensor(99.0),
        }


def _model_with_fake_criterion():
    model = RFDETRMultiTask.__new__(RFDETRMultiTask)
    torch.nn.Module.__init__(model)
    model.criterion = _Criterion()
    return model


def test_box_loss_scale_covers_main_and_auxiliary_localization_losses():
    total, parts = _model_with_fake_criterion().detection_losses(
        {}, [], box_loss_scale=1.5
    )

    assert parts["loss_ce"].item() == pytest.approx(2.0)
    assert parts["loss_bbox"].item() == pytest.approx(22.5)
    assert parts["loss_giou_0"].item() == pytest.approx(12.0)
    assert total.item() == pytest.approx(36.5)
    assert "cardinality_error" not in parts


def test_box_loss_scale_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        _model_with_fake_criterion().detection_losses({}, [], box_loss_scale=0.0)
