import json

import pytest

from train_rfdetr import class_aware_sampler


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)


def test_class_aware_sampler_upweights_selected_classes_and_keeps_epoch_length(tmp_path):
    (tmp_path / "classes.json").write_text(
        json.dumps(
            [
                {"label": 0, "name": "aircraft"},
                {"label": 2, "name": "bridge"},
                {"label": 6, "name": "playground"},
            ]
        )
    )
    dataset = _Rows(
        [{"label": "0"}, {"label": "2"}, {"label": "6"}, {"label": "0"}]
    )

    sampler = class_aware_sampler(
        dataset,
        tmp_path,
        ("bridge", "playground"),
        factor=1.5,
        seed=7,
    )

    assert sampler.weights.tolist() == [1.0, 1.5, 1.5, 1.0]
    assert sampler.num_samples == 4
    assert len(list(sampler)) == 4


def test_class_aware_sampler_rejects_unknown_class(tmp_path):
    (tmp_path / "classes.json").write_text(
        json.dumps([{"label": 2, "name": "bridge"}])
    )
    dataset = _Rows([{"label": "2"}])

    with pytest.raises(ValueError, match="unknown.*playground"):
        class_aware_sampler(dataset, tmp_path, ("playground",), factor=1.5, seed=0)
