"""RF-DETR Small with a shared image-classification branch.

The classifier consumes the P4 feature produced by RF-DETR's single backbone
forward.  A hook is used instead of re-running the DINOv2 encoder, preserving
the course requirement that each image enters the model only once.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.ops import box_convert


RFDETR_IMAGE_MEAN = (0.485,)
RFDETR_IMAGE_STD = (0.229,)


def prepare_detection_targets(targets: list[dict[str, Any]], size: int = 512) -> list[dict[str, torch.Tensor]]:
    """Convert course XYXY/category-1..9 targets to RF-DETR normalized CXCYWH/0..8 targets."""
    converted = []
    for target in targets:
        boxes = box_convert(target["boxes"].as_subclass(torch.Tensor), "xyxy", "cxcywh") / size
        converted.append({
            "boxes": boxes,
            "labels": target["labels"] - 1,
        })
    return converted


class RFDETRMultiTask(nn.Module):
    """One fully trainable RF-DETR Small model for image classification and detection."""

    def __init__(self, num_classes: int = 9, pretrained: bool = True, resolution: int = 512):
        super().__init__()
        try:
            from rfdetr import RFDETRSmall
            from rfdetr.config import TrainConfig
            from rfdetr.models import build_criterion_from_config
        except ImportError as exc:  # pragma: no cover - exercised by the cluster setup path
            raise ImportError(
                "RF-DETR is not installed. Run cluster/setup_rfdetr.sh and activate .venv-rfdetr."
            ) from exc

        # RF-DETR 1.11.2 keeps DETR's legacy "maximum zero-based class id"
        # convention internally and constructs num_classes + 1 logits.  Passing
        # 8 therefore gives exactly the nine course categories 0..8.
        kwargs = {
            "num_classes": num_classes - 1,
            "num_channels": 1,
            "resolution": resolution,
        }
        if not pretrained:
            kwargs["pretrain_weights"] = None
        api = RFDETRSmall(**kwargs)
        detector = api.model.model
        if detector is None:
            raise RuntimeError("RF-DETR did not expose its underlying detection model")

        self.detector = detector
        self.postprocess = api.model.postprocess
        self.model_config = api.model_config
        self.rf_args = api.model.args
        train_config = TrainConfig(
            dataset_dir=None,
            output_dir="output",
            multi_scale=False,
            use_ema=False,
            tensorboard=False,
        )
        self.criterion, _ = build_criterion_from_config(self.model_config, train_config)

        hidden_dim = int(self.model_config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        self._shared_feature = None
        self._feature_hook = self.detector.backbone.register_forward_hook(self._capture_backbone_feature)

    def _capture_backbone_feature(self, _module, _inputs, output):
        features = output[0]
        if not features:
            raise RuntimeError("RF-DETR backbone returned no feature maps")
        self._shared_feature = features[-1]

    @staticmethod
    def _masked_average(feature) -> torch.Tensor:
        tensor = feature.tensors
        mask = feature.mask
        if mask is None:
            return tensor.mean(dim=(-2, -1))
        valid = (~mask).unsqueeze(1).to(tensor.dtype)
        return (tensor * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._shared_feature = None
        outputs = self.detector(images)
        if self._shared_feature is None:
            raise RuntimeError("RF-DETR backbone feature hook did not run")
        pooled = self._masked_average(self._shared_feature)
        logits = self.classifier(pooled)
        self._shared_feature = None
        return logits, outputs

    def detection_losses(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_dict = self.criterion(outputs, targets)
        weighted = {
            name: value * self.criterion.weight_dict[name]
            for name, value in loss_dict.items()
            if name in self.criterion.weight_dict
        }
        return torch.stack(list(weighted.values())).sum(), weighted

    def detections(self, outputs: dict[str, torch.Tensor], size: int = 512) -> list[dict[str, torch.Tensor]]:
        batch = outputs["pred_logits"].shape[0]
        target_sizes = outputs["pred_logits"].new_full((batch, 2), size, dtype=torch.long)
        return self.postprocess(outputs, target_sizes)

    def optimizer_parameter_groups(
        self,
        lr: float,
        encoder_lr: float,
        weight_decay: float,
        layer_decay: float = 0.8,
        component_decay: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Use RF-DETR's layer-wise decay for the detector and a separate new-head group."""
        from rfdetr.training.param_groups import get_param_dict

        args = self.rf_args
        args.lr = lr
        args.lr_encoder = encoder_lr
        args.weight_decay = weight_decay
        args.lr_vit_layer_decay = layer_decay
        args.lr_component_decay = component_decay
        groups = get_param_dict(args, self.detector)
        groups.append({"params": list(self.classifier.parameters()), "lr": lr, "weight_decay": weight_decay})
        return groups
