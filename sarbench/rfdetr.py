"""RF-DETR with a shared image-classification branch.

The classifier consumes the P4 feature produced by RF-DETR's single backbone
forward.  A hook is used instead of re-running the DINOv2 encoder, preserving
the course requirement that each image enters the model only once.
"""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn
from torchvision.ops import box_convert


RFDETR_IMAGE_MEAN = (0.485, 0.456, 0.406)
RFDETR_IMAGE_STD = (0.229, 0.224, 0.225)
RFDETRVariant = Literal["small", "medium"]


def _install_exact_size_assignment_workaround() -> None:
    """Avoid RF-DETR 1.11.2's padded CUDA assignment failure on uneven target counts.

    The optimized padded path can occasionally assign a real row to a padded
    column, after which its fixed-size split is one element short. Solving
    batches grouped by their exact target count uses the same official solver
    without padded columns and preserves the intended assignments.
    """
    from rfdetr.models import _assignment

    if getattr(_assignment.assign_many_bucketed, "_cs701_exact_size", False):
        return
    original = _assignment.assign_many_bucketed

    def assign_many_exact_size(cost_matrices, sizes, group_detr):
        if not sizes or max(sizes) == 0:
            return original(cost_matrices, sizes, group_detr)
        group_widths = []
        for cost_matrix in cost_matrices:
            num_queries = cost_matrix.shape[0]
            if num_queries % group_detr:
                raise ValueError(
                    f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})"
                )
            group_widths.append(num_queries // group_detr)
        if not cost_matrices[0].is_cuda:
            cost_matrices = [cost_matrix.cpu() for cost_matrix in cost_matrices]
        return _assignment._assign_bucketed_by_size(
            cost_matrices, sizes, group_widths, group_detr
        )

    assign_many_exact_size._cs701_exact_size = True
    _assignment.assign_many_bucketed = assign_many_exact_size


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


def apply_detection_policy(
    detection: dict[str, torch.Tensor],
    image_class_probs: torch.Tensor,
    *,
    class_fusion_alpha: float = 0.0,
    class_gate: bool = False,
    topk: int | None = None,
) -> dict[str, torch.Tensor]:
    """Apply an image-class prior and optional top-k pruning to one image's detections.

    The course guarantees that every object in an image has the image class. Score
    fusion softly down-weights detector classes that disagree with the shared
    classifier, while ``class_gate`` is the deliberately stronger diagnostic that
    keeps only the classifier's top class. Degenerate boxes are removed so the
    result is directly valid for the submission scorer.
    """
    if class_fusion_alpha < 0:
        raise ValueError("class_fusion_alpha must be non-negative")
    if topk is not None and topk < 1:
        raise ValueError("topk must be positive when provided")
    if image_class_probs.ndim != 1:
        raise ValueError("image_class_probs must be a one-dimensional tensor")

    boxes = detection["boxes"]
    labels = detection["labels"].long()
    scores = detection["scores"].float()
    if labels.numel() and (labels.min() < 0 or labels.max() >= image_class_probs.numel()):
        raise ValueError("detection label is outside image_class_probs")

    finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    keep = finite & positive_area
    if class_gate:
        keep &= labels == image_class_probs.argmax()
    indices = torch.where(keep)[0]

    adjusted_scores = scores[indices]
    if class_fusion_alpha:
        priors = image_class_probs.to(scores.device)[labels[indices]].clamp_min(1e-8)
        adjusted_scores = adjusted_scores * priors.pow(class_fusion_alpha)
    order = torch.argsort(adjusted_scores, descending=True, stable=True)
    if topk is not None:
        order = order[:topk]
    indices = indices[order]
    return {
        "boxes": boxes[indices],
        "labels": labels[indices],
        "scores": adjusted_scores[order],
    }


class RFDETRMultiTask(nn.Module):
    """One fully trainable RF-DETR model for image classification and detection."""

    def __init__(
        self,
        num_classes: int = 9,
        pretrained: bool = True,
        resolution: int = 512,
        variant: RFDETRVariant = "small",
    ):
        super().__init__()
        try:
            from rfdetr import RFDETRMedium, RFDETRSmall
            from rfdetr.config import TrainConfig
            from rfdetr.models import build_criterion_from_config
        except ImportError as exc:  # pragma: no cover - exercised by the cluster setup path
            raise ImportError(
                "RF-DETR is not installed. Run cluster/setup_rfdetr.sh and activate .venv-rfdetr."
            ) from exc

        _install_exact_size_assignment_workaround()
        variants = {"small": RFDETRSmall, "medium": RFDETRMedium}
        if variant not in variants:
            raise ValueError(f"unsupported RF-DETR variant: {variant}")

        # RF-DETR 1.11.2 keeps DETR's legacy "maximum zero-based class id"
        # convention internally and constructs num_classes + 1 logits.  Passing
        # 8 therefore gives exactly the nine course categories 0..8.
        kwargs = {
            "num_classes": num_classes - 1,
            "num_channels": 3,
            "resolution": resolution,
        }
        if not pretrained:
            kwargs["pretrain_weights"] = None
        api = variants[variant](**kwargs)
        detector = api.model.model
        if detector is None:
            raise RuntimeError("RF-DETR did not expose its underlying detection model")

        self.detector = detector
        self.variant = variant
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
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        box_loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if box_loss_scale <= 0:
            raise ValueError("box_loss_scale must be positive")
        loss_dict = self.criterion(outputs, targets)
        weighted = {
            name: value
            * self.criterion.weight_dict[name]
            * (box_loss_scale if name.startswith(("loss_bbox", "loss_giou")) else 1.0)
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
