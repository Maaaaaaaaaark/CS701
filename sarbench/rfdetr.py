"""RF-DETR with a shared image-classification branch.

The classifier consumes the P4 feature produced by RF-DETR's single backbone
forward.  A hook is used instead of re-running the DINOv2 encoder, preserving
the course requirement that each image enters the model only once.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, box_convert, box_iou, generalized_box_iou_loss


RFDETR_IMAGE_MEAN = (0.485, 0.456, 0.406)
RFDETR_IMAGE_STD = (0.229, 0.224, 0.225)
RFDETRVariant = Literal["small", "medium"]


class SARDetailFusion(nn.Module):
    """Preserve sub-patch SAR detail and inject it into RF-DETR projector levels.

    RF-DETR Small exposes only a stride-16 P4 feature to its decoder. This branch
    extracts stride-4, stride-8 and stride-16 features directly from grayscale
    pixels, pools all three onto each existing projector grid, and adds a learned
    residual. Keeping the projector interface intact preserves every pretrained
    decoder weight.
    """

    def __init__(self, hidden_dim: int, num_levels: int):
        super().__init__()
        if hidden_dim % 32:
            raise ValueError("hidden_dim must be divisible by 32")
        self.to_stride2 = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.to_stride4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.to_stride8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
        )
        self.to_stride16 = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        combined_dim = 64 + 128 + hidden_dim
        self.combine = nn.Sequential(
            nn.Conv2d(combined_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        self.level_adapters = nn.ModuleList(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False) for _ in range(num_levels)
        )
        # Start close to the pretrained detector while allowing gradients to
        # reach the whole detail branch on the first optimizer step.
        self.level_gates = nn.Parameter(torch.full((num_levels,), 0.1))

    def forward(
        self,
        normalized_rgb: torch.Tensor,
        target_shapes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        if len(target_shapes) != len(self.level_adapters):
            raise ValueError(
                f"expected {len(self.level_adapters)} projector levels, got {len(target_shapes)}"
            )
        # The grayscale SAR image was replicated to RGB and ImageNet-normalized.
        # Recover its original [0, 1] intensity exactly from channel zero.
        gray = normalized_rgb[:, :1] * RFDETR_IMAGE_STD[0] + RFDETR_IMAGE_MEAN[0]
        stride2 = self.to_stride2(gray)
        stride4 = self.to_stride4(stride2)
        stride8 = self.to_stride8(stride4)
        stride16 = self.to_stride16(stride8)
        outputs = []
        for index, shape in enumerate(target_shapes):
            multi_scale = torch.cat(
                [
                    F.adaptive_avg_pool2d(stride4, shape),
                    F.adaptive_avg_pool2d(stride8, shape),
                    F.adaptive_avg_pool2d(stride16, shape),
                ],
                dim=1,
            )
            detail = self.level_adapters[index](self.combine(multi_scale))
            outputs.append(detail * self.level_gates[index])
        return outputs


class SmallObjectHead(nn.Module):
    """Stride-4 anchor-free head for objects smaller than a ViT patch."""

    def __init__(
        self,
        num_classes: int = 9,
        input_size: int = 512,
        max_object_size: float = 32.0,
        score_threshold: float = 0.05,
        detections_per_image: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.max_object_size = max_object_size
        self.score_threshold = score_threshold
        self.detections_per_image = detections_per_image
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
        )
        self.class_logits = nn.Conv2d(96, num_classes, 1)
        self.box_ltrb = nn.Conv2d(96, 4, 1)
        prior_probability = 0.01
        nn.init.constant_(
            self.class_logits.bias,
            -math.log((1 - prior_probability) / prior_probability),
        )
        nn.init.normal_(self.class_logits.weight, std=0.01)
        nn.init.zeros_(self.box_ltrb.weight)
        nn.init.constant_(self.box_ltrb.bias, -0.4)

    def forward(self, normalized_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        gray = normalized_rgb[:, :1] * RFDETR_IMAGE_STD[0] + RFDETR_IMAGE_MEAN[0]
        features = self.stem(gray)
        return {
            "class_logits": self.class_logits(features),
            "box_ltrb": self.box_ltrb(features),
        }

    @staticmethod
    def _focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = logits.float(), targets.float()
        probabilities = logits.sigmoid()
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        target_probabilities = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha = 0.25 * targets + 0.75 * (1 - targets)
        return (alpha * (1 - target_probabilities).pow(2) * cross_entropy).sum()

    def losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        logits = outputs["class_logits"]
        raw_ltrb = outputs["box_ltrb"]
        batch, _, height, width = logits.shape
        heatmap = logits.new_zeros(logits.shape, dtype=torch.float32)
        positive_batches, positive_ys, positive_xs = [], [], []
        regression_target_parts, normalized_gt_box_parts = [], []

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].float()
            labels = target["labels"].long()
            if boxes.numel() == 0:
                continue
            small = boxes[:, 2:].amax(dim=1) * self.input_size <= self.max_object_size
            boxes, labels = boxes[small], labels[small]
            if boxes.numel() == 0:
                continue
            center_x, center_y, box_width, box_height = boxes.unbind(dim=1)
            x_indices = (center_x * width).long().clamp(0, width - 1)
            y_indices = (center_y * height).long().clamp(0, height - 1)
            keys = labels * height * width + y_indices * width + x_indices
            _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
            positions = torch.arange(len(keys), device=boxes.device)
            first = torch.full(
                (len(keys),),
                len(keys),
                dtype=torch.long,
                device=boxes.device,
            )
            first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
            first = first[first < len(keys)]
            boxes, labels = boxes[first], labels[first]
            center_x, center_y, box_width, box_height = boxes.unbind(dim=1)
            x_indices, y_indices = x_indices[first], y_indices[first]
            cell_x, cell_y = x_indices.float() + 0.5, y_indices.float() + 0.5
            x1 = (center_x - box_width / 2) * width
            y1 = (center_y - box_height / 2) * height
            x2 = (center_x + box_width / 2) * width
            y2 = (center_y + box_height / 2) * height
            heatmap[batch_index, labels, y_indices, x_indices] = 1
            positive_batches.append(torch.full_like(x_indices, batch_index))
            positive_ys.append(y_indices)
            positive_xs.append(x_indices)
            regression_target_parts.append(
                torch.stack([cell_x - x1, cell_y - y1, x2 - cell_x, y2 - cell_y], dim=1).clamp_min(0.01)
            )
            normalized_gt_box_parts.append(torch.stack([x1 / width, y1 / height, x2 / width, y2 / height], dim=1))

        positive_count = sum(len(part) for part in regression_target_parts)
        normalizer = max(positive_count, 1)
        classification = self._focal_loss(logits, heatmap) / normalizer
        if not regression_target_parts:
            zero = raw_ltrb.sum() * 0
            return {"loss_small_cls": classification, "loss_small_l1": zero, "loss_small_giou": zero}

        batch_indices = torch.cat(positive_batches)
        y_indices = torch.cat(positive_ys)
        x_indices = torch.cat(positive_xs)
        predicted_ltrb = F.softplus(raw_ltrb.float().permute(0, 2, 3, 1)[batch_indices, y_indices, x_indices])
        target_ltrb = torch.cat(regression_target_parts)
        l1 = F.smooth_l1_loss(predicted_ltrb, target_ltrb, reduction="sum") / normalizer

        cell_x = x_indices.float() + 0.5
        cell_y = y_indices.float() + 0.5
        predicted_boxes = torch.stack(
            [
                (cell_x - predicted_ltrb[:, 0]) / width,
                (cell_y - predicted_ltrb[:, 1]) / height,
                (cell_x + predicted_ltrb[:, 2]) / width,
                (cell_y + predicted_ltrb[:, 3]) / height,
            ],
            dim=1,
        )
        gt_boxes = torch.cat(normalized_gt_box_parts)
        giou = generalized_box_iou_loss(predicted_boxes, gt_boxes, reduction="sum") / normalizer
        return {"loss_small_cls": classification, "loss_small_l1": l1, "loss_small_giou": giou}

    def detections(self, outputs: dict[str, torch.Tensor], size: int = 512) -> list[dict[str, torch.Tensor]]:
        logits = outputs["class_logits"].float()
        ltrb = F.softplus(outputs["box_ltrb"].float())
        batch, classes, height, width = logits.shape
        results = []
        for batch_index in range(batch):
            scores = logits[batch_index].sigmoid().permute(1, 2, 0).reshape(-1)
            top_count = min(self.detections_per_image * 4, scores.numel())
            scores, indices = torch.topk(scores, top_count)
            keep = scores >= self.score_threshold
            scores, indices = scores[keep], indices[keep]
            labels = indices.remainder(classes)
            cells = torch.div(indices, classes, rounding_mode="floor")
            y_indices = torch.div(cells, width, rounding_mode="floor")
            x_indices = cells.remainder(width)
            distances = ltrb[batch_index, :, y_indices, x_indices].T
            stride_x, stride_y = size / width, size / height
            center_x = (x_indices.float() + 0.5) * stride_x
            center_y = (y_indices.float() + 0.5) * stride_y
            boxes = torch.stack(
                [
                    center_x - distances[:, 0] * stride_x,
                    center_y - distances[:, 1] * stride_y,
                    center_x + distances[:, 2] * stride_x,
                    center_y + distances[:, 3] * stride_y,
                ],
                dim=1,
            ).clamp(0, size)
            dimensions = boxes[:, 2:] - boxes[:, :2]
            valid = (dimensions > 0).all(dim=1) & (dimensions.amax(dim=1) <= self.max_object_size * 1.5)
            boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
            keep = batched_nms(boxes, scores, labels, 0.5)[: self.detections_per_image]
            results.append({"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]})
        return results


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
        detail_fusion: bool = False,
        detail_classifier_fusion: bool = True,
        small_object_head: bool = False,
        small_object_max_size: float = 32.0,
        small_object_score_threshold: float = 0.05,
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
        hidden_dim = int(self.model_config.hidden_dim)
        num_levels = len(self.model_config.projector_scale)
        self.detail_fusion = SARDetailFusion(hidden_dim, num_levels) if detail_fusion else None
        self.detail_classifier_fusion = detail_classifier_fusion
        self.small_object_head = (
            SmallObjectHead(
                num_classes=num_classes,
                input_size=resolution,
                max_object_size=small_object_max_size,
                score_threshold=small_object_score_threshold,
            )
            if small_object_head
            else None
        )
        train_config = TrainConfig(
            dataset_dir=None,
            output_dir="output",
            multi_scale=False,
            use_ema=False,
            tensorboard=False,
        )
        self.criterion, _ = build_criterion_from_config(self.model_config, train_config)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.classifier[-1].weight, std=0.02)
        nn.init.zeros_(self.classifier[-1].bias)

        self._shared_feature = None
        self._detail_input = None
        self._detail_input_hook = self.detector.backbone.register_forward_pre_hook(
            self._capture_detail_input
        )
        self._feature_hook = self.detector.backbone.register_forward_hook(
            self._fuse_and_capture_backbone_feature
        )

    def _capture_detail_input(self, _module, inputs):
        tensor_list = inputs[0]
        self._detail_input = tensor_list.tensors if hasattr(tensor_list, "tensors") else tensor_list

    def _fuse_and_capture_backbone_feature(self, _module, _inputs, output):
        features, positions, cross_attn_features = output
        if not features:
            raise RuntimeError("RF-DETR backbone returned no feature maps")
        classifier_feature = features[-1]
        if self.detail_fusion is not None:
            if self._detail_input is None:
                raise RuntimeError("RF-DETR detail fusion did not capture the input image")
            details = self.detail_fusion(
                self._detail_input,
                [feature.tensors.shape[-2:] for feature in features],
            )
            fused_features = []
            for feature, detail in zip(features, details):
                fused_features.append(
                    type(feature)(
                        feature.tensors + detail.to(feature.tensors.dtype),
                        feature.mask,
                        feature.no_padding,
                    )
                )
            features = fused_features
        self._shared_feature = features[-1] if self.detail_classifier_fusion else classifier_feature
        self._detail_input = None
        return features, positions, cross_attn_features

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
        small_object_outputs = self.small_object_head(images) if self.small_object_head is not None else None
        outputs = self.detector(images)
        if self._shared_feature is None:
            raise RuntimeError("RF-DETR backbone feature hook did not run")
        pooled = self._masked_average(self._shared_feature)
        logits = self.classifier(pooled)
        self._shared_feature = None
        if small_object_outputs is not None:
            outputs["_small_object"] = small_object_outputs
        return logits, outputs

    def detection_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        box_loss_scale: float = 1.0,
        small_object_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if box_loss_scale <= 0 or small_object_loss_weight < 0:
            raise ValueError("box_loss_scale must be positive and small_object_loss_weight non-negative")
        detector_outputs = {name: value for name, value in outputs.items() if name != "_small_object"}
        loss_dict = self.criterion(detector_outputs, targets)
        weighted = {
            name: value
            * self.criterion.weight_dict[name]
            * (box_loss_scale if name.startswith(("loss_bbox", "loss_giou")) else 1.0)
            for name, value in loss_dict.items()
            if name in self.criterion.weight_dict
        }
        if getattr(self, "small_object_head", None) is not None:
            small_outputs = outputs.get("_small_object")
            if small_outputs is None:
                raise RuntimeError("small-object head is enabled but its outputs are missing")
            small_losses = self.small_object_head.losses(small_outputs, targets)
            weighted.update(
                {
                    name: value * small_object_loss_weight
                    for name, value in small_losses.items()
                }
            )
        return torch.stack(list(weighted.values())).sum(), weighted

    def detections(self, outputs: dict[str, torch.Tensor], size: int = 512) -> list[dict[str, torch.Tensor]]:
        batch = outputs["pred_logits"].shape[0]
        target_sizes = outputs["pred_logits"].new_full((batch, 2), size, dtype=torch.long)
        detector_outputs = {name: value for name, value in outputs.items() if name != "_small_object"}
        detections = self.postprocess(detector_outputs, target_sizes)
        if getattr(self, "small_object_head", None) is None:
            return detections
        small_outputs = outputs.get("_small_object")
        if small_outputs is None:
            raise RuntimeError("small-object head is enabled but its outputs are missing")
        small_detections = self.small_object_head.detections(small_outputs, size=size)
        merged = []
        for detection, small in zip(detections, small_detections):
            keep_small = torch.ones(len(small["boxes"]), dtype=torch.bool, device=small["boxes"].device)
            for label in torch.unique(small["labels"]):
                small_indices = torch.where(small["labels"] == label)[0]
                baseline_indices = torch.where(detection["labels"] == label)[0]
                if len(baseline_indices):
                    overlap = box_iou(small["boxes"][small_indices], detection["boxes"][baseline_indices])
                    keep_small[small_indices] = overlap.amax(dim=1) < 0.7
            boxes = torch.cat([detection["boxes"], small["boxes"][keep_small]])
            labels = torch.cat([detection["labels"], small["labels"][keep_small]])
            scores = torch.cat([detection["scores"].float(), small["scores"][keep_small]])
            order = torch.argsort(scores, descending=True, stable=True)[:300]
            merged.append({"boxes": boxes[order], "labels": labels[order], "scores": scores[order]})
        return merged

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
        if self.detail_fusion is not None:
            groups.append(
                {"params": list(self.detail_fusion.parameters()), "lr": lr, "weight_decay": weight_decay}
            )
        if getattr(self, "small_object_head", None) is not None:
            groups.append(
                {"params": list(self.small_object_head.parameters()), "lr": lr, "weight_decay": weight_decay}
            )
        return groups
