"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops

from .adapters import set_task


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck, optionally fusing a true stride-4 image feature for sub-patch object localization."""

    def __init__(self, dim=768, channels=256, detail_stem=False):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))
        # The ViT has one token per 16 x 16 patch. Upsampling those tokens to stride 4 cannot reconstruct where
        # a 6-15 px object lies inside a patch, so this small CNN preserves spatial detail directly from pixels.
        self.detail_stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        ) if detail_stem else None

    def forward(self, x, images=None):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        if self.detail_stem is not None:
            if images is None:
                raise ValueError('images are required when the stride-4 detail stem is enabled')
            levels[0] = levels[0] + self.detail_stem(images)
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class ClassAgnosticFastRCNNPredictor(nn.Module):
    """A small Cascade stage predictor: per-class scores and one box refinement shared by all classes."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x)


def class_agnostic_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Torchvision's Fast R-CNN loss with one class-agnostic set of four box deltas."""
    labels = torch.cat(labels)
    regression_targets = torch.cat(regression_targets)
    classification_loss = F.cross_entropy(class_logits, labels)
    positive = torch.where(labels > 0)[0]
    box_loss = F.smooth_l1_loss(
        box_regression[positive], regression_targets[positive], beta=1 / 9, reduction='sum')
    return classification_loss, box_loss / labels.numel()


class CompactCascadeRoIHeads(nn.Module):
    """Three compact Cascade R-CNN stages that refine proposals at IoU 0.5, 0.6 and 0.7."""

    def __init__(self, num_classes, channels=256, hidden=384):
        super().__init__()
        thresholds = (0.5, 0.6, 0.7)
        coder_weights = ((10.0, 10.0, 5.0, 5.0),
                         (20.0, 20.0, 10.0, 10.0),
                         (30.0, 30.0, 15.0, 15.0))
        self.loss_weights = (1.0, 0.5, 0.25)
        self.stages = nn.ModuleList([
            RoIHeads(
                MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
                TwoMLPHead(channels * 7 * 7, hidden),
                ClassAgnosticFastRCNNPredictor(hidden, num_classes + 1),
                fg_iou_thresh=threshold, bg_iou_thresh=threshold,
                batch_size_per_image=512, positive_fraction=0.25,
                bbox_reg_weights=weights, score_thresh=0.05, nms_thresh=0.5,
                detections_per_img=100)
            for threshold, weights in zip(thresholds, coder_weights)
        ])

    @staticmethod
    def _refine(stage, box_regression, proposals, image_shapes, labels=None):
        boxes_per_image = [len(proposal) for proposal in proposals]
        decoded = stage.box_coder.decode(box_regression, proposals).squeeze(1)
        if labels is not None:
            # Regression is supervised only for foreground samples. Keep background proposals unchanged rather
            # than moving them with unconstrained deltas before the next training stage.
            original = torch.cat(proposals)
            foreground = torch.cat(labels) > 0
            decoded = torch.where(foreground[:, None], decoded, original)
        decoded = decoded.split(boxes_per_image)
        return [box_ops.clip_boxes_to_image(boxes, shape).detach()
                for boxes, shape in zip(decoded, image_shapes)]

    @staticmethod
    def _predict(stage, features, proposals, image_shapes):
        box_features = stage.box_roi_pool(features, proposals, image_shapes)
        box_features = stage.box_head(box_features)
        return stage.box_predictor(box_features)

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            losses = {}
            for index, (stage, weight) in enumerate(zip(self.stages, self.loss_weights), 1):
                proposals, _, labels, regression_targets = stage.select_training_samples(proposals, targets)
                class_logits, box_regression = self._predict(stage, features, proposals, image_shapes)
                classification_loss, box_loss = class_agnostic_fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets)
                losses[f'loss_classifier_s{index}'] = classification_loss * weight
                losses[f'loss_box_reg_s{index}'] = box_loss * weight
                if index < len(self.stages):
                    proposals = self._refine(stage, box_regression, proposals, image_shapes, labels)
            return [], losses

        probabilities = []
        for index, stage in enumerate(self.stages):
            class_logits, box_regression = self._predict(stage, features, proposals, image_shapes)
            probabilities.append(F.softmax(class_logits, dim=-1))
            if index + 1 < len(self.stages):
                proposals = self._refine(stage, box_regression, proposals, image_shapes)

        # Cascade R-CNN averages classification confidence across stages. The final class-agnostic box
        # regression is repeated for each class only to reuse torchvision's clipping and per-class NMS.
        averaged_logits = torch.stack(probabilities).mean(0).clamp_min(1e-8).log()
        expanded_regression = box_regression.repeat(1, averaged_logits.shape[1])
        boxes, scores, labels = self.stages[-1].postprocess_detections(
            averaged_logits, expanded_regression, proposals, image_shapes)
        return [{'boxes': box, 'scores': score, 'labels': label}
                for box, score, label in zip(boxes, scores, labels)], {}


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, detail_stem=False, cascade=False):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim, detail_stem=detail_stem)
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        if cascade:
            self.roi_heads = CompactCascadeRoIHeads(num_classes)
        else:
            self.roi_heads = RoIHeads(
                MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
                TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
                fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
                bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections in eval mode, detection losses in training mode)."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        h = images.shape[-2] // self.backbone.patch_size
        w = images.shape[-1] // self.backbone.patch_size
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)), images)
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}
