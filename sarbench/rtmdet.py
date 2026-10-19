"""A dependency-light RTMDet-m multi-task model for the CS701 SAR benchmark.

The model uses timm's ImageNet-pretrained CSPNeXt-m backbone, an RTMDet-style
CSPNeXt PAFPN and shared-convolution SepBN dense head.  A global classification
head and the detector consume the same backbone features from one image pass.
"""
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, box_iou, generalized_box_iou_loss


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )


class CSPNeXtBlock(nn.Module):
    """The depthwise 5x5 bottleneck used inside a compact CSPNeXt neck stage."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.depthwise = ConvBNAct(channels, channels, 5, groups=channels)
        self.conv2 = ConvBNAct(channels, channels, 1)

    def forward(self, x):
        return x + self.conv2(self.depthwise(self.conv1(x)))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=2, expand_ratio=0.5):
        super().__init__()
        hidden = int(out_channels * expand_ratio)
        self.main = ConvBNAct(in_channels, hidden)
        self.short = ConvBNAct(in_channels, hidden)
        self.blocks = nn.Sequential(*(CSPNeXtBlock(hidden) for _ in range(blocks)))
        self.fuse = ConvBNAct(2 * hidden, out_channels)

    def forward(self, x):
        return self.fuse(torch.cat((self.blocks(self.main(x)), self.short(x)), dim=1))


class CSPNeXtPAFPN(nn.Module):
    """Three-level top-down/bottom-up feature pyramid used by RTMDet-m."""

    def __init__(self, in_channels, out_channels=192, blocks=2):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f'RTMDet expects three backbone levels, got {in_channels}')
        self.lateral = nn.ModuleList(ConvBNAct(c, out_channels) for c in in_channels)
        self.top4 = CSPLayer(2 * out_channels, out_channels, blocks)
        self.top3 = CSPLayer(2 * out_channels, out_channels, blocks)
        self.down3 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.bottom4 = CSPLayer(2 * out_channels, out_channels, blocks)
        self.down4 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.bottom5 = CSPLayer(2 * out_channels, out_channels, blocks)

    def forward(self, features):
        c3, c4, c5 = (layer(feature) for layer, feature in zip(self.lateral, features))
        p4 = self.top4(torch.cat((c4, F.interpolate(c5, size=c4.shape[-2:], mode='nearest')), dim=1))
        p3 = self.top3(torch.cat((c3, F.interpolate(p4, size=c3.shape[-2:], mode='nearest')), dim=1))
        n4 = self.bottom4(torch.cat((p4, self.down3(p3)), dim=1))
        n5 = self.bottom5(torch.cat((c5, self.down4(n4)), dim=1))
        return [p3, n4, n5]


class SharedSepBNBranch(nn.Module):
    """Convolution weights are shared across pyramid levels; BN statistics are not."""

    def __init__(self, channels=192, levels=3, depth=2):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False) for _ in range(depth))
        self.norms = nn.ModuleList(
            nn.ModuleList(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03) for _ in range(levels))
            for _ in range(depth))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, level):
        for conv, norms in zip(self.convs, self.norms):
            x = self.act(norms[level](conv(x)))
        return x


class RTMDetHead(nn.Module):
    def __init__(self, num_classes=9, channels=192, levels=3):
        super().__init__()
        self.num_classes = num_classes
        self.cls_branch = SharedSepBNBranch(channels, levels)
        self.reg_branch = SharedSepBNBranch(channels, levels)
        self.cls_preds = nn.ModuleList(nn.Conv2d(channels, num_classes, 1) for _ in range(levels))
        self.reg_preds = nn.ModuleList(nn.Conv2d(channels, 4, 1) for _ in range(levels))
        self.scales = nn.Parameter(torch.ones(levels))
        prior_probability = 0.01
        bias = -math.log((1 - prior_probability) / prior_probability)
        for cls_pred, reg_pred in zip(self.cls_preds, self.reg_preds):
            nn.init.normal_(cls_pred.weight, std=0.01)
            nn.init.constant_(cls_pred.bias, bias)
            nn.init.normal_(reg_pred.weight, std=0.01)
            nn.init.constant_(reg_pred.bias, 1.0)

    def forward(self, features):
        cls_logits, distances = [], []
        for level, feature in enumerate(features):
            cls_logits.append(self.cls_preds[level](self.cls_branch(feature, level)))
            raw = self.reg_preds[level](self.reg_branch(feature, level)) * self.scales[level]
            distances.append(F.softplus(raw))
        return cls_logits, distances


def _points(height, width, stride, device, dtype):
    # RTMDet uses point priors with offset 0.
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing='ij')
    return torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=1)


def _distance_to_boxes(points, distances, size):
    return torch.stack((
        (points[:, 0] - distances[:, 0]).clamp(0, size[1]),
        (points[:, 1] - distances[:, 1]).clamp(0, size[0]),
        (points[:, 0] + distances[:, 2]).clamp(0, size[1]),
        (points[:, 1] + distances[:, 3]).clamp(0, size[0])), dim=1)


@torch.no_grad()
def dynamic_soft_label_assign(class_logits, boxes, points, strides, target, topk=13):
    """A compact SimOTA/DynamicSoftLabel assignment used by RTMDet."""
    gt_boxes = target['boxes'].as_subclass(torch.Tensor).float()
    gt_labels = target['labels'].long() - 1
    if len(gt_boxes) == 0:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return empty, empty, boxes.new_empty(0)

    left_top = points[:, None, :] - gt_boxes[None, :, :2]
    right_bottom = gt_boxes[None, :, 2:] - points[:, None, :]
    inside = torch.cat((left_top, right_bottom), dim=-1).amin(dim=-1) > 0
    centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    scaled_distance = ((points[:, None, :] - centers[None]).square().sum(-1).sqrt()
                       / strides[:, None].clamp_min(1))
    # Very small boxes can contain no feature point; retain their nearest prior.
    for gt_index in range(len(gt_boxes)):
        if not inside[:, gt_index].any():
            inside[scaled_distance[:, gt_index].argmin(), gt_index] = True

    overlaps = box_iou(boxes, gt_boxes).clamp_(0, 1)
    selected_logits = class_logits[:, gt_labels]
    cls_cost = F.binary_cross_entropy_with_logits(selected_logits, overlaps, reduction='none')
    iou_cost = -overlaps.clamp_min(1e-7).log()
    cost = cls_cost + 3.0 * iou_cost + (~inside) * 1e6

    matching = torch.zeros_like(cost, dtype=torch.bool)
    candidate_k = min(topk, len(boxes))
    candidate_overlaps = overlaps * inside
    dynamic_ks = candidate_overlaps.topk(candidate_k, dim=0).values.sum(0).to(torch.int64).clamp_min(1)
    dynamic_ks = torch.minimum(dynamic_ks, inside.sum(0))
    for gt_index, dynamic_k in enumerate(dynamic_ks.tolist()):
        chosen = cost[:, gt_index].topk(dynamic_k, largest=False).indices
        matching[chosen, gt_index] = True

    conflicts = matching.sum(1) > 1
    if conflicts.any():
        conflict_indices = conflicts.nonzero(as_tuple=False).squeeze(1)
        best_gt = cost[conflict_indices].argmin(1)
        matching[conflict_indices] = False
        matching[conflict_indices, best_gt] = True
    positive = matching.any(1).nonzero(as_tuple=False).squeeze(1)
    matched_gt = matching[positive].to(torch.int64).argmax(1)
    quality = overlaps[positive, matched_gt].clamp_min(0.05)
    return positive, matched_gt, quality


def quality_focal_loss(logits, positive, labels, quality, beta=2.0):
    probabilities = logits.sigmoid()
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    loss = loss * probabilities.pow(beta)
    if len(positive):
        positive_logits = logits[positive, labels]
        positive_probabilities = probabilities[positive, labels]
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits, quality, reduction='none') * (quality - positive_probabilities).abs().pow(beta)
        loss[positive, labels] = positive_loss
    return loss.sum()


class RTMDetMultiTask(nn.Module):
    """One fully trainable CSPNeXt-m backbone feeding image classification and RTMDet branches."""

    strides = (8, 16, 32)

    def __init__(self, num_classes=9, pretrained=True, score_threshold=0.001, detections_per_image=100):
        super().__init__()
        # timm adapts the ImageNet RGB stem weights to one input channel.
        self.backbone = timm.create_model(
            'cspnext_m.rsb_a1_in1k', pretrained=pretrained, features_only=True,
            in_chans=1, out_indices=(2, 3, 4))
        channels = self.backbone.feature_info.channels()
        reductions = tuple(self.backbone.feature_info.reduction())
        if reductions != self.strides:
            raise ValueError(f'expected CSPNeXt reductions {self.strides}, got {reductions}')
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.neck = CSPNeXtPAFPN(channels, out_channels=192, blocks=2)
        self.detector = RTMDetHead(num_classes, channels=192)
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.detections_per_image = detections_per_image

    def _flatten_predictions(self, cls_levels, distance_levels, image_size):
        logits, distances, points, strides = [], [], [], []
        for cls_level, distance_level, stride in zip(cls_levels, distance_levels, self.strides):
            batch, _, height, width = cls_level.shape
            logits.append(cls_level.permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes).float())
            distances.append(distance_level.permute(0, 2, 3, 1).reshape(batch, -1, 4).float() * stride)
            level_points = _points(height, width, stride, cls_level.device, torch.float32)
            points.append(level_points)
            strides.append(level_points.new_full((len(level_points),), stride))
        logits = torch.cat(logits, dim=1)
        distances = torch.cat(distances, dim=1)
        points = torch.cat(points)
        strides = torch.cat(strides)
        boxes = torch.stack([_distance_to_boxes(points, item, image_size) for item in distances])
        return logits, boxes, points, strides

    def _losses(self, logits, boxes, points, strides, targets):
        cls_total = logits.new_zeros(())
        box_total = logits.new_zeros(())
        normalizer = logits.new_zeros(())
        for image_logits, image_boxes, target in zip(logits, boxes, targets):
            positive, matched_gt, quality = dynamic_soft_label_assign(
                image_logits.detach(), image_boxes.detach(), points, strides, target)
            labels = target['labels'].long()[matched_gt] - 1
            cls_total = cls_total + quality_focal_loss(image_logits, positive, labels, quality)
            if len(positive):
                gt_boxes = target['boxes'].as_subclass(torch.Tensor).float()[matched_gt]
                box_loss = generalized_box_iou_loss(
                    image_boxes[positive], gt_boxes, reduction='none')
                box_total = box_total + (box_loss * quality).sum()
                normalizer = normalizer + quality.sum()
        normalizer = normalizer.clamp_min(1.0)
        return {'loss_quality_focal': cls_total / normalizer,
                'loss_giou': 2.0 * box_total / normalizer}

    def _detections(self, logits, boxes, image_size):
        outputs = []
        for image_logits, image_boxes in zip(logits, boxes):
            scores = image_logits.sigmoid().flatten()
            keep = scores > self.score_threshold
            indices = keep.nonzero(as_tuple=False).squeeze(1)
            scores = scores[indices]
            if len(scores) > 30000:
                scores, order = scores.topk(30000)
                indices = indices[order]
            point_indices = torch.div(indices, self.num_classes, rounding_mode='floor')
            labels = indices.remainder(self.num_classes)
            selected_boxes = image_boxes[point_indices]
            widths = selected_boxes[:, 2] - selected_boxes[:, 0]
            heights = selected_boxes[:, 3] - selected_boxes[:, 1]
            valid = (widths > 0) & (heights > 0)
            selected_boxes, scores, labels = selected_boxes[valid], scores[valid], labels[valid]
            keep = batched_nms(selected_boxes, scores, labels, 0.65)[:self.detections_per_image]
            outputs.append({'boxes': selected_boxes[keep], 'scores': scores[keep], 'labels': labels[keep] + 1})
        return outputs

    def forward(self, images, targets=None):
        features = self.backbone(images)
        classification_logits = self.classifier(features[-1].mean(dim=(-2, -1)))
        pyramid = self.neck(features)
        cls_levels, distance_levels = self.detector(pyramid)
        logits, boxes, points, strides = self._flatten_predictions(
            cls_levels, distance_levels, tuple(images.shape[-2:]))
        if self.training:
            if targets is None:
                raise ValueError('targets are required while training RTMDet')
            return classification_logits, [], self._losses(logits, boxes, points, strides, targets)
        return classification_logits, self._detections(logits, boxes, tuple(images.shape[-2:])), {}
