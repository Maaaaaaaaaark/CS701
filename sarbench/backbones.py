"""ViT-style backbones for 512 px inputs, returning patch tokens without prefix/register tokens."""
import timm
from torch import nn


class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=pretrained,
                                     img_size=512, in_chans=1, num_classes=0)
        self.embed_dim = 768
        self.patch_size = 16

    @property
    def blocks(self):
        return self.vit.blocks

    @property
    def final_norm(self):
        return self.vit.norm

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768
        self.patch_size = 16

    @property
    def blocks(self):
        return self.terramind.encoder

    @property
    def final_norm(self):
        return self.terramind.encoder_norm

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


class DINOv3(nn.Module):
    """DINOv3 ViT-B/16 pretrained on Meta's LVD-1689M web-image dataset.

    The checkpoint is distributed under Meta's DINOv3 license. The first download may require accepting the
    model terms on Hugging Face and logging in. ``in_chans=1`` lets timm adapt the RGB patch embedding to gray.
    """

    def __init__(self, pretrained: bool):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_dinov3.lvd1689m', pretrained=pretrained,
            img_size=512, in_chans=1, num_classes=0)
        self.embed_dim = 768
        self.patch_size = 16

    @property
    def blocks(self):
        return self.vit.blocks

    @property
    def final_norm(self):
        return self.vit.norm

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        # DINOv3 has a CLS token and storage/register tokens. timm includes all of them in num_prefix_tokens.
        return tokens[:, self.vit.num_prefix_tokens:]


def build_backbone(name, pretrained):
    """Build a supported backbone; ``pretrained=False`` keeps the architecture but uses random weights."""
    return {'vit': ViT, 'terramind': TerraMind, 'dinov3': DINOv3}[name](pretrained)
