"""Customized Vision Transformer"""
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
import torch
from torch import nn
import logging


LOG = logging.getLogger(__name__)


class ViTFixedPE(VisionTransformer):
    """
    Vision Transformer with fixed position encoding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim),
                                      requires_grad=False)  # Load fixed sin-cos embedding [1, 197, 768]


@register_model
def vit_base_fixedpe_patch16_224(pretrained=False, **kwargs):
    model = ViTFixedPE(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                       norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       # drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.1,
                       **kwargs)
    model.default_cfg = _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return model


@register_model
def vit_small_fixedpe_patch16_224(pretrained=False, **kwargs):
    model = ViTFixedPE(patch_size=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                       norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return model


@register_model
def vit_tiny_fixedpe_patch16_224(pretrained=False, **kwargs):
    model = ViTFixedPE(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                       norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return model

