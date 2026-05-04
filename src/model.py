"""BinaryAttention ViT backbone wrapped in torchvision Faster R-CNN.

The backbone reuses the ``Block`` module from ``binaryattn.py`` (binary QK-attention
with optional 8-bit P/V quantization) and exposes a single-scale ``H/16 x W/16``
feature map suitable for ``torchvision.models.detection.FasterRCNN``.

Input images are forced to a fixed square size by ``GeneralizedRCNNTransform``
(``fixed_size``), which keeps the patch-embed grid constant — no per-batch
positional-embedding interpolation needed for training.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, trunc_normal_
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from binaryattn import Block, resize_pos_embed

VARIANTS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
}


class BinaryViTBackbone(nn.Module):
    """ViT feature extractor with binary QK-attention.

    Returns a ``[B, embed_dim, H/patch, W/patch]`` feature map.  ``out_channels``
    is set so the module plugs directly into ``FasterRCNN``.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        attn_quant: bool = True,
        pv_quant: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim
        )
        self.grid_size = self.patch_embed.grid_size  # (gh, gw)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attn_quant=attn_quant,
                    pv_quant=pv_quant,
                    attn_bias=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.out_channels = embed_dim

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _resized_pos_embed(self, h: int, w: int) -> torch.Tensor:
        gh, gw = self.grid_size
        if (h, w) == (gh, gw):
            return self.pos_embed
        cls_pe, patch_pe = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        pe = patch_pe.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pe, pe], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self._resized_pos_embed(h, w)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 1:].transpose(1, 2).reshape(B, -1, h, w)
        return x


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style simple feature pyramid (Li et al., 2022).

    Builds a 4-level pyramid (strides 4, 8, 16, 32) from a single stride-16 ViT
    feature map by per-level rescaling (transposed convs / pool) followed by a
    1x1 + 3x3 conv tail.  GroupNorm chosen to be small-batch friendly.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        def gn(c: int) -> nn.Module:
            return nn.GroupNorm(32, c)

        self.scale_ops = nn.ModuleList(
            [
                # stride 4: 4x upsample
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, 2, 2),
                    gn(in_channels),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels, in_channels, 2, 2),
                ),
                # stride 8: 2x upsample
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2),
                # stride 16: identity
                nn.Identity(),
                # stride 32: 2x downsample
                nn.MaxPool2d(2, 2),
            ]
        )
        self.tails = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    gn(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    gn(out_channels),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for i, (op, tail) in enumerate(zip(self.scale_ops, self.tails)):
            out[str(i)] = tail(op(x))
        return out


class BinaryViTBackboneWithFPN(nn.Module):
    """``BinaryViTBackbone`` + ``SimpleFeaturePyramid`` for FasterRCNN with FPN."""

    def __init__(self, body: BinaryViTBackbone, fpn_out_channels: int = 256):
        super().__init__()
        self.body = body
        self.fpn = SimpleFeaturePyramid(body.out_channels, fpn_out_channels)
        self.out_channels = fpn_out_channels

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        return self.fpn(self.body(x))


def load_pretrained_backbone(
    backbone: nn.Module, ckpt_path: str
) -> tuple[list[str], list[str]]:
    """Load an ImageNet-1K pretrained BinaryAttention checkpoint into the backbone.

    Drops the classifier head and the (input-size-tied) relative position bias,
    then bicubically interpolates ``pos_embed`` from the pretraining grid
    (14x14 @ 224) to the detection grid (e.g. 32x32 @ 512).  Accepts either a
    bare ``BinaryViTBackbone`` or a ``BinaryViTBackboneWithFPN`` wrapper — only
    the ViT body is touched; the FPN is left at its random init.

    Returns ``(missing_keys, unexpected_keys)`` from ``load_state_dict``.
    """
    if hasattr(backbone, "body") and isinstance(backbone.body, BinaryViTBackbone):
        backbone = backbone.body

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    pe_w = sd.get("patch_embed.proj.weight")
    if pe_w is not None and pe_w.shape[0] != backbone.out_channels:
        raise ValueError(
            f"checkpoint embed_dim={pe_w.shape[0]} != backbone embed_dim="
            f"{backbone.out_channels} (variant mismatch?)"
        )

    drop_prefixes = ("head.", "head_dist.", "dist_token")
    sd = {
        k: v
        for k, v in sd.items()
        if not any(k.startswith(p) for p in drop_prefixes)
        and "relative_position" not in k
    }

    if "pos_embed" in sd and sd["pos_embed"].shape != backbone.pos_embed.shape:
        sd["pos_embed"] = resize_pos_embed(
            sd["pos_embed"],
            backbone.pos_embed,
            num_tokens=1,
            gs_new=backbone.grid_size,
        )

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    return list(missing), list(unexpected)


def build_faster_rcnn(
    variant: str = "tiny",
    num_classes: int = 21,
    img_size: int = 512,
    attn_quant: bool = True,
    pv_quant: bool = True,
    drop_path_rate: float = 0.1,
    use_fpn: bool = True,
    fpn_out_channels: int = 256,
    anchor_scale: float = 1.0,
) -> FasterRCNN:
    """Faster R-CNN with a BinaryAttention-T/S/B backbone.

    ``use_fpn=True`` (default) wraps the ViT in a ViTDet-style simple feature
    pyramid producing 4 levels at strides 4/8/16/32. ``use_fpn=False`` falls
    back to a single-scale stride-16 feature with multi-size anchors.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    if img_size % 16 != 0:
        raise ValueError("img_size must be divisible by patch_size (16)")

    body = BinaryViTBackbone(
        img_size=img_size,
        attn_quant=attn_quant,
        pv_quant=pv_quant,
        drop_path_rate=drop_path_rate,
        **VARIANTS[variant],
    )

    def _scale(sizes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(max(1, round(s * anchor_scale)) for s in sizes)

    if use_fpn:
        backbone = BinaryViTBackboneWithFPN(body, fpn_out_channels=fpn_out_channels)
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in _scale((32, 64, 128, 256))),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )
        roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
    else:
        backbone = body
        anchor_generator = AnchorGenerator(
            sizes=(_scale((32, 64, 128, 256, 512)),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        min_size=img_size,
        max_size=img_size,
        fixed_size=(img_size, img_size),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
