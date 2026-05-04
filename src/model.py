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
from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, trunc_normal_
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from binaryattn import Block, resize_pos_embed
from utils import _assert_strides_are_log2_contiguous

VARIANTS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
}

ShapeSpec = namedtuple("ShapeSpec", ["channels", "stride"])


def _ln2d(num_channels: int) -> nn.Module:
    """Channel-first LayerNorm for (B, C, H, W) feature maps (==GroupNorm with 1 group)."""
    return nn.GroupNorm(1, num_channels)


class LastLevelMaxPool(nn.Module):
    """Detectron2-style top block: subsample one FPN level by stride-2 max-pool to add a coarser level."""

    def __init__(self, in_feature: str = "p5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class BinaryViTBackbone(nn.Module):
    """ViT feature extractor with binary QK-attention.

    Returns ``{"last_feat": [B, embed_dim, H/patch, W/patch]}`` — a single-feature
    dict so the module slots into either a ``SimpleFeaturePyramid`` (which keys by
    ``in_feature``) or directly into ``FasterRCNN`` (which forwards dicts through
    to the RPN/ROI pool).
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

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {"last_feat": ShapeSpec(channels=self.out_channels, stride=self.patch_size)}

    def _resized_pos_embed(self, h: int, w: int) -> torch.Tensor:
        gh, gw = self.grid_size
        if (h, w) == (gh, gw):
            return self.pos_embed
        cls_pe, patch_pe = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        pe = patch_pe.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat([cls_pe, pe], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self._resized_pos_embed(h, w)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 1:].transpose(1, 2).reshape(B, -1, h, w)
        return {"last_feat": x}


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        :param net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
        :param in_feature (str): names of the input feature maps coming
                from the net.
        :param out_channels (int): number of channels in the output feature maps.
        :param scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        :param top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
        :param norm (str): the normalization to use.
        :param square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [
            int(input_shapes[in_feature].stride / scale) for scale in scale_factors
        ]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    _ln2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=use_bias),
                    _ln2d(out_channels),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
                    ),
                    _ln2d(out_channels),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self.out_channels = out_channels

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        :param x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


# class BinaryViTBackboneWithFPN(nn.Module):
#     """``BinaryViTBackbone`` + ``SimpleFeaturePyramid`` for FasterRCNN with FPN."""

#     def __init__(self, body: BinaryViTBackbone, fpn_out_channels: int = 256):
#         super().__init__()
#         self.body = body
#         self.fpn = SimpleFeaturePyramid(
#             body,
#             in_feature="last_feat",
#             out_channels=256,
#             scale_factors=(4.0, 2.0, 1.0, 0.5),
#             top_block=LastLevelMaxPool(),
#             norm="LN",
#             square_pad=512,
#         )
#         self.out_channels = fpn_out_channels

#     def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
#         return self.fpn(self.body(x))


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
    if hasattr(backbone, "net") and isinstance(backbone.net, BinaryViTBackbone):
        backbone = backbone.net

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
        backbone = SimpleFeaturePyramid(
            body,
            in_feature="last_feat",
            out_channels=fpn_out_channels,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=LastLevelMaxPool(in_feature="p5"),
            norm="LN",
            square_pad=img_size,
        )
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in _scale((32, 64, 128, 256, 512))),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"], output_size=7, sampling_ratio=2
        )
    else:
        backbone = body
        anchor_generator = AnchorGenerator(
            sizes=(_scale((32, 64, 128, 256, 512)),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        roi_pool = MultiScaleRoIAlign(
            featmap_names=["last_feat"], output_size=7, sampling_ratio=2
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
