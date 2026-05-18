# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List


from util.misc import NestedTensor

from ltr.models.neck.position_encoding import build_position_encoding

from ltr.models.backbone.lsnet_v1 import LSNetV1
from ltr.models.backbone.lsnet_v3 import LSNetV3
from ltr.models.backbone.lsnet_v2 import LSNetV2

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        # 处理不同类型的输出
        if isinstance(xs, dict):
            # 如果返回的是字典（原来的逻辑）
            for name, x in xs.items():
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
        else:
            # 如果返回的是单一Tensor（新的逻辑）
            x = xs
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out['0'] = NestedTensor(x, mask)

        return out

# class Backbone(BackboneBase):
#     """LSNet backbone V1 (lightweight)."""
#     def __init__(self, pretrained=False):
#
#         backbone = LSNetV1(pretrained=pretrained)
#         num_channels = 320
#         super().__init__(backbone, num_channels)


class Backbone(BackboneBase):
    """LSNet backbone V2 (balanced)."""
    def __init__(self, pretrained=False):

        backbone = LSNetV2(pretrained=pretrained)
        # LSNetV2 output channels: 448
        num_channels = 448
        super().__init__(backbone, num_channels)


# class Backbone(BackboneBase):
#     """LSNet backbone V3 (full)."""
#     def __init__(self, pretrained=False):
#
#         backbone = LSNetV3(pretrained=pretrained)
#         # LSNetV3 output channels: 448
#         num_channels = 448
#         super().__init__(backbone, num_channels)


#  将主干网络和位置编码结合起来的模块
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(settings, backbone_pretrained=True, frozen_backbone_layers=()):
    position_embedding = build_position_encoding(settings)
    # LSNet 只有单输出，Joiner 会将其作为唯一层使用
    backbone = Backbone(pretrained=backbone_pretrained)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


