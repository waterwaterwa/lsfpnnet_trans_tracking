import torch
from .lsnet_backbone import LSNetBackbone16x

class LSNetV1(LSNetBackbone16x):
    """
    LSNet V1: Lightweight variant using the first 3 stages of LSNet-S.
    - 16x downsampling (4+2+2+1)
    - 3 stages: [96, 192, 320] channels
    - depth: [1, 2, 8]
    """
    def __init__(self, in_chans=3, pretrained=False, **kwargs):
        super().__init__(
            in_chans=in_chans,
            embed_dim=[96, 192, 320],
            key_dim=[16, 16, 16],
            depth=[1, 2, 8, 0],
            num_heads=[3, 3, 3],
            **kwargs
        )

    def forward(self, x):
        return super().forward(x)

def lsnet_v1(**kwargs):
    return LSNetV1(**kwargs)
