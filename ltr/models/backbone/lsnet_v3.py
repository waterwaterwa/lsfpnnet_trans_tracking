import torch
from .lsnet_backbone import LSNetBackbone16x

class LSNetV3(LSNetBackbone16x):
    """
    LSNet V3: Full variant using complete LSNet-S 4 stages.
    - 16x downsampling (4+2+2+1)
    - 4 stages: [96, 192, 320, 448] channels
    - depth: [1, 2, 8, 8]
    """
    def __init__(self, in_chans=3, pretrained=False, **kwargs):
        super().__init__(
            in_chans=in_chans,
            embed_dim=[96, 192, 320, 448],
            key_dim=[16, 16, 16, 16],
            depth=[1, 2, 8, 8],
            num_heads=[3, 3, 3, 4],
            **kwargs
        )

    def forward(self, x):
        return super().forward(x)

def lsnet_v3(**kwargs):
    return LSNetV3(**kwargs)
