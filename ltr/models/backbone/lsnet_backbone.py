import torch
import itertools
import math
import torch.nn as nn
import torch.nn.functional as F

# 替代 timm 的 trunc_normal_ 函数
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 基于 PyTorch 的原始实现
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

# 替代 timm 的 SqueezeExcite
class SqueezeExcite(torch.nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, int(channels // reduction))
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(channels, reduced_channels, 1)
        self.act = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(reduced_channels, channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

# 替代 timm 的 register_model 装饰器
def register_model(func):
    return func

# 替代 timm 的 build_model_with_cfg
def build_model_with_cfg(model_cls, variant, pretrained, **kwargs):
    return model_cls(**kwargs)

# 替代 timm 的 IMAGENET_DEFAULT_MEAN 和 IMAGENET_DEFAULT_STD
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

from .ska import SKA


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

        # 这种 Attention 只能接受固定大小的输入，不能动态改变分辨率！！！


        # 在LSNetbackbone中，需要根据stage输入尺寸动态设置Attention的resolution：


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4.0):
        """
        Attention supporting two fixed feature resolutions: 8x8 (template) and 18x18 (search).
        - bias parameters are registered so they are saved/loaded/optimized.
        - idx buffers are registered to move with model.to(device).
        """
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = float(key_dim) ** -0.5
        self.key_dim = int(key_dim)
        self.nh_kd = self.key_dim * self.num_heads
        self.attn_ratio = float(attn_ratio)
        self.d = int(self.attn_ratio * self.key_dim)
        self.dh = self.d * self.num_heads

        h = self.dh + self.nh_kd * 2
        # qkv conv and projection
        self.qkv = Conv2d_BN(dim, h, ks=1)
        try:
            self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(self.dh, dim, ks=1, bn_weight_init=0))
        except Exception:
            self.proj = nn.Sequential(nn.ReLU(), nn.Conv2d(self.dh, dim, kernel_size=1, bias=False), nn.BatchNorm2d(dim))

        # depthwise conv on q
        try:
            self.dw = Conv2d_BN(self.nh_kd, self.nh_kd, 3, 1, 1, groups=self.nh_kd)
        except Exception:
            self.dw = nn.Sequential(nn.Conv2d(self.nh_kd, self.nh_kd, kernel_size=3, padding=1, groups=self.nh_kd, bias=False),
                                    nn.BatchNorm2d(self.nh_kd))
#                   构建Attention模块，支持8x8和18x18两种分辨率
        # Build and register biases & idx buffers for supported resolutions
        for res in (8, 18):
            points = [(i, j) for i in range(res) for j in range(res)]
            N = res * res
            attn_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    off = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if off not in attn_offsets:
                        attn_offsets[off] = len(attn_offsets)
                    idxs.append(attn_offsets[off])

            idxs_tensor = torch.LongTensor(idxs).view(N, N)  # shape (N, N)
            # register idxs as buffer
            self.register_buffer(f'attn_idxs_{res}', idxs_tensor, persistent=True)
            # create biases as Parameter (will be saved/optimized)
            biases = nn.Parameter(torch.zeros(self.num_heads, len(attn_offsets)))
            self.register_parameter(f'attn_biases_{res}', biases)

        # For safety, expose supported resolutions
        self.supported_resolutions = (8,18)

    def forward(self, x):

        B, C, H, W = x.shape
        if H != W:
            raise ValueError(f"Attention requires square feature map; got {H}x{W}")

        res = H  # 允许任意分辨率
        # 不再对 res 做限制

        # if H != W:
        #     raise ValueError(f"Attention requires square feature map; got {H}x{W}")
        # res = int(H)
        # if res not in self.supported_resolutions:
        #     raise ValueError(f"Attention only supports feature maps of sizes {self.supported_resolutions}, got {res}x{res}")

        # load idxs buffer and bias parameter
        idxs = getattr(self, f'attn_idxs_{res}')  # buffer -> already on device with model.to()
        # idxs = self.get_attn_idxs(res, x.device)

        biases = getattr(self, f'attn_biases_{res}')  # Parameter -> on model device

        qkv = self.qkv(x)
        # split q,k,v: [nh_kd, nh_kd, dh]
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)

        # depthwise on q
        q = self.dw(q)

        N = res * res
        # reshape for multi-head: (B, heads, head_dim, N)
        q = q.view(B, self.num_heads, -1, N)
        k = k.view(B, self.num_heads, -1, N)
        v = v.view(B, self.num_heads, -1, N)

        attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, heads, N, N)

        # biases is (heads, num_offsets); idxs maps N×N -> offsets index
        # gather per-head biases for N×N positions
        # idxs is registered buffer (LongTensor) -> already correct device
        attn = attn + biases[:, idxs].unsqueeze(0)  # broadcast over batch dim

        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        out = self.proj(out)
        return out

    # helper so optimizer exclusion for weight decay can find parameter names
    def bias_parameter_names(self):
        return [f'attn_biases_{r}' for r in self.supported_resolutions]




class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


import torch.nn as nn


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class Block(nn.Module):
    def __init__(self, ed, kd, nh=8, ar=4, stage=-1, depth=-1):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = nn.Identity()
            if stage == 3:
                # Attention 本身不是 Residual，这里直接使用 Attention，然后外层的 FFN 为 Residual
                self.mixer = Attention(ed, kd, num_heads=nh, attn_ratio=ar)
            else:
                self.mixer = LSConv(ed)
        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


#  三阶段特征连接：统一分辨率
class LSNetThreeStageAlign(nn.Module):
    """
    将 LSNet 的 Stage1, Stage2, Stage3 统一调整为：1/16   stage3  target_dim (默认 320)

    输入：
        stage1_out: (B, C1, H/4,  W/4)
        stage2_out: (B, C2, H/8,  W/8)
        stage3_out: (B, C3, H/16, W/16)

    输出：
        p1, p2, p3 （全部为 (B, target_dim, H/16, W/16)）
    """

    def __init__(self, c1, c2, c3, target_dim=320):
        super().__init__()
        self.lateral1 = nn.Conv2d(c1, target_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, target_dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, target_dim, kernel_size=1)

    def forward(self, s1, s2, s3):
        # s1: 1/4 -> 1/16
        p1 = F.interpolate(s1, scale_factor=1 / 4, mode='bilinear', align_corners=False)
        # s2: 1/8 -> 1/16
        p2 = F.interpolate(s2, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        p3 = s3
        p1 = self.lateral1(p1)
        p2 = self.lateral2(p2)
        p3 = self.lateral3(p3)
        return p1, p2, p3

    #img_size=224=14x14
class LSNetBackbone16x(nn.Module):
    def __init__(self, patch_size=16, in_chans=3,
                 embed_dim=[96, 192, 320, 448],
                 key_dim=[16, 16, 16, 16],
                 depth=[1, 2, 8, 10],
                 num_heads=[3, 3, 3, 4],
                 distillation=False):
        super().__init__()

        # patch_embed: total 4x downsample (-> 1/4)
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1)
        )

        attn_ratio = [float(embed_dim[i]) / (float(key_dim[i]) * float(num_heads[i])) for i in range(len(embed_dim))]

        # blocks1..4 without resolution argument
        self.blocks1 = nn.Sequential(*[
            Block(embed_dim[0], key_dim[0], num_heads[0], attn_ratio[0], stage=0, depth=d)
            for d in range(depth[0])
        ])

        self.blocks2 = nn.Sequential(*[
            Block(embed_dim[1], key_dim[1], num_heads[1], attn_ratio[1], stage=1, depth=d)
            for d in range(depth[1] if len(depth) > 1 else 0)
        ])

        self.blocks3 = nn.Sequential(*[
            Block(embed_dim[2], key_dim[2], num_heads[2], attn_ratio[2], stage=2, depth=d)
            for d in range(depth[2] if len(depth) > 2 else 0)
        ])

        self.blocks4 = nn.Sequential(*[
            Block(embed_dim[3], key_dim[3], num_heads[3], attn_ratio[3], stage=3, depth=d)
            for d in range(depth[3] if len(depth) > 3 else 0)
        ])

        # add downsample layers between stages (depth>1 cases)
        if len(depth) > 1:
            downsample_layers_1to2 = nn.Sequential(
                Conv2d_BN(embed_dim[0], embed_dim[0], ks=3, stride=2, pad=1, groups=embed_dim[0]),
                Conv2d_BN(embed_dim[0], embed_dim[1], ks=1, stride=1, pad=0)
            )
            self.blocks2 = nn.Sequential(downsample_layers_1to2, *list(self.blocks2))

        if len(depth) > 2:
            downsample_layers_2to3 = nn.Sequential(
                Conv2d_BN(embed_dim[1], embed_dim[1], ks=3, stride=2, pad=1, groups=embed_dim[1]),
                Conv2d_BN(embed_dim[1], embed_dim[2], ks=1, stride=1, pad=0)
            )
            self.blocks3 = nn.Sequential(downsample_layers_2to3, *list(self.blocks3))

        if len(depth) > 3:
            downsample_layers_3to4 = nn.Sequential(
                Conv2d_BN(embed_dim[2], embed_dim[3], ks=1, stride=1, pad=0)
            )
            self.blocks4 = nn.Sequential(downsample_layers_3to4, *list(self.blocks4))

        self.num_features = embed_dim[-1]

        # three-stage align & fusion
        self.align3stages = LSNetThreeStageAlign(
            c1=embed_dim[0], c2=embed_dim[1], c3=embed_dim[2], target_dim=embed_dim[2]
        )
        self.fuse_se = SqueezeExcite(embed_dim[2])
        self.fuse_lsconv = LSConv(embed_dim[2])

    def forward(self, x):
        # Input x is expected (B,3,H_in,W_in), where H_in may be 128 or 288 (or other)
        x = self.patch_embed(x)           # -> 1/4
        s1 = self.blocks1(x)              # stage1: 1/4
        s2 = self.blocks2(s1)             # stage2: 1/8
        s3 = self.blocks3(s2)             # stage3: 1/16
        p1, p2, p3 = self.align3stages(s1, s2, s3)  # unify to 1/16
        fused = p1 + p2 + p3
        fused = self.fuse_se(fused)
        fused = self.fuse_lsconv(fused)
        x = self.blocks4(fused)           # stage4 consumer (keeps 1/16)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        # ensure attention biases are excluded from weight decay if you want
        names = set()
        # find registered bias names from any Attention instance (if present in state_dict)
        for n in self.state_dict().keys():
            if 'attn_biases_' in n:
                names.add(n.split('.')[0])  # top-level module name containing param
        return names
