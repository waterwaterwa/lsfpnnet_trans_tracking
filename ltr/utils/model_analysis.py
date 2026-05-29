"""
模型 FLOPs 和参数量计算模块

支持两种方式：
1. thop 库（推荐）：自动统计 FLOPs 和 Params，支持 CNN + Transformer
2. 纯 PyTorch 手动计算（无额外依赖）
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from collections import OrderedDict


# ───────────────────────── 方式一：thop 库 ─────────────────────────
def _try_import_thop():
    try:
        import thop
        return thop
    except ImportError:
        return None


def count_flops_thop(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    使用 thop 库计算 FLOPs 和参数量

    Args:
        model: 模型
        inputs: 输入字典，key 为参数名，value 为 Tensor
        verbose: 是否打印详细信息

    Returns:
        (total_flops, total_params): FLOPs 数和参数量
    """
    thop = _try_import_thop()
    if thop is None:
        raise ImportError("请先安装 thop: pip install thop")

    # thop.profile 需要一个无参数 forward 函数
    input_tensors = list(inputs.values())

    class ModelWrapper(nn.Module):
        def __init__(self, model, inputs):
            super().__init__()
            self.model = model
            self.input_keys = list(inputs.keys())
            self.inputs = inputs

        def forward(self):
            kwargs = {k: v.cuda() if v.device.type != 'cuda' else v
                       for k, v in self.inputs.items()}
            # 只支持字典解包
            return self.model(**kwargs)

    wrapper = ModelWrapper(model, inputs)
    total_ops, total_params = thop.profile(
        model, inputs=tuple(input_tensors), verbose=verbose
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Total FLOPs:  {format_flops(total_ops)}")
        print(f"  Total Params: {format_params(total_params)}")
        print(f"{'='*60}")

    return total_ops, total_params


# ───────────────────────── 方式二：纯 PyTorch 手动计算 ─────────────────────────
def count_params(
    model: nn.Module,
    format_str: bool = True,
    include_grad: bool = False,
) -> Tuple[int, int]:
    """
    纯 PyTorch 计算模型参数量（不依赖任何外部库）

    Args:
        model: 模型
        format_str: 是否返回格式化字符串
        include_grad: 是否只统计需要梯度的参数

    Returns:
        (total_params, trainable_params)
    """
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num

    if not include_grad:
        return total

    return total, trainable


def count_flops_manual(
    model: nn.Module,
    img_size: Tuple[int, int] = (288, 288),
    hidden_dim: int = 256,
    num_encoder_layers: int = 2,
    num_heads: int = 8,
    dim_feedforward: int = 1024,
    backbone_channels: int = 448,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    手动估算模型 FLOPs（解析计算，不依赖 thop）

    适用于无法使用 thop 的特殊环境。基于模型结构公式化估算。
    注：此为近似值，精确值请使用 thop。

    Returns:
        dict: 各模块 FLOPs 估算值 (单位: MFLOPs)
    """
    H, W = img_size
    h = H // 16  # backbone 16x 下采样后的特征图大小
    w = W // 16

    N_search = h * w   # 搜索区域 token 数
    N_template = 64    # 模板 token 数 (128/16=8, 8x8)

    results = OrderedDict()

    # ── LSNet Backbone ──
    # 基于 LSNet-S 的经验值，约 0.2 GFLOPs @ 224x224
    scale_factor = (H * W) / (224 * 224)
    backbone_flops = 0.2 * scale_factor * 1e3  # GFLOPs -> MFLOPs
    results['LSNet Backbone'] = backbone_flops

    # ── Input Projection (1x1 Conv) ──
    proj_flops = 2 * backbone_channels * hidden_dim * N_search * 2  # x2 for search+template
    results['Input Projection'] = proj_flops / 1e6

    # ── Feature Fusion Encoder ──
    # 每层：2x self-attn + 2x cross-attn + 2x FFN
    enc_per_layer = 0
    # Self-attention (template + search)
    for N in [N_template, N_search]:
        # QKV projection
        sa_qkv = 2 * N * hidden_dim * hidden_dim * 3
        # Attention computation
        sa_attn = 2 * N * N * hidden_dim
        # Output projection
        sa_out = N * hidden_dim * hidden_dim
        enc_per_layer += sa_qkv + sa_attn + sa_out

    # Cross-attention (template->search + search->template)
    ca_qkv = (2 * N_search * hidden_dim * hidden_dim +
              2 * N_template * hidden_dim * hidden_dim * 2)  # search-q, template-kv
    ca_attn = 2 * N_search * N_template * hidden_dim * 2  # bidirectional
    ca_out = (N_search + N_template) * hidden_dim * hidden_dim
    enc_per_layer += ca_qkv + ca_attn + ca_out

    # FFN (template + search)
    for N in [N_template, N_search]:
        ffn = 2 * N * hidden_dim * dim_feedforward * 2  # linear1 + linear2
        enc_per_layer += ffn

    enc_flops = enc_per_layer * num_encoder_layers / 1e6
    results[f'Encoder ({num_encoder_layers} layers)'] = enc_flops

    # ── Decoder ──
    # Cross-attention (search query, template key/value)
    dec_qkv = (N_search * hidden_dim * hidden_dim +
               2 * N_template * hidden_dim * hidden_dim)
    dec_attn = 2 * N_search * N_template * hidden_dim
    dec_out = N_search * hidden_dim * hidden_dim
    dec_ffn = 2 * N_search * hidden_dim * dim_feedforward * 2
    dec_flops = (dec_qkv + dec_attn + dec_out + dec_ffn) / 1e6
    results['Decoder'] = dec_flops

    # ── Prediction Heads ──
    head_flops = (2 * N_search * hidden_dim * hidden_dim * 3 * 2) / 1e6  # class + bbox
    results['Prediction Heads'] = head_flops

    total = sum(results.values())
    results['TOTAL'] = total

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Manual FLOPs Estimation (@ {H}x{W} input)")
        print(f"{'='*60}")
        for name, flops in results.items():
            print(f"  {name:.<40s} {flops:>10.2f} MFLOPs")
        print(f"{'='*60}")

    return results


# ───────────────────────── 辅助函数 ─────────────────────────
def format_flops(flops: float) -> str:
    """格式化 FLOPs 数值"""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops / 1e3:.2f} KFLOPs"


def format_params(params: float) -> str:
    """格式化参数量"""
    if params >= 1e6:
        return f"{params / 1e6:.2f} M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f} K"
    else:
        return f"{params:.0f}"


def param_breakdown(model: nn.Module, top_k: int = 10) -> OrderedDict:
    """
    逐模块参数统计

    Args:
        model: 模型
        top_k: 返回参数最多的 top_k 个模块

    Returns:
        OrderedDict: 模块名 → 参数量
    """
    breakdown = {}
    for name, module in model.named_modules():
        # 跳过叶子节点以上的容器
        params = sum(p.numel() for p in module.parameters())
        if params > 0 and len(list(module.children())) == 0:
            breakdown[name] = params

    # 按参数量降序排列
    sorted_items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
    return OrderedDict(sorted_items[:top_k])


def analyze_model(
    model: nn.Module,
    search_size: Tuple[int, int] = (288, 288),
    template_size: Tuple[int, int] = (128, 128),
    hidden_dim: int = 256,
    use_thop: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    一站式模型分析：FLOPs + 参数量 + 逐模块统计

    Args:
        model: SOTTracker 模型
        search_size: 搜索区域尺寸 (H, W)
        template_size: 模板区域尺寸 (H, W)
        hidden_dim: Transformer 隐藏维度
        use_thop: 是否使用 thop 精确计算
        verbose: 是否打印

    Returns:
        dict: 包含 flops, params, flops_per_module 等
    """
    total_params, trainable_params = count_params(model)
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_formatted': format_params(total_params),
    }

    # thop FLOPs 计算
    if use_thop:
        thop = _try_import_thop()
        if thop is not None:
            try:
                # 创建 dummy inputs
                search = torch.randn(1, 3, *search_size)
                template = torch.randn(1, 3, *template_size)
                flops, _ = thop.profile(model, inputs=(search, template), verbose=False)
                results['flops'] = flops
                results['flops_formatted'] = format_flops(flops)

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"  Model Analysis (thop)")
                    print(f"{'='*60}")
                    print(f"  FLOPs:            {format_flops(flops)}")
                    print(f"  Total Params:     {format_params(total_params)}")
                    print(f"  Trainable Params: {format_params(trainable_params)}")
                    print(f"{'='*60}")
                return results
            except Exception as e:
                if verbose:
                    print(f"  [WARN] thop profiling failed: {e}")
                    print(f"  [INFO] Falling back to manual estimation...")

    # 手动 FLOPs 估算
    manual_results = count_flops_manual(
        model, img_size=search_size,
        hidden_dim=hidden_dim,
        verbose=verbose,
    )

    results['flops'] = manual_results['TOTAL'] * 1e6
    results['flops_formatted'] = f"{manual_results['TOTAL']:.2f} MFLOPs (estimated)"
    results['flops_per_module'] = manual_results

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Model Analysis (Manual)")
        print(f"{'='*60}")
        print(f"  FLOPs (estimated): {results['flops_formatted']}")
        print(f"  Total Params:      {format_params(total_params)}")
        print(f"  Trainable Params:  {format_params(trainable_params)}")
        print(f"{'='*60}")

        # 逐模块参数统计
        breakdown = param_breakdown(model)
        if breakdown:
            print(f"\n  Top modules by parameters:")
            for name, params in breakdown.items():
                print(f"    {name:.<50s} {format_params(params)}")

    results['param_breakdown'] = param_breakdown(model)
    return results


if __name__ == '__main__':
    print("FLOPs / Params 计算模块已就绪。")
    print("使用方法：")
    print("  from ltr.utils.model_analysis import analyze_model, count_params, count_flops_manual")
    print("  python tools/analyze_model.py --thop   # 精确计算（需要 thop）")
    print("  python tools/analyze_model.py          # 手动估算（无需额外依赖）")
