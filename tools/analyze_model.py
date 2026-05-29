#!/usr/bin/env python
"""
模型分析工具：FLOPs + 参数量 + CAM 热力图

用法：
    # 模型 FLOPs/Params 分析
    python tools/analyze_model.py --mode flops

    # CAM 热力图生成（需要 checkpoint 和测试图片）
    python tools/analyze_model.py --mode cam \
        --checkpoint path/to/checkpoint.pth.tar \
        --image path/to/search.jpg \
        --template path/to/template.jpg

    # 完整分析
    python tools/analyze_model.py --mode all \
        --checkpoint path/to/checkpoint.pth.tar \
        --image path/to/search.jpg \
        --template path/to/template.jpg
"""

import os
import sys
import argparse

# 添加项目根目录到 path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np
import cv2


# ──────────────────────────── FLOPs / Params 分析 ────────────────────────────
def run_flops_analysis(args):
    """运行 FLOPs 和参数量分析"""
    print("\n" + "=" * 60)
    print("  FLOPs & Parameters Analysis")
    print("=" * 60)

    from ltr.models.tracking.sot_model import sot_model
    from ltr.admin.settings import Settings
    from ltr.utils.model_analysis import analyze_model, count_flops_manual, format_flops, format_params

    # 创建 Settings 对象
    settings = Settings()
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.hidden_dim = args.hidden_dim
    settings.dropout = args.dropout
    settings.nheads = args.nheads
    settings.dim_feedforward = args.dim_feedforward
    settings.featurefusion_layers = args.featurefusion_layers
    settings.position_embedding = 'sine'  # 位置编码类型
    settings.search_feature_sz = 18
    settings.template_feature_sz = 8

    print(f"  Device: {settings.device}")
    print(f"  Hidden Dim: {settings.hidden_dim}, Heads: {settings.nheads}")
    print(f"  FFN Dim: {settings.dim_feedforward}, Encoder Layers: {settings.featurefusion_layers}")

    # 构建模型
    print("\n  Building model...")
    model = sot_model(settings)
    model.to(settings.device)
    model.eval()

    # 分析
    results = analyze_model(
        model,
        search_size=(args.search_size, args.search_size),
        template_size=(args.template_size, args.template_size),
        hidden_dim=settings.hidden_dim,
        use_thop=args.use_thop,
        verbose=True,
    )

    # 保存结果
    if args.output:
        import json
        save_dict = {
            'flops': results.get('flops', 0),
            'flops_formatted': results.get('flops_formatted', ''),
            'total_params': results['total_params'],
            'trainable_params': results['trainable_params'],
            'params_formatted': results['params_formatted'],
        }
        with open(args.output, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    return results


# ──────────────────────────── CAM 热力图分析 ────────────────────────────
def run_cam_analysis(args):
    """运行 CAM 热力图分析"""
    print("\n" + "=" * 60)
    print("  CAM Heatmap Analysis")
    print("=" * 60)

    from ltr.admin.settings import Settings
    from ltr.models.tracking.sot_model import sot_model
    from ltr.utils.cam_visualize import (
        generate_cam_heatmap, 
        generate_cross_attn_heatmap,
        CAMVisualizer,
    )
    from ltr.admin.loading import load_network

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构建或加载模型
    if args.checkpoint:
        print(f"\n  Loading checkpoint: {args.checkpoint}")
        net, _ = load_network(args.checkpoint)
        net.to(device)
        net.eval()
    else:
        print("\n  Building model from scratch (no checkpoint)...")
        settings = Settings()
        settings.device = device
        settings.hidden_dim = args.hidden_dim
        settings.dropout = args.dropout
        settings.nheads = args.nheads
        settings.dim_feedforward = args.dim_feedforward
        settings.featurefusion_layers = args.featurefusion_layers
        net = sot_model(settings)
        net.to(device)
        net.eval()

    # 加载图像
    if args.image is None or args.template is None:
        print("\n  [WARN] 未提供图像路径。使用随机噪声演示 CAM。")
        search_np = np.random.randint(0, 255, (args.search_size, args.search_size, 3), dtype=np.uint8)
        template_np = np.random.randint(0, 255, (args.template_size, args.template_size, 3), dtype=np.uint8)
    else:
        print(f"\n  Loading images...")
        print(f"    Search:   {args.image}")
        print(f"    Template: {args.template}")
        search_np = cv2.imread(args.image)
        search_np = cv2.cvtColor(search_np, cv2.COLOR_BGR2RGB)
        search_np = cv2.resize(search_np, (args.search_size, args.search_size))
        template_np = cv2.imread(args.template)
        template_np = cv2.cvtColor(template_np, cv2.COLOR_BGR2RGB)
        template_np = cv2.resize(template_np, (args.template_size, args.template_size))

    # 预处理 → tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def preprocess(img):
        x = img.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)
        return torch.from_numpy(x).unsqueeze(0)

    search_tensor = preprocess(search_np)
    template_tensor = preprocess(template_np)

    save_dir = args.save_dir or 'analysis_results'
    os.makedirs(save_dir, exist_ok=True)

    # ── Grad-CAM ──
    print(f"\n  Generating Grad-CAM heatmap...")
    cam_results = generate_cam_heatmap(
        net,
        search_tensor, template_tensor,
        target_layer=args.cam_layer,
        image_np=search_np,
        save_path=os.path.join(save_dir, 'gradcam.png'),
        cmap=args.cam_cmap,
        alpha=args.cam_alpha,
        device=device,
    )
    print(f"  [OK] Grad-CAM completed.")

    # ── 多尺度 Grad-CAM 网格 ──
    if args.grid:
        print(f"\n  Generating multi-layer Grad-CAM grid...")
        fig = _generate_multiscale_cam(net, search_tensor, template_tensor, search_np, device)
        grid_path = os.path.join(save_dir, 'gradcam_grid.png')
        fig.savefig(grid_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Grid saved: {grid_path}")

    # ── 交叉注意力 ──
    if args.cross_attn:
        print(f"\n  Extracting cross-attention heatmap...")
        attn_results = generate_cross_attn_heatmap(
            net,
            search_tensor, template_tensor,
            image_np=search_np,
            save_path=os.path.join(save_dir, 'cross_attn.png'),
            head_idx=args.attn_head,
            template_pos=args.template_pos,
            device=device,
        )
        print(f"  [OK] Cross-attention map completed.")

    print(f"\n  All results saved to: {save_dir}/")


def _generate_multiscale_cam(model, search, template, image_np, device):
    """生成多尺度 CAM 网格图"""
    import matplotlib.pyplot as plt
    from ltr.utils.cam_visualize import GradCAMHook, CAMVisualizer

    model.eval()
    model.to(device)
    search, template = search.to(device), template.to(device)
    visualizer = CAMVisualizer(cmap='jet', alpha=0.5)

    # 尝试对 backbone 的不同位置做 Grad-CAM
    layers = {'input_proj': model.input_proj}

    # 尝试获取 backbone stages
    try:
        body = model.backbone[0].body
        for name in ['blocks2', 'blocks3', 'blocks4']:
            if hasattr(body, name):
                blocks = getattr(body, name)
                layers[name] = blocks[-1]
    except Exception:
        pass

    N = len(layers)
    cols = min(N, 4)
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    axes = [ax for row in axes for ax in row]

    for idx, (name, layer) in enumerate(layers.items()):
        if idx >= len(axes):
            break
        try:
            hook = GradCAMHook(model, layer)
            hook.register_hooks()
            output = model(search, template)
            pred_logits = output['pred_logits']
            score = pred_logits[0, :, 0].mean()
            model.zero_grad()
            score.backward(retain_graph=True)
            cam = hook.generate()
            hook.remove_hooks()
            overlayed = visualizer.overlay(image_np, cam)
            axes[idx].imshow(overlayed)
            axes[idx].set_title(f'Layer: {name}')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
        axes[idx].axis('off')

    for idx in range(len(layers), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


# ──────────────────────────── 主入口 ────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='模型分析工具：FLOPs + Params + CAM 热力图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模式
    parser.add_argument('--mode', type=str, default='flops',
                        choices=['flops', 'cam', 'all'],
                        help='分析模式: flops (FLOPs/Params), cam (CAM 热力图), all (全部)')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径 (.pth.tar)')

    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Transformer 隐藏维度 (默认 256)')
    parser.add_argument('--nheads', type=int, default=8,
                        help='注意力头数 (默认 8)')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='FFN 中间维度 (默认 1024)')
    parser.add_argument('--featurefusion_layers', type=int, default=2,
                        help='Encoder 层数 (默认 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 值 (默认 0.1)')

    # FLOPs 参数
    parser.add_argument('--search_size', type=int, default=288,
                        help='搜索区域尺寸 (默认 288)')
    parser.add_argument('--template_size', type=int, default=128,
                        help='模板区域尺寸 (默认 128)')
    parser.add_argument('--use_thop', action='store_true', default=True,
                        help='使用 thop 库精确计算 FLOPs (默认 True)')
    parser.add_argument('--no_thop', action='store_false', dest='use_thop',
                        help='使用手动 FLOPs 估算')

    # CAM 参数
    parser.add_argument('--image', type=str, default=None,
                        help='搜索图像路径')
    parser.add_argument('--template', type=str, default=None,
                        help='模板图像路径')
    parser.add_argument('--cam_layer', type=str, default='input_proj',
                        help='CAM hook 目标层 (input_proj / backbone)')
    parser.add_argument('--cam_cmap', type=str, default='jet',
                        help='热力图色彩映射 (jet/hot/cool/inferno/plasma/viridis)')
    parser.add_argument('--cam_alpha', type=float, default=0.5,
                        help='热力图叠加透明度 (0-1)')
    parser.add_argument('--grid', action='store_true',
                        help='生成多尺度 Grad-CAM 网格图')
    parser.add_argument('--cross_attn', action='store_true',
                        help='提取交叉注意力热力图')
    parser.add_argument('--attn_head', type=int, default=0,
                        help='交叉注意力的 head 索引')
    parser.add_argument('--template_pos', type=int, default=32,
                        help='交叉注意力的模板 token 位置 (8x8 网格中心=32)')

    # 通用
    parser.add_argument('--output', type=str, default=None,
                        help='FLOPs/Params 分析结果输出 JSON 路径')
    parser.add_argument('--save_dir', type=str, default='analysis_results',
                        help='CAM 热力图保存目录')

    args = parser.parse_args()

    if args.mode in ('flops', 'all'):
        run_flops_analysis(args)

    if args.mode in ('cam', 'all'):
        run_cam_analysis(args)


if __name__ == '__main__':
    main()
