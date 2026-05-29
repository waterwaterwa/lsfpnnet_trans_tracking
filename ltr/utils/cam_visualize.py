"""
CAM (Class Activation Map) 热力图可视化模块

支持两种模式：
1. Grad-CAM: 基于梯度反向传播，可视化模型关注的区域
2. Cross-Attn Map: 提取特征融合网络中的交叉注意力权重，可视化模板引导下的搜索区域关注

Hook 点说明：
- backbone 搜索分支输出：提取原始特征图用于 Grad-CAM
- DecoderCFALayer 交叉注意力：提取模板→搜索的 attention map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Dict
import os


# ──────────────────────────── 自定义热力图颜色映射 ────────────────────────────
_CMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'cool': cv2.COLORMAP_COOL,
    'plasma': cv2.COLORMAP_PLASMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'viridis': cv2.COLORMAP_VIRIDIS,
}


# ──────────────────────────── Grad-CAM Hook ────────────────────────────
class GradCAMHook:
    """Grad-CAM hook，用于捕获目标层的前向特征和反向梯度"""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: 完整的跟踪模型 (SOTTracker)
            target_layer: 目标卷积层（如 input_proj 或 backbone 的某个 stage）
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        self._hooks.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(self, class_idx: Optional[int] = None) -> np.ndarray:
        """
        生成 Grad-CAM 热力图

        Args:
            class_idx: 目标类别索引（None 时使用 score 最大的类别）

        Returns:
            np.ndarray: shape (H, W)，值归一化到 [0, 1] 的热力图
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("请先执行前向和反向传播后再生成热力图")

        # 全局平均池化梯度 → 得到每个通道的权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # 加权组合激活图
        cam = (weights * self.activations).sum(dim=1)  # [B, H, W]
        cam = F.relu(cam)  # 只保留正贡献

        # 归一化到 [0, 1]
        for i in range(cam.shape[0]):
            _min = cam[i].min()
            _max = cam[i].max()
            if _max > _min:
                cam[i] = (cam[i] - _min) / (_max - _min)

        return cam[0].cpu().numpy()


class CAMVisualizer:
    """CAM 热力图可视化工具"""

    def __init__(self, cmap: str = 'jet', alpha: float = 0.5):
        """
        Args:
            cmap: OpenCV 色彩映射名称 ('jet', 'hot', 'cool', 'inferno', 'plasma', 'viridis')
            alpha: 热力图叠加透明度
        """
        self.cmap = cmap
        self.alpha = alpha

    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """将 CAM 缩放到目标尺寸"""
        return cv2.resize(cam, (target_size[1], target_size[0]))

    def _apply_colormap(self, cam: np.ndarray) -> np.ndarray:
        """对热力图应用色彩映射"""
        cam_uint8 = np.uint8(cam * 255)
        return cv2.applyColorMap(cam_uint8, _CMAPS.get(self.cmap, cv2.COLORMAP_JET))

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        将热力图叠加到原始图像上

        Args:
            image: 原始图像 (H, W, C) RGB 格式
            cam: CAM 热力图 (H, W) 值域 [0, 1]
            target_size: 目标输出尺寸，None 则使用原图尺寸

        Returns:
            np.ndarray: 叠加后的图像 (H, W, C) RGB
        """
        if target_size is not None:
            cam = self._resize_cam(cam, target_size)
        else:
            cam = self._resize_cam(cam, (image.shape[0], image.shape[1]))

        heatmap = self._apply_colormap(cam)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
        overlayed = cv2.addWeighted(image_bgr, 1 - self.alpha, heatmap, self.alpha, 0)
        return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

    def save_comparison(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        save_path: str,
        title: str = "CAM Visualization",
    ):
        """
        保存对比图：原图 + 热力图叠加

        Args:
            image: 原始图像 (H, W, C) RGB
            cam: CAM 热力图
            save_path: 保存路径
            title: 图片标题
        """
        overlayed = self.overlay(image, cam)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(overlayed)
        axes[1].set_title(title)
        axes[1].axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [CAM] 对比图已保存至: {save_path}")


# ──────────────────────────── 交叉注意力热力图 ────────────────────────────
class CrossAttnExtractor:
    """提取 DecoderCFALayer 中交叉注意力的权重，作为模板引导的注意力热力图"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.attn_weights = None
        self._hooks = []

    def _attn_hook(self, module, input, output):
        """在 DecoderCFALayer.forward_post 中提取 multihead_attn 的权重"""
        pass

    def register_hooks(self):
        """注册到 DecoderCFALayer 的交叉注意力层"""
        ff_net = self.model.featurefusion_network
        decoder_layer = ff_net.decoder.layers[0]

        def hook_fn(module, input, output):
            # 在 DecoderCFALayer.forward_post 后捕获 attn weights
            # 通过重新计算获取 attention weights
            pass

        # 注册到 multihead_attn
        def attn_hook(module, input, output):
            self.attn_weights = output[1]  # attn_output_weights

        self._hooks.append(
            decoder_layer.multihead_attn.register_forward_hook(attn_hook)
        )

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_attention_map(self, head_idx: int = 0, template_pos: int = 0) -> Optional[np.ndarray]:
        """
        提取特定 attention head 的热力图

        Args:
            head_idx: attention head 索引
            template_pos: 模板序列中关注的 token 位置（0 为中心 token）

        Returns:
            np.ndarray: 搜索区域的注意力图，shape (sqrt_N, sqrt_N)
        """
        if self.attn_weights is None:
            return None
        # attn_weights: [batch*nheads, search_len, template_len]
        # 取平均 head 或指定 head
        nheads = self.model.featurefusion_network.nhead
        attn = self.attn_weights  # [B*nhead, L_search, L_template]
        attn = attn.view(-1, nheads, attn.shape[1], attn.shape[2])
        attn = attn[:, head_idx, :, :]  # [B, L_search, L_template]
        attn = attn[:, :, template_pos]  # [B, L_search]  关注第 template_pos 个模板 token

        L = attn.shape[1]
        side = int(np.sqrt(L))
        attn_map = attn.view(-1, side, side)[0].cpu().numpy()

        # 归一化
        _min = attn_map.min()
        _max = attn_map.max()
        if _max > _min:
            attn_map = (attn_map - _min) / (_max - _min)

        return attn_map


# ──────────────────────────── 便捷接口 ────────────────────────────
def generate_cam_heatmap(
    model: nn.Module,
    search_image: torch.Tensor,
    template_image: torch.Tensor,
    target_layer: str = 'input_proj',
    image_np: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    cmap: str = 'jet',
    alpha: float = 0.5,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """
    一站式生成 CAM 热力图

    Args:
        model: SOTTracker 模型
        search_image: 搜索区域图像 tensor [1, 3, H, W]
        template_image: 模板图像 tensor [1, 3, H, W]
        target_layer: 目标卷积层名称 ('input_proj' 或 'backbone')
        image_np: 可视化用的原始图像 (H, W, 3) RGB，用于叠加
        save_path: 保存路径
        cmap: 色彩映射名称
        alpha: 叠加透明度
        device: 设备

    Returns:
        dict: {'cam': cam热力图, 'overlayed': 叠加图, 'attn_map': 注意力图}
    """
    model.eval()
    model.to(device)
    search_image = search_image.to(device)
    template_image = template_image.to(device)

    results = {}

    # ---- Grad-CAM ----
    if target_layer == 'input_proj':
        layer = model.input_proj
    elif target_layer == 'backbone':
        # 获取 backbone 中最后阶段的卷积层
        body = model.backbone[0].body
        # 尝试获取最后一个卷积层
        try:
            layer = body.blocks4[-1].mixer
            if isinstance(layer, nn.Identity):
                layer = body.blocks4[-1].ffn.m.pw1
        except Exception:
            layer = model.input_proj
    else:
        layer = model.input_proj

    cam_hook = GradCAMHook(model, layer)
    cam_hook.register_hooks()

    # 前向传播
    output = model(search_image, template_image)

    # 反向传播
    pred_logits = output['pred_logits']
    score = pred_logits[0, :, 0].mean()
    model.zero_grad()
    score.backward(retain_graph=True)

    # 生成 CAM
    cam = cam_hook.generate()
    results['cam'] = cam

    cam_hook.remove_hooks()

    # ---- 叠加可视化 ----
    if image_np is not None:
        visualizer = CAMVisualizer(cmap=cmap, alpha=alpha)
        if save_path:
            visualizer.save_comparison(image_np, cam, save_path, title='Grad-CAM Heatmap')
        results['overlayed'] = visualizer.overlay(image_np, cam)

    return results


def generate_cross_attn_heatmap(
    model: nn.Module,
    search_image: torch.Tensor,
    template_image: torch.Tensor,
    image_np: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    head_idx: int = 0,
    template_pos: int = 32,  # 模板中心 token（8x8 网格中心）
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """
    提取 Decoder 交叉注意力的热力图

    Args:
        model: SOTTracker 模型
        search_image: 搜索区域图像 tensor [1, 3, H, W]
        template_image: 模板图像 tensor [1, 3, H, W]
        image_np: 可视化用的原始图像 (H, W, 3) RGB
        save_path: 保存路径
        head_idx: attention head 索引
        template_pos: 模板序列中关注的 token 位置
        device: 设备

    Returns:
        dict: {'attn_map': 注意力图, 'overlayed': 叠加图}
    """
    model.eval()
    model.to(device)
    search_image = search_image.to(device)
    template_image = template_image.to(device)

    extractor = CrossAttnExtractor(model)
    extractor.register_hooks()

    with torch.no_grad():
        model(search_image, template_image)

    attn_map = extractor.get_attention_map(head_idx=head_idx, template_pos=template_pos)
    extractor.remove_hooks()

    results = {'attn_map': attn_map}

    if image_np is not None and attn_map is not None:
        visualizer = CAMVisualizer(cmap='jet', alpha=0.5)
        if save_path:
            base, ext = os.path.splitext(save_path)
            attn_path = f"{base}_cross_attn{ext}"
            visualizer.save_comparison(
                image_np, attn_map, attn_path,
                title=f'Cross-Attention Map (Head {head_idx})'
            )
        results['overlayed'] = visualizer.overlay(image_np, attn_map)

    return results


if __name__ == '__main__':
    print("CAM 热力图模块已就绪。")
    print("使用方法：")
    print("  from ltr.utils.cam_visualize import generate_cam_heatmap, CAMVisualizer")
    print("  详见 tools/demo_cam.py")
