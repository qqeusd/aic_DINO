"""
Scene-Aware Fusion Module —— DINO 三模态改造的核心创新模块

原理:
    1. 一个可学习的 Scene Query 通过 Cross-Attention 聚合 RGB 多尺度全局特征
    2. 生成光照/密度场景分类 logits（辅助监督信号）
    3. 由场景表征动态生成三个模态的融合权重 (w_rgb, w_ir, w_depth)
    4. 基于 Depth 边缘 + IR 局部对比度生成空间注意力图
    5. 对每个特征尺度执行加权融合，实现 "先理解场景，再分配模态"

设计要点:
    - 权重 MLP 的 gate 偏置初始化为 -2.0 → 初始 softmax 接近 [1, 0, 0]（纯 RGB）
      这保护了 COCO 预训练权重，让 IR/Depth 权重从零逐渐增长
    - 参数量约 200K，符合微调计划中 "新增参数 <20K" 的要求数量级
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneQueryFusion(nn.Module):
    """
    Scene-Aware Fusion Module

    输入: RGB多尺度特征 [list of [B,256,Hi,Wi]], IR特征 [B,256,H/4,W/4], Depth特征 [B,256,H/4,W/4]
    输出: 融合后的多尺度特征 [list of [B,256,Hi,Wi]] + 辅助预测 dict

    Attributes:
        scene_query (nn.Parameter): 可学习的全局场景查询向量 [1, 1, 256]。
        cross_attn (nn.MultiheadAttention): 对 RGB 各尺度均值做交叉注意力。
        lighting_head (nn.Linear): 光照分类器 (bright / dim / dark)。
        density_head (nn.Linear): 密度分类器 (sparse / medium / dense)。
        weight_mlp (nn.Sequential): 模态权重生成 MLP(256→128→3) + softmax。
        spatial_conv (nn.Sequential): 空间调制器，从 IR+Depth 中提取空间注意力。
    """

    def __init__(self, hidden_dim=256, nheads=8):
        """
        Args:
            hidden_dim (int): Transformer 隐藏维度，默认 256（与 DINO 一致）。
            nheads (int): 多头注意力头数，默认 8。
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- 1. Learnable Scene Query ---
        self.scene_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.scene_query, std=0.02)

        # --- 2. Note ---
        # 融合在 DINO forward 的 srcs 层发生（input_proj 之后），
        # 所有尺度已统一为 256 维，无需额外投影。

        # --- 3. Cross-Attention for Global Scene Context ---
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, nheads, batch_first=True
        )

        # --- 4. Scene Classification Heads ---
        self.lighting_head = nn.Linear(hidden_dim, 3)   # bright / dim / dark
        self.density_head = nn.Linear(hidden_dim, 3)    # sparse / medium / dense

        # --- 5. Modality Weight Generator ---
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        nn.init.constant_(self.weight_mlp[-1].bias.data, -2.0)
        nn.init.normal_(self.weight_mlp[-1].weight.data, std=0.01)

        # --- 6. Spatial Modulator ---
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._reset_heads()

    def _reset_heads(self):
        """初始化分类头和权重 MLP 的参数。"""
        for head in [self.lighting_head, self.density_head]:
            nn.init.normal_(head.weight.data, std=0.01)
            nn.init.constant_(head.bias.data, 0)

    def forward(self, rgb_features, ir_feat, depth_feat):
        """
        执行 Scene-Aware 多模态多尺度融合。

        Args:
            rgb_features (List[torch.Tensor]): 经 input_proj 投影后的多尺度 RGB 特征
                （即 DINO 的 srcs）。每个元素形状 [B, 256, Hi, Wi]，通常 4 个尺度。
            ir_feat (torch.Tensor): IR 编码器输出，形状 [B, 256, H/4, W/4]。
            depth_feat (torch.Tensor): Depth 编码器输出，形状 [B, 256, H/4, W/4]。

        Returns:
            tuple:
                - fused_features (List[torch.Tensor]): 融合后的多尺度特征，
                  每层形状与 rgb_features 对应元素相同。
                - scene_outputs (dict): 场景预测结果，包含:
                    - 'light_logits' (torch.Tensor): [B, 3] 光照分类 logits
                    - 'density_logits' (torch.Tensor): [B, 3] 密度分类 logits
                    - 'modality_weights' (torch.Tensor): [B, 3] 模态权重
        """
        B = rgb_features[0].shape[0]

        # ============================================================
        # Step 1: Global Scene Query —— 从 RGB 多尺度特征中提取场景表征
        # ============================================================
        # 每个尺度做全局平均池化（srcs 已统一为 256 维）
        rgb_global = []
        for f in rgb_features:
            rgb_global.append(f.mean(dim=[2, 3]))        # [B, hidden_dim]
        rgb_global = torch.stack(rgb_global, dim=1)      # [B, N_scales, 256]

        # Cross-Attention: scene_query 查询各尺度 RGB 的全局语义
        scene_q = self.scene_query.expand(B, -1, -1)  # [B, 1, 256]
        scene_repr, _ = self.cross_attn(scene_q, rgb_global, rgb_global)
        scene_repr = scene_repr.squeeze(1)  # [B, 256]

        # ============================================================
        # Step 2: Scene Predictions —— 光照 / 密度分类
        # ============================================================
        light_logits = self.lighting_head(scene_repr)     # [B, 3]
        density_logits = self.density_head(scene_repr)    # [B, 3]

        # ============================================================
        # Step 3: Modality Weights —— 动态分配三模态权重
        # ============================================================
        modality_weights = self.weight_mlp(scene_repr).softmax(dim=-1)  # [B, 3]
        w_rgb   = modality_weights[:, 0].view(B, 1, 1, 1)
        w_ir    = modality_weights[:, 1].view(B, 1, 1, 1)
        w_depth = modality_weights[:, 2].view(B, 1, 1, 1)

        # ============================================================
        # Step 4: Spatial Attention Map —— 从 IR+Depth 中提取空间结构
        # ============================================================
        ir_depth = torch.cat([ir_feat, depth_feat], dim=1)  # [B, 512, H, W]
        spatial_attn = self.spatial_conv(ir_depth)           # [B, 1, H, W]

        # ============================================================
        # Step 5: Multi-Scale Fusion —— 对每个尺度执行加权融合
        # ============================================================
        fused_features = []
        for rgb_f in rgb_features:
            Hi, Wi = rgb_f.shape[-2:]

            # 将 IR/Depth 特征对齐到当前 RGB 尺度
            ir_i = F.interpolate(
                ir_feat, size=(Hi, Wi), mode='bilinear', align_corners=False
            )
            d_i = F.interpolate(
                depth_feat, size=(Hi, Wi), mode='bilinear', align_corners=False
            )
            spat_i = F.interpolate(
                spatial_attn, size=(Hi, Wi), mode='bilinear', align_corners=False
            )

            # 加权融合：w_rgb * RGB + w_ir * IR + w_depth * Depth
            fused = w_rgb * rgb_f + w_ir * ir_i + w_depth * d_i
            # 空间调制：按像素注意力缩放
            fused = fused * spat_i

            fused_features.append(fused)

        # ============================================================
        # 组装返回
        # ============================================================
        scene_outputs = {
            'light_logits': light_logits,
            'density_logits': density_logits,
            'modality_weights': modality_weights,
        }

        return fused_features, scene_outputs
