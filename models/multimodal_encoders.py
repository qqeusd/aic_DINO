"""
三模态轻量编码器模块

为 Infrared(热红外) 和 Depth(深度) 模态提供轻量 CNN 编码器，
将原始图像映射到 256 维特征空间，与 RGB Backbone 输出的特征维度对齐。

设计原则:
    - 参数量极小 (~180K)，适合 2000 组小样本微调
    - 输出通道固定为 256，与 DINO hidden_dim 一致
    - 下采样 4× (两个 stride=2 层)，与 ResNet C3/C4/C5 的空间尺度兼容
"""

import torch
import torch.nn as nn


class LightweightEncoder(nn.Module):
    """
    轻量 CNN 编码器，用于 IR 和 Depth 模态的特征提取。

    结构:
        Conv(3×3, in_c→64) → BN → ReLU
        Conv(3×3, 64→128, stride=2) → BN → ReLU   (H/2, W/2)
        Conv(3×3, 128→256, stride=2) → BN → ReLU  (H/4, W/4)

    输出: [B, 256, H/4, W/4]

    Attributes:
        conv (nn.Sequential): 三层卷积 + BN + ReLU 的主干网络。
                              末层无激活函数，由后续 Transformer 的 LayerNorm 统一归一化。
        in_channels (int): 输入通道数（IR=3, Depth=1）。
        out_channels (int): 输出通道数，固定为 256。

    Example:
        >>> # IR 编码器
        >>> ir_enc = LightweightEncoder(in_channels=3)
        >>> ir_input = torch.randn(2, 3, 800, 800)
        >>> ir_feat = ir_enc(ir_input)   # torch.Size([2, 256, 200, 200])
        >>>
        >>> # Depth 编码器
        >>> d_enc = LightweightEncoder(in_channels=1)
        >>> d_input = torch.randn(2, 1, 800, 800)
        >>> d_feat = d_enc(d_input)      # torch.Size([2, 256, 200, 200])
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): 输入通道数。
                               IR 模态使用 3（RGB 兼容），Depth 模态使用 1。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 256

        self.conv = nn.Sequential(
            # Layer 1: 保持分辨率
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: 下采样 2×
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 再下采样 2×
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming 初始化卷积权重，BN 使用默认初始化。"""
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入图像 tensor。
                              形状: [B, in_channels, H, W]，值域 [0, 1]。

        Returns:
            torch.Tensor: 编码后的特征图。
                          形状: [B, 256, H/4, W/4]。
        """
        return self.conv(x)
