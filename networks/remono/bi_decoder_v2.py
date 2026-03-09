from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入必要的层，如果环境中没有，使用下方的 Mock 类
try:
    from layers import *
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(tensor, std=0.02):
        nn.init.normal_(tensor, std=std)


    def upsample(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


    class Conv3x3(nn.Module):
        def __init__(self, in_channels, out_channels, use_refl=True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        def forward(self, x): return self.conv(x)


    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = Conv3x3(in_channels, out_channels)
            self.nonlin = nn.ELU(inplace=True)

        def forward(self, x): return self.nonlin(self.conv(x))


class BiDepthDecoder2(nn.Module):
    """
    改进版 BiFPN 解码器：带有多级上采样精细化模块 (Refinement Head)

    输出尺度说明 (假设输入图像为 H x W):
    - Scale 0: H x W       (全分辨率, 192x640) -> 通过 Refinement Head 生成
    - Scale 1: H/2 x W/2   (1/2分辨率, 96x320) -> 通过 Refinement Head 生成
    - Scale 2: H/4 x W/4   (1/4分辨率, 48x160) -> 直接来自 BiFPN 输出

    通道数设计:
    - BiFPN 内部维持 [48, 96, 192] 对应 1/4, 1/8, 1/16 尺度
    - 上采样头逐渐减少通道数以节省计算量 (48 -> 24 -> 16)
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc

        # BiFPN 核心层的通道数 (对应 Encoder 的 1/4, 1/8, 1/16)
        self.bifpn_channels = [48, 96, 192]

        self.convs = OrderedDict()

        # ==========================================================
        # Part 1: BiFPN Core (处理语义信息融合)
        # ==========================================================

        # 1. 投影层 (Project Layers)
        for i in range(3):
            self.convs[("project", i)] = ConvBlock(self.num_ch_enc[i], self.bifpn_channels[i])

        # 2. Top-Down Path
        # P2(192) -> P1(96)
        self.convs[("td", 1)] = ConvBlock(self.bifpn_channels[1] + self.bifpn_channels[2], self.bifpn_channels[1])
        # P1(96) -> P0(48)
        self.convs[("td", 0)] = ConvBlock(self.bifpn_channels[0] + self.bifpn_channels[1], self.bifpn_channels[0])

        # 3. Bottom-Up Path
        self.convs[("down", 0)] = nn.Conv2d(self.bifpn_channels[0], self.bifpn_channels[0], kernel_size=3, stride=2,
                                            padding=1)
        self.convs[("down", 1)] = nn.Conv2d(self.bifpn_channels[1], self.bifpn_channels[1], kernel_size=3, stride=2,
                                            padding=1)

        # Node 1 Fusion
        self.convs[("bu", 1)] = ConvBlock(self.bifpn_channels[1] * 2 + self.bifpn_channels[0], self.bifpn_channels[1])
        # Node 2 Fusion
        self.convs[("bu", 2)] = ConvBlock(self.bifpn_channels[2] + self.bifpn_channels[1], self.bifpn_channels[2])

        # ==========================================================
        # Part 2: Refinement Head (解决单穿上采样模糊的问题)
        # ==========================================================

        # 这里的 Scale 定义符合 Lite-Mono/Monodepth2 的习惯：
        # Scale 0 = 全分辨率, Scale 1 = 1/2, Scale 2 = 1/4

        # Refine Layer 1: 从 1/4 (BiFPN P0) -> 1/2
        # 输入 48通道, 输出 24通道 (增加非线性变换，学习边缘细节)
        self.convs[("refine", 1)] = ConvBlock(self.bifpn_channels[0], 24)

        # Refine Layer 0: 从 1/2 -> Full Res (1/1)
        # 输入 24通道, 输出 16通道 (进一步精细化)
        self.convs[("refine", 0)] = ConvBlock(24, 16)

        # ==========================================================
        # Part 3: Depth Heads (生成视差图)
        # ==========================================================

        for s in self.scales:
            # 根据尺度选择输入通道数
            if s == 0:  # Full Res
                in_ch = 16
            elif s == 1:  # 1/2 Res
                in_ch = 24
            elif s == 2:  # 1/4 Res (来自 BiFPN P0)
                in_ch = self.bifpn_channels[0]  # 48
            elif s == 3:  # 1/8 Res (来自 BiFPN P1)
                in_ch = self.bifpn_channels[1]  # 96
            else:
                in_ch = self.bifpn_channels[0]  # Fallback

            self.convs[("dispconv", s)] = Conv3x3(in_ch, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}

        # --- 1. BiFPN Process (处理 1/4, 1/8, 1/16 特征) ---
        p0 = self.convs[("project", 0)](input_features[0])  # 48ch (1/4)
        p1 = self.convs[("project", 1)](input_features[1])  # 96ch (1/8)
        p2 = self.convs[("project", 2)](input_features[2])  # 192ch (1/16)

        # Top-Down
        p2_up = F.interpolate(p2, size=p1.shape[2:], mode="bilinear", align_corners=False)
        td_1 = self.convs[("td", 1)](torch.cat([p1, p2_up], dim=1))

        td_1_up = F.interpolate(td_1, size=p0.shape[2:], mode="bilinear", align_corners=False)
        td_0 = self.convs[("td", 0)](torch.cat([p0, td_1_up], dim=1))

        # Bottom-Up
        out_bifpn_0 = td_0  # Scale 2 (1/4 Res) Output

        down_0 = self.convs[("down", 0)](out_bifpn_0)
        if down_0.shape[2:] != p1.shape[2:]: down_0 = F.interpolate(down_0, size=p1.shape[2:], mode="nearest")

        out_bifpn_1 = self.convs[("bu", 1)](torch.cat([p1, td_1, down_0], dim=1))  # Scale 3 (1/8 Res) Output

        down_1 = self.convs[("down", 1)](out_bifpn_1)
        if down_1.shape[2:] != p2.shape[2:]: down_1 = F.interpolate(down_1, size=p2.shape[2:], mode="nearest")

        out_bifpn_2 = self.convs[("bu", 2)](torch.cat([p2, down_1], dim=1))  # Scale 4 (1/16 Res)

        # --- 2. Refinement & Upsampling (生成 1/2 和 Full Res 特征) ---

        # Scale 2 (1/4 Res): 直接使用 BiFPN 最精细的输出 (48ch)
        feat_scale_2 = out_bifpn_0

        # Scale 1 (1/2 Res): 1/4 -> Upsample -> ConvBlock (48->24ch)
        # 使用 ConvBlock 进行特征平滑和细化，比单纯 插值效果好
        feat_scale_1 = self.convs[("refine", 1)](upsample(feat_scale_2))

        # Scale 0 (Full Res): 1/2 -> Upsample -> ConvBlock (24->16ch)
        feat_scale_0 = self.convs[("refine", 0)](upsample(feat_scale_1))

        # --- 3. Output Generation (多尺度输出) ---
        # 整理特征图字典
        feats = {
            0: feat_scale_0,  # Full Res
            1: feat_scale_1,  # 1/2 Res
            2: feat_scale_2,  # 1/4 Res
            3: out_bifpn_1  # 1/8 Res (可选)
        }

        for s in self.scales:
            if s in feats:
                # 生成视差图
                disp = self.sigmoid(self.convs[("dispconv", s)](feats[s]))

                # 如果需要强制输出尺寸严格对齐 (处理奇数padding问题)，可在这里添加 interpolate
                # 但通常 Refinement 过程中的 upsample 已经保证了尺寸
                self.outputs[("disp", s)] = disp

        return self.outputs


if __name__ == '__main__':
    # 测试代码：模拟 KITTI 尺寸输入 (3, 192, 640)
    # Encoder 输出通常是:
    # 1/4:  48ch,  48 x 160
    # 1/8:  80ch,  24 x 80
    # 1/16: 128ch, 12 x 40
    print("Testing BiFPN Decoder with Refinement Head...")

    input_h, input_w = 192, 640
    features = [
        torch.randn(1, 48, input_h // 4, input_w // 4),  # Scale 2 (Encoder out)
        torch.randn(1, 80, input_h // 8, input_w // 8),  # Scale 3
        torch.randn(1, 128, input_h // 16, input_w // 16)  # Scale 4
    ]

    # 我们要求输出 Scale 0, 1, 2 (对应 Full, 1/2, 1/4)
    decoder = BiDepthDecoder2(num_ch_enc=[48, 80, 128], scales=[0, 1, 2])

    outputs = decoder(features)

    for k, v in outputs.items():
        print(f"Scale {k[1]} Output: {v.shape}")

    # 预期输出:
    # Scale 0: (1, 1, 192, 640) -> Full Resolution
    # Scale 1: (1, 1, 96, 320)  -> 1/2
    # Scale 2: (1, 1, 48, 160)  -> 1/4