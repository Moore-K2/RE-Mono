from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Lite-Mono 的 layers，如果是在独立环境测试，则使用下面的 Mock 类
try:
    from layers import *
    from timm.models.layers import trunc_normal_
except ImportError:
    # --- Mock layers for standalone testing ---
    print("Warning: 'layers' module not found. Using mock layers for demonstration.")


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


class BiDepthDecoder(nn.Module):
    """
    基于 BiFPN 结构的深度估计解码器
    添加了refine层，用于将1/16, 1/8, 1/4的特征图上采样到1/4，1/2, 1/1,

    功能：
    1. 接收 Encoder 的多尺度特征
    2. 使用 BiFPN (双向特征金字塔) 进行特征融合 (Top-Down + Bottom-Up)
    3. 输出指定通道数的特征 [48, 96, 192]
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc  #num_ch_enc: [48, 80, 128]

        # --- 核心修改：定义 BiFPN 的目标通道数 ---
        # 对应 Scale 0 (浅层/大图) -> 48
        # 对应 Scale 1 (中层)      -> 96
        # 对应 Scale 2 (深层/小图) -> 192
        self.bifpn_channels = [48, 96, 192]

        self.convs = OrderedDict()

        # --- 1. 投影层 (Project Layers) ---
        # 将编码器输出的任意通道数映射到 BiFPN 需要的固定通道 [48, 96, 192]
        for i in range(3):
            # 使用 ConvBlock 进行通道转换
            self.convs[("project", i)] = ConvBlock(self.num_ch_enc[i], self.bifpn_channels[i])

        # --- 2. Top-Down Path (自顶向下路径: 传递语义信息) ---

        # Level 2 -> Level 1
        # 输入: Level 1 (96) + Upsample(Level 2 (192)) = 288
        self.convs[("td", 1)] = ConvBlock(self.bifpn_channels[1] + self.bifpn_channels[2],
                                          self.bifpn_channels[1])

        # Level 1 -> Level 0
        # 输入: Level 0 (48) + Upsample(Level 1_td (96)) = 144
        self.convs[("td", 0)] = ConvBlock(self.bifpn_channels[0] + self.bifpn_channels[1],
                                          self.bifpn_channels[0])

        # --- 3. Bottom-Up Path (自底向上路径: 传递细节信息并融合) ---

        # 下采样层 (使用 stride=2 的卷积代替池化)
        self.convs[("down", 0)] = nn.Conv2d(self.bifpn_channels[0], self.bifpn_channels[0], kernel_size=3, stride=2,
                                            padding=1)
        self.convs[("down", 1)] = nn.Conv2d(self.bifpn_channels[1], self.bifpn_channels[1], kernel_size=3, stride=2,
                                            padding=1)

        # Node 1 (Level 1): 融合 P1_original + P1_td + Downsample(P0_out)
        # 输入: 96 + 96 + 48 = 240
        self.convs[("bu", 1)] = ConvBlock(self.bifpn_channels[1] * 2 + self.bifpn_channels[0],
                                          self.bifpn_channels[1])

        # Node 2 (Level 2): 融合 P2_original + Downsample(P1_out)
        # 输入: 192 + 96 = 288
        self.convs[("bu", 2)] = ConvBlock(self.bifpn_channels[2] + self.bifpn_channels[1],
                                          self.bifpn_channels[2])

        # Refine Layer 0: 从 1/2 -> Full Res (1/1)
        # 输入 24通道, 输出 16通道 (进一步精细化)
        self.convs[("refine", 0)] = ConvBlock(48, 24)
        self.convs[("refine", 1)] = ConvBlock(96, 48)
        self.convs[("refine", 2)] = ConvBlock(192, 96)

        # --- 4. 视差生成层 (DispConv) ---
        # 根据当前层级的通道数生成单通道视差图
        for s in self.scales:
            if s in [0, 1, 2]:
                self.convs[("dispconv", s)] = Conv3x3(self.bifpn_channels[s]//2, self.num_output_channels)

        # 注册所有层
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """
        Args:
            input_features: [feat0, feat1, feat2] (Scale 0, 1, 2)
        """
        self.outputs = {}

        # 1. Projection (对齐通道)
        p0 = self.convs[("project", 0)](input_features[0])  # -> 48
        p1 = self.convs[("project", 1)](input_features[1])  # -> 96
        p2 = self.convs[("project", 2)](input_features[2])  # -> 192

        # 2. Top-Down Path
        # P2 -> P1
        p2_up = upsample(p2)
        # 尺寸对齐 (防止奇数尺寸导致的 shape 不匹配)
        if p2_up.shape[2:] != p1.shape[2:]:
            p2_up = F.interpolate(p2_up, size=p1.shape[2:], mode="bilinear", align_corners=False)
        td_1 = self.convs[("td", 1)](torch.cat([p1, p2_up], dim=1))

        # P1 -> P0
        td_1_up = upsample(td_1)
        if td_1_up.shape[2:] != p0.shape[2:]:
            td_1_up = F.interpolate(td_1_up, size=p0.shape[2:], mode="bilinear", align_corners=False)
        td_0 = self.convs[("td", 0)](torch.cat([p0, td_1_up], dim=1))

        # 3. Bottom-Up Path
        # Node 0 Output (Scale 0)
        out_0 = td_0  # Scale 2 (1/4 Res) Output---48

        # Node 0 -> Node 1 (Scale 1)
        down_0 = self.convs[("down", 0)](out_0)
        if down_0.shape[2:] != p1.shape[2:]:
            down_0 = F.interpolate(down_0, size=p1.shape[2:], mode="nearest")

        # 融合: Original + Top-Down + Bottom-Up
        out_1 = self.convs[("bu", 1)](torch.cat([p1, td_1, down_0], dim=1))  # Scale 3 (1/8 Res) Output---96

        # Node 1 -> Node 2 (Scale 2)
        down_1 = self.convs[("down", 1)](out_1)
        if down_1.shape[2:] != p2.shape[2:]:
            down_1 = F.interpolate(down_1, size=p2.shape[2:], mode="nearest")

        # 融合: Original + Bottom-Up
        out_2 = self.convs[("bu", 2)](torch.cat([p2, down_1], dim=1))  # Scale 4 (1/16 Res) Output---192

        #TODO 上采样

        out_0_refine = upsample(self.convs[("refine", 0)](upsample(out_0))) # Scale 0 (1/1 Res) Output---24
        out_1_refine = upsample(self.convs[("refine", 1)](upsample(out_1))) # Scale 1 (1/2 Res) Output---48
        out_2_refine = upsample(self.convs[("refine", 2)](upsample(out_2))) # Scale 2 (1/4 Res) Output---96


        # 整理 BiFPN 输出特征
        # Scale 0: 48, Scale 1: 96, Scale 2: 192
        # bifpn_feats = {0: out_0, 1: out_1, 2: out_2}
        bifpn_feats = {0: out_0_refine, 1: out_1_refine, 2: out_2_refine}

        # 4. 生成视差图
        # 按照 Lite-Mono 惯例，这里我们输出不同尺度的视差图
        # 如果需要将所有视差图上采样到原图尺寸，可以在这里操作，也可以在 loss 计算时操作
        for s in self.scales:
            if s in bifpn_feats:
                # 生成视差图特征 (Scale s)
                disp_logit = self.convs[("dispconv", s)](bifpn_feats[s])

                # 双线性插值上采样 (模拟原版 decoder 的行为)
                # 原版 decoder 逐级上采样，最终每一层都接近原图或上一级
                # 这里我们假设目标是恢复到全分辨率 (通常输入 feat0 是 1/4 分辨率)
                # 为了保持通用性，这里进行简单的 2x 上采样，或者根据需求调整
                # ----------------------------------------------------------------
                # 修正：Lite-Mono 原版是输出多尺度的 disp 字典，
                # 并在计算 Loss 时将 GT 降采样到对应尺度，或者将 disp 上采样到 GT。
                # 这里的 disp 直接对应当前 bifpn_feats 的分辨率。
                # ----------------------------------------------------------------
                disp = self.sigmoid(disp_logit)
                #
                # # 为了让输出可视化或计算方便，通常在这里做一个简单的上采样(可选)
                # self.outputs[("disp", s)] = upsample(disp) # 或者是保持原分辨率
                self.outputs[("disp", s)] = disp

                # # 这里演示上采样到 input_features[0] 的 4倍 (假设 input[0] 是 1/4 下采样)
                # target_size = [input_features[0].shape[2] * 4, input_features[0].shape[3] * 4]
                #
                # disp_up = F.interpolate(disp_logit, size=target_size, mode='bilinear', align_corners=False)
                # self.outputs[("disp", s)] = self.sigmoid(disp_up)

        return self.outputs


if __name__ == '__main__':
    # --- 测试代码 ---
    print("Initializing BiFPN Depth Decoder...")

    # 模拟输入：Batch=1
    # Scale 0: 48ch, 80x256 (假设原图 320x1024)
    # Scale 1: 80ch, 40x128
    # Scale 2: 128ch, 20x64
    features = [torch.randn(1, 48, 48, 160),
                torch.randn(1, 80, 24, 80),
                torch.randn(1, 128, 12, 40)]

    # 创建解码器，指定输入通道与 Encoder 匹配
    decoder = BiDepthDecoder(num_ch_enc=[48, 80, 128], scales=[0, 1, 2])

    # 前向传播
    outputs = decoder(features)

    print("\nCheck Outputs:")
    for k, v in outputs.items():
        print(f"Key {k}: Shape {v.shape}")

    print("\nStructure Validation:")
    # 验证 BiFPN 节点输出通道数是否符合 [48, 96, 192]
    # 我们无法直接 print channel 因为 layers 实现可能不同，
    # 但我们可以通过前向传播时的中间变量推断，或者打印网络层结构
    print("Decoder structure initialized. Forward pass successful.")

    # 打印其中一个层的结构以确认
    print(f"Project Layer 0: {decoder.convs[('project', 0)]}")
    # 注意：此处不再尝试访问 .conv[0].out_channels，避免报错
