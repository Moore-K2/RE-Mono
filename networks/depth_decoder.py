# 导入Python2/3兼容的特性（绝对导入、除法、打印函数），保证代码跨版本兼容
from __future__ import absolute_import, division, print_function
# 导入有序字典，用于按顺序存储解码器的卷积层（保证前向传播时层的执行顺序）
from collections import OrderedDict

import numpy as np

# 导入自定义层（如ConvBlock/Conv3x3/upsample，通常包含卷积+激活+归一化、上采样等基础组件）
from layers import *
# 从timm库导入截断正态分布初始化函数，用于权重初始化（比普通正态分布更稳定）
from timm.models.layers import trunc_normal_


class DepthDecoder(nn.Module):
    """
    深度估计模型的解码器（Decoder）
    功能：接收编码器（Encoder）输出的多尺度特征，通过上采样、跳跃连接（Skip Connection）和卷积，
          解码出不同尺度的视差/深度图（Disparity/Depth Map）
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        """
        解码器初始化
        Args:
            num_ch_enc: 编码器各阶段输出的通道数，如[48, 80, 128]（对应编码器3个尺度的特征通道）
            scales: 需要输出的深度图尺度索引，如range(4)表示输出0/1/2/3尺度（实际常用0/1/2）
            num_output_channels: 输出通道数，深度图为单通道（1），若做多任务可调整
            use_skips: 是否使用跳跃连接（融合编码器浅层特征，提升细节）
        """
        super().__init__()  # 调用nn.Module父类的构造函数

        # 保存输出通道数（深度图为1通道）
        self.num_output_channels = num_output_channels
        # 保存是否使用跳跃连接的标记
        self.use_skips = use_skips
        # 上采样模式：双线性插值（bilinear），比最近邻插值更平滑，适合深度图生成
        self.upsample_mode = 'bilinear'
        # 保存需要输出的深度图尺度
        self.scales = scales

        # 保存编码器的通道数（用于计算解码器通道数）
        self.num_ch_enc = num_ch_enc
        # 解码器每个阶段的通道数 = 编码器对应阶段通道数的一半（整数型），是深度估计的经典通道设计
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # 用OrderedDict存储解码器的所有卷积层（保证按添加顺序执行，避免字典无序导致错误）
        self.convs = OrderedDict()
        # 反向遍历编码器尺度（i=2→1→0），从最深的特征开始解码（编码器输出3个尺度：0浅/1中/2深）
        for i in range(2, -1, -1):
            # ---------------- 构建上采样卷积层0（upconv_0） ----------------
            # 输入通道数：若i=2（最深层），输入是编码器最后一个特征的通道数；否则是解码器下一级的输出通道数
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            # 输出通道数：解码器当前阶段的通道数（编码器对应通道的一半）
            num_ch_out = self.num_ch_dec[i]
            # 添加upconv_0层：ConvBlock通常包含Conv2d + BN/IN + 激活函数（如ReLU/GELU），用于上采样后的特征融合
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)  # 调试用：打印当前层的输入/输出通道数

            # ---------------- 构建上采样卷积层1（upconv_1） ----------------
            # 输入通道数：先初始化为upconv_0的输出通道数
            num_ch_in = self.num_ch_dec[i]
            # 若使用跳跃连接且不是最浅层（i>0），则拼接编码器对应浅层的特征，输入通道数需增加
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # 输出通道数：保持解码器当前阶段的通道数（融合跳跃连接后重新压缩）
            num_ch_out = self.num_ch_dec[i]
            # 添加upconv_1层：融合跳跃连接后的特征卷积
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # ---------------- 构建视差生成卷积层（dispconv） ----------------
        # 为每个指定尺度构建3x3卷积层，将解码器特征映射为单通道视差图
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # 将OrderedDict中的卷积层转换为ModuleList（nn.Module需要用ModuleList管理子模块，才能被优化器更新）
        self.decoder = nn.ModuleList(list(self.convs.values()))
        # Sigmoid激活函数：将视差图的值归一化到[0, 1]区间（深度/视差的经典归一化方式）
        self.sigmoid = nn.Sigmoid()

        # 初始化所有层的权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        自定义权重初始化函数（会被self.apply()遍历所有子模块调用）
        Args:
            m: 单个子模块（如Conv2d/Linear/ConvBlock等）
        """
        # 若模块是卷积层或线性层，用截断正态分布初始化权重（std=0.02，避免权重过大）
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # 若有偏置项，将偏置初始化为0（稳定训练初始阶段）
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """
        解码器前向传播：从编码器特征解码出多尺度深度/视差图
        Args:
            input_features: 编码器输出的多尺度特征列表，如[feat0, feat1, feat2]（feat0浅/feat2深）
        Returns:
            self.outputs: 字典，key为("disp", 尺度)，value为对应尺度的视差图
        """
        # 初始化输出字典，存储不同尺度的视差图
        self.outputs = {}
        # 从编码器最深的特征开始解码（input_features[-1] = feat2）
        x = input_features[-1]

        # 反向遍历解码器阶段（i=2→1→0），逐层上采样+融合特征
        for i in range(2, -1, -1):
            # 第一步：通过upconv_0卷积层，压缩/融合特征（上采样前的卷积，减少通道数）
            x = self.convs[("upconv", i, 0)](x)
            # 第二步：上采样（默认bilinear），将特征图尺寸放大2倍（恢复到上一级尺度），转为列表便于拼接
            x = [upsample(x)]

            # 若使用跳跃连接且不是最浅层（i>0），拼接编码器对应尺度的浅层特征（补充细节信息）
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]  # 列表拼接：[上采样特征, 编码器浅层特征]
            # 将列表中的特征在通道维度（dim=1）拼接（如通道数从80→80+48=128）
            x = torch.cat(x, 1)
            # 第三步：通过upconv_1卷积层，融合拼接后的特征（压缩通道数回原尺寸）
            x = self.convs[("upconv", i, 1)](x)

            # 若当前i在指定输出尺度中，生成对应尺度的视差图
            if i in self.scales:
                # 1. 通过dispconv卷积层将特征映射为单通道视差特征
                # 2. 上采样（bilinear）到目标分辨率（通常是输入图像的尺度）
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                # 3. Sigmoid激活，将视差值归一化到[0, 1]，存入输出字典
                self.outputs[("disp", i)] = self.sigmoid(f)

        # 返回多尺度视差图字典（后续可通过视差→深度的公式转换为深度图）
        return self.outputs


if __name__ == '__main__':
    features = [torch.randn(1, 48, 48, 160),
                torch.randn(1, 80, 24, 80),
                torch.randn(1, 128, 12, 40)]
    decoder = DepthDecoder(num_ch_enc=np.array([48, 80, 128]), scales=range(3))
    disp_map = decoder(features)
