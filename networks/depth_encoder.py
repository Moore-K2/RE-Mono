import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习
from torch import nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口，包含各种激活函数、池化等操作
from timm.models.layers import DropPath  # 从timm库导入DropPath层，用于随机深度正则化
import math  # 导入math库，用于数学运算
import torch.cuda  # 导入PyTorch的CUDA模块，用于GPU加速
import onnx
import netron

class PositionalEncodingFourier(nn.Module):
    """
    位置编码类，基于"Attention is all you need"论文中的傅里叶核实现
    实现参考了DeTR代码：https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()  # 调用父类构造函数
        # 1x1卷积，将2*hidden_dim维度映射到dim维度
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi  # 缩放因子，用于位置编码的周期
        self.temperature = temperature  # 温度参数，用于位置编码的频率缩放
        self.hidden_dim = hidden_dim  # 隐藏维度
        self.dim = dim  # 输出维度

    def forward(self, B, H, W):
        # 创建掩码（全False），用于计算位置累积（实际未使用掩码遮挡，仅用于获取设备）
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask  # 取反掩码（全True）
        # 计算y方向的位置累积（模拟行索引）
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 计算x方向的位置累积（模拟列索引）
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6  # 防止除零的小常数
        # 归一化y方向位置到[0, 2π]
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        # 归一化x方向位置到[0, 2π]
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成频率维度的缩放因子
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature **(2 * (dim_t // 2) / self.hidden_dim)

        # 计算x方向的位置编码（sin和cos交替）
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),  # 偶数索引用sin
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 奇数索引用cos，展平维度
        # 计算y方向的位置编码（sin和cos交替）
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接y和x方向的位置编码，调整维度为(B, C, H, W)并投影到目标维度
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """
    交叉协方差注意力（XCA）操作：通过交叉协方差矩阵（Q^T K ∈ d_h × d_h）的softmax归一化权重更新通道
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()  # 调用父类构造函数
        self.num_heads = num_heads  # 注意力头数
        # 温度参数，用于缩放注意力权重（可学习）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 线性层，同时生成Q、K、V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出dropout

    def forward(self, x):
        B, N, C = x.shape  # B:批次大小, N:序列长度, C:通道数
        # 生成QKV并拆分：(B, N, 3*C) → (B, N, 3, num_heads, C//num_heads) → 调整维度顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q、K、V

        # 调整Q、K、V的维度（将通道维度提前，便于矩阵乘法）
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # 归一化Q和K（沿最后一维）
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力权重：(Q @ K^T) * 温度参数 → softmax → dropout
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权V，调整维度并投影输出
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore  # 忽略TorchScript编译
    def no_weight_decay(self):
        # 指定不需要权重衰减的参数（温度参数）
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()  # 调用父类构造函数
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 偏移参数
        self.eps = eps  # 防止除零的小常数
        self.data_format = data_format  # 数据格式：channels_last或channels_first
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 不支持的格式则报错
        self.normalized_shape = (normalized_shape,)  # 归一化的形状

    def forward(self, x):
        if self.data_format == "channels_last":
            # 通道最后格式（如(N, H, W, C)），直接调用F.layer_norm
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 通道优先格式（如(N, C, H, W)），手动计算层归一化
            u = x.mean(1, keepdim=True)  # 沿通道维度求均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 沿通道维度求方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
            # 应用缩放和偏移（权重形状适配通道优先）
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()  # 调用父类构造函数
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)  # 批归一化层
        self.act = nn.GELU()  # GELU激活函数

    def forward(self, x):
        output = self.bn(x)  # 批归一化
        output = self.act(output)  # 激活
        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()  # 调用父类构造函数
        self.bn_act = bn_act  # 是否使用批归一化和激活函数

        # 卷积层
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            # 如果需要，添加BN+GELU
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)  # 卷积操作

        if self.bn_act:
            output = self.bn_gelu(output)  # 应用BN和激活

        return output


class CDilated(nn.Module):
    """定义膨胀卷积层"""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: 输入通道数
        :param nOut: 输出通道数
        :param kSize: 卷积核大小
        :param stride: 步长（下采样率）
        :param d: 膨胀率
        """
        super().__init__()  # 调用父类构造函数
        # 根据膨胀率计算padding，保证输出尺寸与输入一致（当stride=1时）
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)  # 膨胀卷积

    def forward(self, input):
        """
        :param input: 输入特征图
        :return: 转换后的特征图
        """
        output = self.conv(input)  # 膨胀卷积操作
        return output


class DilatedConv(nn.Module):
    """连续膨胀卷积（CDC）模块中的单个膨胀卷积层"""
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: 输入维度
        :param k: 卷积核大小
        :param dilation: 膨胀率
        :param drop_path: 随机深度概率
        :param layer_scale_init_value: 层缩放初始值
        :param expan_ratio: 倒置瓶颈的扩展比例
        """
        super().__init__()  # 调用父类构造函数

        # 深度可分离膨胀卷积（groups=dim实现深度卷积）
        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)  # 批归一化

        self.norm = LayerNorm(dim, eps=1e-6)  # 层归一化
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # 1x1卷积（用线性层实现，处理通道）
        self.act = nn.GELU()  # GELU激活
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)  # 1x1卷积（还原通道）
        # 层缩放参数（可选）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # 随机深度层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # 残差连接的输入

        x = self.ddwconv(x)  # 深度可分离膨胀卷积
        x = self.bn1(x)  # 批归一化

        # 维度转换：(N, C, H, W) → (N, H, W, C)，适配线性层
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)  # 通道扩展
        x = self.act(x)  # 激活
        x = self.pwconv2(x)  # 通道还原
        if self.gamma is not None:
            x = self.gamma * x  # 层缩放
        # 维度转换回：(N, H, W, C) → (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 残差连接 + 随机深度
        x = input + self.drop_path(x)
        return x


class LGFI(nn.Module):
    """局部-全局特征交互模块（Local-Global Features Interaction）"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()  # 调用父类构造函数

        self.dim = dim  # 输入维度
        self.pos_embd = None  # 位置编码（可选）
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)  # 初始化位置编码

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)  # XCA注意力前的层归一化

        # XCA的层缩放参数（可选）
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        # 初始化XCA注意力模块
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)  # 倒置瓶颈前的层归一化
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)  # 通道扩展
        self.act = nn.GELU()  # 激活
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)  # 通道还原
        # 瓶颈层的层缩放参数（可选）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # 随机深度层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x  # 残差连接的输入

        # XCA注意力部分
        B, C, H, W = x.shape  # B:批次, C:通道, H:高度, W:宽度
        # 维度转换：(B, C, H, W) → (B, H*W, C)（展平空间维度为序列）
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            # 生成位置编码并添加到特征中
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        # 应用XCA注意力（带层归一化和缩放）
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        # 维度转换回：(B, H*W, C) → (B, H, W, C)
        x = x.reshape(B, H, W, C)

        # 倒置瓶颈部分
        x = self.norm(x)  # 层归一化
        x = self.pwconv1(x)  # 通道扩展
        x = self.act(x)  # 激活
        x = self.pwconv2(x)  # 通道还原
        if self.gamma is not None:
            x = self.gamma * x  # 层缩放
        # 维度转换回：(B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 残差连接 + 随机深度
        x = input_ + self.drop_path(x)
        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()  # 调用父类构造函数
        self.pool = nn.ModuleList()  # 存储多个平均池化层
        # 添加ratio个3x3平均池化层（步长2，padding1），总下采样率为2^ratio
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        # 依次应用每个池化层，实现多步下采样
        for pool in self.pool:
            x = pool(x)
        return x


class LiteMono(nn.Module):
    """Lite-Mono模型：用于深度估计的编码器"""
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False],** kwargs):

        super().__init__()  # 调用父类构造函数

        # 根据模型类型配置参数（通道数、深度、膨胀率等）
        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])  # 编码器各阶段输出通道数
            self.depth = [4, 4, 10]  # 各阶段的块数量
            self.dims = [48, 80, 128]  # 各阶段的特征维度
            # 根据输入尺寸配置膨胀率
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        # 验证全局块类型是否合法
        for g in global_block_type:
            assert g in ['None', 'LGFI']

        # 下采样层（包含stem和中间下采样卷积）
        self.downsample_layers = nn.ModuleList()
        # stem1：初始卷积序列（3层卷积，用于特征提取）
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        # stem2：结合stem1输出和下采样输入的卷积
        self.stem2 = nn.Sequential(
            Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)  # 添加stem1到下采样层

        # 输入下采样模块（生成不同尺度的输入特征）
        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))  # 下采样率为2^i

        # 添加后续下采样层（用于各阶段之间的降采样）
        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        # 模型阶段（每个阶段包含多个块）
        self.stages = nn.ModuleList()
        # 生成随机深度概率（线性递增）
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0  # 记录当前块索引
        for i in range(3):
            stage_blocks = []  # 存储当前阶段的块
            for j in range(self.depth[i]):
                # 阶段末尾的块使用全局交互模块（如LGFI）
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))
                    else:
                        raise NotImplementedError  # 不支持的全局块类型
                else:
                    # 其他块使用膨胀卷积块
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))
            # 添加当前阶段的块序列
            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]  # 更新块索引

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 卷积层和线性层使用kaiming正态分布初始化
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 层归一化参数初始化（偏置0，权重1）
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # 批归一化参数初始化（权重1，偏置0）
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []  # 存储各阶段输出特征
        x = (x - 0.45) / 0.225  # 输入标准化（均值0.45，标准差0.225）

        # 生成不同尺度的下采样输入
        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        # 第一阶段stem处理
        x = self.downsample_layers[0](x)  # stem1处理
        # 结合stem1输出和x_down[0]（下采样1次的输入），通过stem2
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        # 处理第一阶段的块
        for s in range(len(self.stages[0])-1): # 前三个模块CDC blocks
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x) # LFGI block
        tmp_x.append(x)
        features.append(x)  # 保存第一阶段输出

        # 处理第二、三阶段
        for i in range(1, 3):
            tmp_x.append(x_down[i])  # 添加对应尺度的下采样输入
            x = torch.cat(tmp_x, dim=1)  # 拼接特征
            x = self.downsample_layers[i](x)  # 下采样

            tmp_x = [x]
            # 处理当前阶段的块
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            tmp_x.append(x)

            features.append(x)  # 保存当前阶段输出

        return features

    def forward(self, x):
        # 前向传播：提取特征并返回
        x = self.forward_features(x)
        return x


if __name__ == '__main__':
    lite_mono = LiteMono()
    print(lite_mono)
    x = torch.randn(1, 3, 192, 640)
    l2 = [1 if i > 3 else i for i in range(5)]
    print(l2)
    # features = lite_mono(x)
    # for feature in features:
    #     print(feature.shape)

#     torch.onnx.export(lite_mono, x, "lite_mono.onnx")
#
#     netron.start('lite_mono.onnx')
# if __name__ == '__main__':
#     # 1. 初始化模型
#     # 注意：LiteMono 默认输入是 (192, 640)，如果要支持其他尺寸，确保参数配置正确
#     model = LiteMono(height=192, width=640)
#
#     # 2. 切换到评估模式 (非常重要！关闭 Dropout，固定 BatchNorm)
#     model.eval()
#
#     # 3. 创建虚拟输入 (Dummy Input)
#     # 形状必须符合模型的预期输入：[Batch, Channel, Height, Width]
#     dummy_input = torch.randn(1, 3, 192, 640)
#
#     # 4. 定义导出路径
#     onnx_path = "litemono.onnx"
#
#     print(f"正在导出模型到 {onnx_path} ...")
#
#     # 5. 执行导出
#     torch.onnx.export(
#         model,  # 待导出的模型
#         dummy_input,  # 虚拟输入
#         onnx_path,  # 输出路径
#         export_params=True,  # 导出训练好的参数权重
#         opset_version=11,  # 【关键】Transformer 类模型建议使用 11 或 13
#         do_constant_folding=True,  # 优化常量折叠
#         input_names=['input_image'],  # 输入节点的名称，在 Netron 中显示
#         output_names=['feat_stage1', 'feat_stage2', 'feat_stage3'],  # 输出节点的名称
#         dynamic_axes={  # 动态维度设置
#             'input_image': {0: 'batch_size'},  # 允许 batch_size 变化
#             'feat_stage1': {0: 'batch_size'},
#             'feat_stage2': {0: 'batch_size'},
#             'feat_stage3': {0: 'batch_size'}
#         }
#     )
#     print("导出成功！")
#
#     # 6. 验证 ONNX 模型 (可选，检查模型结构是否有误)
#     try:
#         onnx_model = onnx.load(onnx_path)
#         onnx.checker.check_model(onnx_model)
#         print("ONNX 模型文件验证通过。")
#     except Exception as e:
#         print(f"ONNX 模型验证失败: {e}")
#
#     # 7. 使用 Netron 可视化
#     # 如果你在本地运行（有图形界面），这会自动打开浏览器
#     # 如果你在服务器/Colab 运行，请下载 .onnx 文件到本地，访问 https://netron.app 打开
#     try:
#         print("正在启动 Netron...")
#         netron.start(onnx_path)
#     except Exception as e:
#         print(f"无法自动启动 Netron (可能是服务器环境): {e}")
#         print(f"请手动下载 {onnx_path} 并使用 https://netron.app 查看。")