import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math

from torch.nn.init import trunc_normal_
from einops import rearrange


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv1(nn.Module):
    # Standard convolution
    # [64, 6, 2, 2] -> [3，32，6，2，2]
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Identity() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Tanh() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Sigmoid() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = AconC(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = MetaAconC(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = SiLU_beta(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU_noBN_biasFalse(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU_noBN_biasTrue(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Partition_module_v2(nn.Module):
    """
    使用dfc attention
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, mode='original'):
        super().__init__()
        # use RELU as partition_Conv 本来是SiLU
        self.mode = mode
        act = nn.ReLU() if act else act
        # todo original p-block
        if self.mode in ['original']:
            c_ = c2 // 4
            self.cv1 = Conv1(c1, c_, k, s, None, g, act)  # ch_in, ch_out, kernel, stride, pad, groups
            self.cv2_3x3 = Conv1(c_, 2 * c_, 3, 1, None, c_, act)
            self.maxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        elif self.mode in ['super', 'Super']:
            c_ = c2 // 4
            self.cv1 = Conv1(c1, c_, k, s, None, g, act)  # ch_in, ch_out, kernel, stride, pad, groups
            self.cv2_3x3 = Conv1(c_, 2 * c_, 3, 1, None, c_, act)
            self.maxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

            self.dfc_attention = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                Conv1(c1, c2, 1, 1, None, act=False),
                nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Sigmoid(),
                # nn.Upsample(scale_factor=2, mode='nearest')
            )
        elif self.mode in ['super333']:
            c_ = c2 // 4
            self.cv1 = Conv1(c1, c_, k, s, None, g, act)  # ch_in, ch_out, kernel, stride, pad, groups
            self.cv2_3x3 = Conv1(c_, 2 * c_, 3, 1, None, c_, act)
            self.maxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

            self.dfc_attention333 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                Conv(c1, c2, 3, 1, None, act=False),
                nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Sigmoid(),
            )
        elif self.mode in ['super7x7']:
            c_ = c2 // 4
            self.cv1 = Conv1(c1, c_, k, s, None, g, act)  # ch_in, ch_out, kernel, stride, pad, groups
            self.cv2_3x3 = Conv1(c_, 2 * c_, 3, 1, None, c_, act)
            self.maxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

            self.dfc_attention7x7 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                Conv(c1, c2, 1, 1, None, act=False),
                nn.Conv2d(c2, c2, kernel_size=(1, 7), stride=1, padding=(0, 3), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=(7, 1), stride=1, padding=(3, 0), groups=c2, bias=False),
                nn.BatchNorm2d(c2),
                nn.Sigmoid(),
            )
        else:
            pass

    def forward(self, x):
        if self.mode in ['original']:
            x1 = self.cv1(x)  # 3
            x2 = self.cv2_3x3(x1)  # 3
            pool = self.maxPool(x1)  # 3
            x4 = torch.cat((x1, x2, pool), dim=1)
            # return x4[:, :self.out, :, :]
            return x4
        elif self.mode in ['super']:
            dfc_atte = self.dfc_attention(x)
            x1 = self.cv1(x)
            x2 = self.cv2_3x3(x1)  #
            pool = self.maxPool(x1)  #
            x4 = torch.cat((x1, x2, pool), dim=1)
            # x5 = x4[:, :self.out, :, :] * dfc_atte
            x5 = x4 * F.interpolate(dfc_atte, size=(x4.shape[-2], x4.shape[-1]), mode='nearest')
            return x + x5
        elif self.mode in ['super333']:
            dfc_atte = self.dfc_attention333(x)
            x1 = self.cv1(x)
            x2 = self.cv2_3x3(x1)  #
            pool = self.maxPool(x1)  #
            x4 = torch.cat((x1, x2, pool), dim=1)
            x5 = x4 * F.interpolate(dfc_atte, size=(x4.shape[-2], x4.shape[-1]), mode='nearest')
            return x + x5
        elif self.mode in ['super7x7']:
            dfc_atte = self.dfc_attention7x7(x)
            x1 = self.cv1(x)
            x2 = self.cv2_3x3(x1)  #
            pool = self.maxPool(x1)  #
            x4 = torch.cat((x1, x2, pool), dim=1)
            x5 = x4 * F.interpolate(dfc_atte, size=(x4.shape[-2], x4.shape[-1]), mode='nearest')
            return x + x5


class PBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, mode='super'):
        super(PBottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv1(c1, c_, 1, 1)
        self.cv2 = Partition_module_v2(c_, c2, mode=mode)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        if self.shortcut:
            y = self.cv2(self.cv1(x))
            return x + y
        else:
            return self.cv2(self.cv1(x))


class C3P(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, mode='super', g=1, e=0.5):
        super(C3P, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv1(c1, c_, 1, 1)
        self.cv2 = Conv1(c1, c_, 1, 1)
        self.cv3 = Conv1(2 * c_, c2, 1, 1)
        self.p = nn.Sequential(*(PBottleneck(c_, c_, shortcut, g=g, e=1.0, mode=mode) for _ in range(n)))

    def forward(self, x):
        y1 = torch.cat((self.cv2(x), self.p(self.cv1(x))), dim=1)
        y2 = self.cv3(y1)
        return y2
        pass


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
    def __init__(self, nIn, nOut, kSize, stride, padding=1, bn_act=False, dilation=(1, 1), groups=1, bias=False):
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


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension)


class AvgPool(nn.Module):
    """平均池化下采样模块，保留3通道信息"""

    def __init__(self, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 保留3通道信息
        return self.pool(x)


# 多步平均池化下采样模块
class MulAvgPool(nn.Module):
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
        return None


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
        # 示例：B=1,H=2,W=2 → mask = [[[False,False],[False,False]]]
        not_mask = ~mask  # 取反掩码（全True） # 取反 → [[[True,True],[True,True]]]
        # 计算y方向的位置累积（模拟行索引）| cumsum(1)：沿维度1（高度）累积，True=1，False=0
        # 示例：not_mask=[[[1,1],[1,1]]] → cumsum(1)后：
        # y_embed = [[[1.0, 1.0], [2.0, 2.0]]]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 计算x方向的位置累积（模拟列索引）
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # x_embed 示例：[[[1.0, 2.0], [1.0, 2.0]]]
        eps = 1e-6  # 防止除零的小常数
        # 归一化y方向位置到[0, 2π]
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        # 归一化x方向位置到[0, 2π]
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 生成频率维度的缩放因子| //向下取整
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

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
        self.proj = nn.Linear(dim, dim)  # 输出投影层 WO
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
        q = torch.nn.functional.normalize(q, dim=-1)  # 作用相当于除以sqrt(d_h)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力权重：(Q @ K^T) * 温度参数 → softmax → dropout
        # 矩阵乘法结果：(2,6,8,8)（每个头的 “通道 × 通道” 互协方差矩阵）
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权V，调整维度并投影输出| 用通道的注意力权重，加权每个像素的特征
        # 第一步attn @ v：
        # attn
        # 维度：(2, 6, 8, 8)，v
        # 维度：(2, 6, 8, 1920)；
        # 乘法结果：(2, 6, 8, 1920)（用通道的注意力权重，加权每个像素的特征）；
        # 意义：每个通道的特征都融合了全局关联通道的信息，最终每个像素的特征都有全局视野。
        # 第二步permute(0, 3, 1, 2)：调整维度为(2, 1920, 6, 8)（把 “序列维度” 1920提前，方便合并多头）；
        # 第三步reshape(B, N, C)：把6×8
        # 合并为48，维度回到(2, 1920, 48)（和输入维度一致）
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore  # 忽略TorchScript编译
    def no_weight_decay(self):
        # 指定不需要权重衰减的参数（温度参数）
        return {'temperature'}


class GAM(nn.Module):
    def __init__(self, dim, use_pos_emb=True, eps=1e-6, kernel_function=nn.ReLU()):
        super().__init__()
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)  # 初始化位置编码
        self.linear_Q = nn.Linear(dim, dim)
        self.linear_K = nn.Linear(dim, dim)
        self.linear_V = nn.Linear(dim, dim)

        self.linear_Q1 = nn.Linear(dim, dim)
        self.linear_K1 = nn.Linear(dim, dim)

        self.eps = eps
        self.kernel_fun = kernel_function
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()

        if self.pos_embd:
            # 生成位置编码并添加到特征中
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        # x=x+self.pos_emb
        Q = self.linear_Q(x)  # blc
        K = self.linear_K(x)  # blc
        V = self.linear_V(x)  # blc
        Q1 = self.linear_Q1(x)
        K1 = self.linear_K1(x)

        Q = self.kernel_fun(Q)
        K = self.kernel_fun(K)

        K = K.transpose(-2, -1)  # bcl
        KV = torch.einsum("bml, blc->bmc", K, V)  # bcc

        Z = 1 / (torch.einsum("blc,bc->bl", Q, K.sum(dim=-1) + self.eps))  # bl

        result = torch.einsum("blc,bcc,bl->blc", Q, KV, Z)  # blc

        mid_result = (Q1.transpose(-1, -2) @ K1).softmax(dim=-1)
        result = result @ mid_result

        x = x + self.gamma * result
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.act(self.norm(x))

        return x


class GLGXCA(nn.Module):
    """
    Gated Local-Global Cross-Covariance Attention (GL-XCA)

    改进点：
    1. Local: 在 Value 生成路径加入 3x3 DWConv，引入局部归纳偏置。
    2. Gated: 使用门控机制 V = Linear(x) * Act(DWConv(x))，筛选有效特征。
    3. Global: 保持 XCA 的线性复杂度 Q^T @ K。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1. Q 和 K 的生成 (保持轻量级 Linear)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        # 2. Gated Local Value 生成模块
        # 分支 A: 内容流 (Linear)
        self.v_content = nn.Linear(dim, dim, bias=qkv_bias)
        # 分支 B: 局部与门控流 (DWConv)
        # groups=dim 实现了 Depthwise Convolution
        self.v_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=qkv_bias),
            nn.BatchNorm2d(dim),  # 增加 BN 稳定训练
            nn.SiLU()  # Swish 激活函数，适合门控
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 适配两种输入格式：
        # 1. (B, N, C) -> Transformer 标准格式
        # 2. (B, C, H, W) -> CNN 标准格式

        is_feature_map = False
        if x.dim() == 4:
            # 输入是 (B, C, H, W)，记录尺寸并展平以便处理 Q/K
            is_feature_map = True
            B, C, H, W = x.shape
            x_img = x
            # 展平为 (B, N, C) 用于 Q/K/V_content
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            N = H * W
        else:
            # 输入是 (B, N, C)，需要知道 H, W 才能做卷积
            # 如果是在纯 Transformer 块中，可能需要传入 size，或者强制 reshape
            # 这里为了通用性，假设输入已经是 (B, C, H, W) 或者是 flat 但我们无法得知 H,W
            # *注意*：为了使用 DWConv，必须知道 H和W。
            # 如果你的架构只传 (B,N,C)，请确保有办法推导 H,W。
            # 下面假设输入必须是 (B, C, H, W) 以发挥局部性优势。
            raise ValueError("GL-XCA requires input with spatial dimensions (B, C, H, W) to apply Local DWConv.")

        # ----------------------------------------------------------------
        # Step 1: 生成 Q 和 K (Global Path)
        # ----------------------------------------------------------------
        # q, k: (B, N, C) -> (B, N, heads, dim_head) -> permute
        q = self.q(x_flat).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(x_flat).reshape(B, N, self.num_heads, C // self.num_heads)

        # 变换维度用于 XCA 计算: (B, heads, dim_head, N)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)

        # 归一化 (XCA 特征) - 沿 N 维度归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算通道协方差注意力图 (B, heads, dim_head, dim_head)
        # (dim_head, N) @ (N, dim_head) -> (dim_head, dim_head)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ----------------------------------------------------------------
        # Step 2: 生成 Gated Local Value (Local + Gated Path)
        # ----------------------------------------------------------------
        # Path A: Content (Linear)
        v_content = self.v_content(x_flat)  # (B, N, C)

        # Path B: Spatial Gate (DWConv)
        # x_img 是 (B, C, H, W)
        v_gate = self.v_gate(x_img)  # (B, C, H, W)
        v_gate = v_gate.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C)

        # Gating Operation: 内容 * 空间门控
        # 这里的 v 融合了全局语义(通过Linear)和局部几何(通过DWConv)
        v = v_content * v_gate

        # 调整 V 的维度以适配多头: (B, heads, dim_head, N)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)

        # ----------------------------------------------------------------
        # Step 3: Global Aggregation & Projection
        # ----------------------------------------------------------------
        # 协方差注意力加权: (dim_head, dim_head) @ (dim_head, N) -> (dim_head, N)
        # 每个通道特征通过全局相关性进行重组
        x_out = (attn @ v)

        # 还原维度: (B, heads, dim_head, N) -> (B, N, heads, dim_head) -> (B, N, C)
        x_out = x_out.permute(0, 3, 1, 2).reshape(B, N, C)

        # 输出投影
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        # 恢复回图像格式 (B, C, H, W)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x_out

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


# GSA gated_self-attention?
class LGSA(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()
        self.dim = dim  # 输入维度
        self.pos_embd = None  # 位置编码（可选）
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)  # 初始化位置编码
        # 1. use XCA
        self.norm1 = LayerNorm(dim, eps=1e-6)  # 支持 (B, N, C)
        self.glgxca = GLGXCA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.norm2 = LayerNorm(dim, eps=1e-6)
        # FFN 结构: 1x1 Conv -> 3x3 DWConv -> GELU -> 1x1 Conv
        hidden_dim = int(dim * expan_ratio)
        self.mlp = nn.Sequential(
            # 1. 升维 (1x1 Conv)
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            # 2. 局部特征提取 (3x3 Depthwise Conv) -> 这是 "Local" 的核心！
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            # 3. 降维 (1x1 Conv)
            nn.Conv2d(hidden_dim, dim, 1)
        )

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        B, C, H, W = x.shape
        # ----------------------------------------------------------
        # Branch 1: Global Attention (XCA)
        # -----------------------------------------------------------
        input_ = x  # 残差连接点 1
        # 维度变换: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        if self.pos_embd:
            # 假设 pos_embd 返回 (B, C, H, W)，需要调整维度
            pos = self.pos_embd(B, H, W).flatten(2).transpose(1, 2)
            x_flat = x_flat + pos

        x_norm = self.norm1(x_flat)  # B h*w C
        # 将 Norm 后的数据 Reshape 回 (B, C, H, W) 传给 GLGXCA
        x_norm_spatial = x_norm.transpose(1, 2).reshape(B, C, H, W)
        # Pre-Norm -> XCA
        x_attn = self.glgxca(x_norm_spatial)  # (B, 48, H, 640)
        # Layer Scale
        if self.gamma1 is not None:
            # x_attn = self.gamma1 * x_attn
            x_attn = x_attn * self.gamma1.view(1, C, 1, 1)

        # 变回图像格式: (B, H*W, C) -> (B, C, H, W) 用于残差相加
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        # Residual Connection 1
        x = input_ + self.drop_path(x_attn)
        # -----------------------------------------------------------
        # Branch 2: Local MLP (ConvFFN with DWConv)
        # -----------------------------------------------------------
        input_ = x  # 残差连接点 2
        # 为了使用 Conv2d 实现 MLP，不需要把维度变平
        # 但我们需要 Norm，LayerNorm 通常对通道在最后一维工作
        # Norm: (B, C, H, W) -> Permute -> Norm -> Permute
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)

        # MLP (包含 3x3 DWConv)
        x_mlp = self.mlp(x_norm)
        # Layer Scale (需要处理维度匹配，gamma 是 (C))
        if self.gamma2 is not None:
            x_mlp = x_mlp * self.gamma2.view(1, C, 1, 1)
        # Residual Connection 2
        x = input_ + self.drop_path(x_mlp)
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

        # 相当于MLP layer
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


class LGSA_EDFFN(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()
        self.dim = dim  # 输入维度
        self.pos_embd = None  # 位置编码（可选）
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)  # 初始化位置编码
        # 1. use XCA
        self.norm1 = LayerNorm(dim, eps=1e-6)  # 支持 (B, N, C)
        self.glgxca = GLGXCA(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             attn_drop=attn_drop, proj_drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.norm2 = LayerNorm(dim, eps=1e-6)
        # FFN 结构: 1x1 Conv -> 3x3 DWConv -> GELU -> 1x1 Conv
        # hidden_dim = int(dim * expan_ratio)
        # self.mlp = nn.Sequential(
        #     # 1. 升维 (1x1 Conv)
        #     nn.Conv2d(dim, hidden_dim, 1),
        #     nn.GELU(),
        #     # 2. 局部特征提取 (3x3 Depthwise Conv) -> 这是 "Local" 的核心！
        #     nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        #     nn.GELU(),
        #     # 3. 降维 (1x1 Conv)
        #     nn.Conv2d(hidden_dim, dim, 1)
        # )
        self.edffn = EDFFN(dim, patch_size=4, ffn_expansion_factor=expan_ratio)

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        B, C, H, W = x.shape
        # ----------------------------------------------------------
        # Branch 1: Global Attention (XCA)
        # -----------------------------------------------------------
        input_ = x  # 残差连接点 1
        # 维度变换: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        if self.pos_embd:
            # 假设 pos_embd 返回 (B, C, H, W)，需要调整维度
            pos = self.pos_embd(B, H, W).flatten(2).transpose(1, 2)
            x_flat = x_flat + pos

        x_norm = self.norm1(x_flat)  # B h*w C
        # 将 Norm 后的数据 Reshape 回 (B, C, H, W) 传给 GLGXCA
        x_norm_spatial = x_norm.transpose(1, 2).reshape(B, C, H, W)
        # Pre-Norm -> XCA
        x_attn = self.glgxca(x_norm_spatial)  # (B, 48, H, 640)
        # Layer Scale
        if self.gamma1 is not None:
            # x_attn = self.gamma1 * x_attn
            x_attn = x_attn * self.gamma1.view(1, C, 1, 1)

        # 变回图像格式: (B, H*W, C) -> (B, C, H, W) 用于残差相加
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        # Residual Connection 1
        x = input_ + self.drop_path(x_attn)
        # -----------------------------------------------------------
        # Branch 2: Local MLP (ConvFFN with DWConv)
        # -----------------------------------------------------------
        input_ = x  # 残差连接点 2
        # 为了使用 Conv2d 实现 MLP，不需要把维度变平
        # 但我们需要 Norm，LayerNorm 通常对通道在最后一维工作
        # Norm: (B, C, H, W) -> Permute -> Norm -> Permute
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)

        # MLP (包含 3x3 DWConv)
        # x_mlp = self.mlp(x_norm)

        x_edffn = self.edffn(x_norm)

        # Layer Scale (需要处理维度匹配，gamma 是 (C))
        if self.gamma2 is not None:
            x_edffn = x_edffn * self.gamma2.view(1, C, 1, 1)
        # Residual Connection 2
        x = input_ + self.drop_path(x_edffn)
        return x

class EDFFN(nn.Module):
    def __init__(self, dim, patch_size, ffn_expansion_factor=4, bias=True):
        super(EDFFN, self).__init__()
        # 计算隐藏层的特征维度，通常是输入维度的若干倍
        hidden_features = int(dim * ffn_expansion_factor)
        # 保存patch大小，用于后续分块处理
        self.patch_size = patch_size
        self.dim = dim

        # 第一个1x1卷积层，用于提升特征维度，输出维度是隐藏层维度的两倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 深度可分离卷积，对每个通道单独处理，进一步提取特征
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        # 可学习的FFT参数，用于频域操作
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        # 第二个1x1卷积层，用于将特征维度降回输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 通过第一个卷积层提升特征维度【提升维度】
        x = self.project_in(x)
        # 经过深度可分离卷积后，将输出沿通道维度分成两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对第一部分应用GELU激活函数，然后与第二部分相乘
        x = F.gelu(x1) * x2
        # 通过第二个卷积层降低特征维度【降低维度】
        x = self.project_out(x)

        # 将特征图按指定patch大小进行分块重组
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        # 对分块后的特征图进行二维快速傅里叶变换，转换到频域
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 在频域中应用可学习的参数，对频域特征进行加权调整
        x_patch_fft = x_patch_fft * self.fft
        # 进行二维逆快速傅里叶变换，将特征从频域转回空间域
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 将分块的特征图重新组合成完整的特征图
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                      patch1=self.patch_size, patch2=self.patch_size)
        return x


def cal_params(model):
    """计算模型参数数量"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:.0f}")


if __name__ == '__main__':
    m = LGFI(48, 0.01, 1e-06, 6, True, 8)
    im = torch.rand(1, 48, 192, 640)
    out = m(im)
    # print(out.shape)
    cal_params(m)

    m1 = LGSA(48, 0.01, 1e-06, 6, True, 8)
    im1 = torch.rand(1, 48, 192, 640)
    out1 = m1(im1)
    # print(out.shape)
    cal_params(m1)
    # m1 = GAM(48)
    # out2 = m1(im)
    # # print(out2.shape)
    # cal_params(m1)
    m1 = LGSA_EDFFN(48, 0.01, 1e-06, 4, True, 8)
    im1 = torch.rand(1, 48, 192, 640)
    out1 = m1(im1)
    # print(out.shape)
    cal_params(m1)



    pass
