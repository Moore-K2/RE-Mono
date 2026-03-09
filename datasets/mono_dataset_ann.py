# 从__future__导入特性，确保代码在Python2和Python3下兼容（绝对导入、除法、打印函数）
from __future__ import absolute_import, division, print_function

# 导入必要的库：os用于文件路径操作，random用于随机数生成
import os
import random
# numpy用于数值计算，copy用于对象复制
import numpy as np
import copy
# PIL的Image用于图像加载和处理（使用pillow-simd加速）
from PIL import Image

# 导入PyTorch及相关工具：torch是核心库，data用于数据集处理，transforms用于图像变换
import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    """加载图像的函数，使用PIL库

    Args:
        path: 图像文件的路径
    Returns:
        转换为RGB格式的图像对象
    """
    # 以二进制只读模式打开文件
    with open(path, 'rb') as f:
        # 打开图像并转换为RGB格式（避免RGBA等其他格式）
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """单目数据集加载器的基类（继承自PyTorch的Dataset）

    用于加载单目视觉数据（可能包含时间序列或立体对图像），为深度估计、视觉里程计等任务提供数据

    Args:
        data_path: 数据集的根目录路径
        filenames: 包含数据文件名的列表（每个元素可能包含文件夹、帧索引等信息）
        height: 图像需要调整到的高度
        width: 图像需要调整到的宽度
        frame_idxs: 需要加载的帧索引（如[-1,0,1]表示前一帧、当前帧、后一帧，"s"表示立体对的另一视角）
        num_scales: 图像金字塔的尺度数量（用于多尺度模型输入）
        is_train: 是否为训练模式（训练模式会启用数据增强）
        img_ext: 图像文件的扩展名（如.jpg、.png）
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        # 调用父类Dataset的初始化方法
        super(MonoDataset, self).__init__()

        # 初始化数据集参数
        self.data_path = data_path  # 数据根路径
        self.filenames = filenames  # 文件名列表
        self.height = height  # 目标图像高度
        self.width = width  # 目标图像宽度
        self.num_scales = num_scales  # 图像金字塔尺度数
        self.interp = Image.ANTIALIAS  # 图像插值方法（抗锯齿，用于缩放）

        self.frame_idxs = frame_idxs  # 需要加载的帧索引

        self.is_train = is_train  # 是否训练模式
        self.img_ext = img_ext  # 图像扩展名

        self.loader = pil_loader  # 图像加载函数
        self.to_tensor = transforms.ToTensor()  # 用于将PIL图像转换为PyTorch张量

        # 处理颜色增强参数（兼容不同版本的torchvision）
        # 尝试新版本的元组格式参数；如果失败则使用旧版本的标量格式
        try:
            # 亮度、对比度、饱和度、色调的调整范围（元组表示上下限）
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # 测试参数是否兼容（新版本接受元组）
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            # 旧版本使用标量（表示±该值的范围）
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # 初始化不同尺度的图像缩放器
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i  # 缩放因子（第i尺度为原图的1/2^i）
            # 存储每个尺度的Resize变换（目标尺寸为原始高度/宽度除以2^i）
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # 检查当前数据集是否包含深度真值（由子类实现）
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """对图像进行预处理：缩放到不同尺度，并应用颜色增强（如果需要）

        所有图像使用相同的增强参数，确保输入到模型的图像增强一致（如时间序列或立体对）

        Args:
            inputs: 存储图像数据的字典（键为(frame_id, scale)等，值为图像）
            color_aug: 颜色增强变换（可为空操作）
        """
        # 先对所有图像进行多尺度缩放（基于-1尺度的原始加载图像）
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                # k的格式为(n, im, i)，其中i为尺度（初始为-1，即原始尺寸）
                n, im, i = k
                # 为每个尺度生成缩放后的图像（从i-1尺度缩放，避免多次从原图缩放损失精度）
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # 将缩放后的图像转换为张量，并应用颜色增强
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # 原始图像（未增强）转换为张量
                inputs[(n, im, i)] = self.to_tensor(f)
                # 增强后的图像转换为张量
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.filenames)

    def __getitem__(self, index):
        """获取单个训练样本（核心方法）

        返回一个字典，包含以下键值对：
            ("color", <frame_id>, <scale>): 原始彩色图像（张量）
            ("color_aug", <frame_id>, <scale>): 增强后的彩色图像（张量）
            ("K", scale) / ("inv_K", scale): 相机内参矩阵及逆矩阵（张量）
            "stereo_T": 立体相机外参（如果包含立体对）
            "depth_gt": 深度真值（如果有）

        <frame_id>：帧索引（如-1/0/1表示时间帧，"s"表示立体对另一视角）
        <scale>：尺度（0为原始目标尺寸，1为1/2，2为1/4等）
        """
        inputs = {}  # 存储样本数据的字典

        # 随机决定是否进行颜色增强和水平翻转（仅训练模式）
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # 解析文件名列表中的当前样本信息（文件夹、帧索引、视角）
        line = self.filenames[index].split()  # 按空格分割文件名行
        folder = line[0]  # 样本所在文件夹

        # 解析帧索引（如果文件名行有3个元素，则第二个为帧索引）
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        # 解析视角（左右眼，如'l'/'r'，如果文件名行有3个元素，则第三个为视角）
        if len(line) == 3:
            side = line[2]
        else:
            side = None

        # 加载所有需要的帧（根据frame_idxs）
        for i in self.frame_idxs:
            if i == "s":
                # 如果是立体对的另一视角，切换左右（如'l'→'r'，'r'→'l'）
                other_side = {"r": "l", "l": "r"}[side]
                # 加载另一视角的图像（可能需要翻转）
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                # 加载时间帧（当前帧±i）
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # 为每个尺度调整相机内参（K）和逆内参（inv_K）
        for scale in range(self.num_scales):
            K = self.K.copy()  # 复制原始内参（self.K由子类定义）

            # 内参随图像缩放而调整（焦距和主点坐标按尺度缩放）
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)  # 计算内参的逆矩阵

            # 存储内参和逆内参（转换为张量）
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 初始化颜色增强变换（如果需要）
        if do_color_aug:
            # 创建颜色抖动变换（亮度、对比度、饱和度、色调调整）
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            # 不增强时，使用恒等变换（输入即输出）
            color_aug = (lambda x: x)

        # 对图像进行预处理（多尺度缩放和增强）
        self.preprocess(inputs, color_aug)

        # 删除原始尺度（-1）的图像（已生成其他尺度，无需保留）
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # 如果数据集包含深度真值，加载并处理
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            # 增加一个维度（变为[1, H, W]，符合PyTorch的通道优先格式）
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            # 转换为float32张量
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # 如果包含立体对（frame_idxs中有"s"），添加立体相机外参
        if "s" in self.frame_idxs:
            # 初始化立体变换矩阵（4x4，表示右相机到左相机的变换）
            stereo_T = np.eye(4, dtype=np.float32)
            # 基线方向（翻转时基线方向反转）
            baseline_sign = -1 if do_flip else 1
            # 左右视角的基线符号（右视角为正，左为负）
            side_sign = -1 if side == "l" else 1
            # 设置平移分量（假设基线长度为0.1米，实际需根据数据集调整）
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        """获取指定的彩色图像（需要子类实现，因数据集路径格式不同）

        Args:
            folder: 图像所在文件夹
            frame_index: 帧索引
            side: 视角（'l'/'r'）
            do_flip: 是否水平翻转图像
        Returns:
            加载并可能翻转后的PIL图像
        """
        raise NotImplementedError  # 未实现错误（子类必须重写）

    def check_depth(self):
        """检查数据集是否包含深度真值（需要子类实现）

        Returns:
            bool: 若有深度真值则为True，否则为False
        """
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        """获取指定的深度真值（需要子类实现）

        Args:
            同get_color
        Returns:
            深度图像的numpy数组
        """
        raise NotImplementedError