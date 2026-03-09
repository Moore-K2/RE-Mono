# @Author: BigMoore
# @Time: 2025/11/28 17:51
# 从__future__导入绝对导入、除法、打印函数等特性，确保Python2/3兼容性
from __future__ import absolute_import, division, print_function

# 导入时间模块，用于计时
import time

import cv2
# 导入PyTorch的优化器模块
import torch.optim as optim
from sympy.physics.units import current
from torch.utils.checkpoint import checkpoint
# 导入PyTorch的数据加载器，用于批处理数据
from torch.utils.data import DataLoader
# 导入tensorboardX的SummaryWriter，用于日志记录和可视化
from tensorboardX import SummaryWriter

# 导入json模块，用于处理JSON格式数据
import json

from tqdm import tqdm

# 导入自定义工具函数
from utils import *
# 导入KITTI数据集相关工具函数
from kitti_utils import *
# 导入自定义层
from layers import *

# 导入数据集模块
import datasets
# 导入网络模块
import networks
# 导入自定义的学习率调度器
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

from networks.base_model_encoder import Model
from networks.monoVit import DeepNet
from networks import BiDepthDecoder, BiDepthDecoder2, LiteMono, ReMono,VitDepthDecoder,DepthDecoder,Mono2DepthDecoder, Mono2ResnetEncoder

# 设置CUDA基准测试模式（注释掉，可能根据需求开启）
# torch.backends.cudnn.benchmark = True


def time_sync():
    # 同步CUDA操作（如果可用），确保时间测量准确
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 返回当前时间
    return time.time()


class TrainerDist:
    def __init__(self, options):
        # 保存配置参数
        self.teacher_encoder_out_channels_group = None
        self.student_encoder_out_channels_group = None

        self.opt = options
        #----------------------------------------------------------------
        # 定义日志路径：日志目录+模型名称
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # 检查输入图像的高和宽是否为32的倍数（网络结构要求）
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        # 初始化模型字典（深度相关模型）
        self.models = {}
        # 初始化姿态模型字典（姿态估计相关模型）
        self.models_pose = {}
        # 初始化需要训练的参数列表（深度相关）
        self.parameters_to_train = []
        # 初始化需要训练的参数列表（姿态相关）
        self.parameters_to_train_pose = []

        # 定义设备：CPU或GPU
        # self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        # 定义设备：CPU或指定ID的GPU
        self.device = torch.device(f"cuda:{self.opt.gpu_id}" if not self.opt.no_cuda else "cpu")

        # self.prepare_teacher()
        self.prepare_teacher_adapter()

        # 是否启用性能分析
        self.profile = self.opt.profile

        # 获取尺度数量
        self.num_scales = len(self.opt.scales)
        # 获取帧ID数量
        self.frame_ids = len(self.opt.frame_ids)
        # 姿态网络输入帧数量：如果是"pairs"模式则为2，否则为输入帧总数
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # 断言帧ID必须以0开头（0通常表示参考帧）
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # 判断是否使用姿态网络：当不使用立体视觉且帧ID仅为[0]时不使用
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        # 如果使用立体视觉，在帧ID中添加"s"（表示立体对的另一视角）
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        model_struct = "\033[31m"'\n【训练结构】using '
        if self.opt.cfg_first:
            print(f'\n----------using cfg:{self.opt.cfg} to construct depth_encoder model---------')
            self.models["encoder"] = Model(self.opt.cfg, width=self.opt.width, height=self.opt.height)
            # 将编码器移动到指定设备
            self.models["encoder"].to(self.device)
            # 将编码器参数添加到训练参数列表
            self.parameters_to_train += list(self.models["encoder"].parameters())
        else:
            print('\n----------Using original file to construct depth_encoder model----------')
            #TODO 初始化encoder
            en = self.opt.depth_encoder
            encoder = eval(en) if isinstance(en, str) else en
            self.models['encoder'] = encoder()
            self.models['encoder'].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
            model_struct += f'encoder:{en}, '

        #TODO 初始化decoder
        de = self.opt.depth_decoder
        decoder = eval(de) if isinstance(de, str) else de
        if decoder in (DepthDecoder, BiDepthDecoder, BiDepthDecoder2, Mono2DepthDecoder):
            args = [self.models["encoder"].num_ch_enc, self.opt.scales]
            decoder = decoder(*args) # modules
            self.student_encoder_out_channels_group = self.models["encoder"].num_ch_enc
        elif decoder is VitDepthDecoder:
            decoder = decoder()

        self.models['depth'] = decoder
        self.models['depth'].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 打印训练信息
        print_use_kd = f"使用{self.opt.use_kd}知识蒸馏" if self.opt.use_kd else "未使用知识蒸馏"
        model_size = self.opt.model_size
        model_struct += f'depth decoder:{de}, {print_use_kd}, 模型{model_size}'"\033[0m"
        print(model_struct)

        lr_depth_max, lr_depth_min, lr_depth_step, lr_pose_max, lr_pose_min, lr_pose_step = self.opt.lr
        lr_info = f"lr_depth(max/min/step)={lr_depth_max:.6f}/{lr_depth_min:.6f}/{lr_depth_step}, lr_pose(max/min/step)={lr_pose_max:.6f}/{lr_pose_min:.6f}/{lr_pose_step}"
        red_print_content = (
            "\033[31m"  # 开启红色字体
            f"【训练参数】model_name={self.opt.model_name}, 训练尺寸(h×w)={self.opt.height}×{self.opt.width}, use_kd:{self.opt.use_kd}, warm_up_epochs:{self.opt.warmup_steps}, val_epochs:{self.opt.val_of_epochs},batch_size={self.opt.batch_size}, epochs={self.opt.num_epochs}, {lr_info}"
            "\033[0m"  # 恢复默认字体颜色
        )
        print(red_print_content)

        # TODO
        print(f'teacher.num_ch_enc:{self.teacher_encoder_out_channels_group}, student.num_ch_enc:{self.student_encoder_out_channels_group}, ')
        # self.adapters = nn.ModuleList()
        # for t_c, s_c in zip(self.teacher_encoder_out_channels_group, self.student_encoder_out_channels_group):
        #     if t_c != s_c:
        #         # 定义 1x1 卷积适配器
        #         adapter = nn.Conv2d(t_c, s_c, kernel_size=1, stride=1, padding=0, bias=False)
        #         # 初始化权重 (可选，但推荐)
        #         nn.init.kaiming_normal_(adapter.weight, mode='fan_out', nonlinearity='relu')
        #         self.adapters.append(adapter)
        #     else:
        #         self.adapters.append(nn.Identity())  # 如果通道相同，不做处理
        # # 确保将 self.adapters 移动到 GPU，并加入到优化器中
        # self.adapters.to(self.device)
        # self.parameters_to_train += list(self.adapters.parameters())


        # 如果使用姿态网络
        if self.use_pose_net:
            # 姿态模型类型为"separate_resnet"（独立的ResNet编码器）
            if self.opt.pose_model_type == "separate_resnet":
                # TODO 初始化姿态编码器（ResNet）
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,  # 网络层数
                    self.opt.weights_init == "pretrained",  # 是否使用预训练权重
                    num_input_images=self.num_pose_frames)  # 输入图像数量

                # 将姿态编码器移动到指定设备
                self.models_pose["pose_encoder"].to(self.device)
                # 添加姿态编码器参数到训练列表
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                # 初始化姿态解码器
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,  # 姿态编码器输出通道数
                    num_input_features=1,  # 输入特征数
                    num_frames_to_predict_for=2)  # 预测的帧数
                # #TODO 初始化姿态解码器
                # self.models_pose["pose"] = networks.PoseDecoder(
                #     self.models_pose["pose_encoder"].num_ch_enc,  # 姿态编码器输出通道数
                #     num_input_features=1,  # 输入特征数
                #     num_frames_to_predict_for=1)  # 预测的帧数

            # 姿态模型类型为"shared"（与深度编码器共享）
            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc,  # 共享深度编码器的输出通道数
                    self.num_pose_frames)  # 姿态输入帧数

            # 姿态模型类型为"posecnn"（特定的姿态CNN）
            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    # 输入帧数：如果是"all"模式则为总输入帧数，否则为2
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            # 将姿态解码器移动到指定设备
            self.models_pose["pose"].to(self.device)
            # 添加姿态解码器参数到训练列表
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        # 如果使用预测掩码（predictive_mask）
        if self.opt.predictive_mask:
            # 断言：使用预测掩码时必须禁用自动掩码
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # 初始化预测掩码解码器（与深度解码器架构相同，输出通道数为源帧数-1）
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            # 移动到指定设备
            self.models["predictive_mask"].to(self.device)
            # 添加到训练参数列表
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # 初始化深度相关模型的优化器（AdamW）
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        # 如果使用姿态网络，初始化姿态模型的优化器
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3],
                                                    weight_decay=self.opt.weight_decay)
        # 初始化深度模型的学习率调度器（自定义的ChainedScheduler）
        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),  # 初始周期
            T_mul=1,  # 周期倍增因子
            eta_min=self.opt.lr[1],  # 最小学习率
            last_epoch=-1,  # 上一个epoch
            max_lr=self.opt.lr[0],  # 最大学习率
            warmup_steps=0,  # 热身步数
            gamma=0.9  # 衰减因子
        )
        # 初始化姿态模型的学习率调度器
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        # 如果指定了加载权重的文件夹，加载模型
        if self.opt.load_weights_folder is not None:
            self.load_model()

        # 如果指定了自定义预训练模型，加载预训练权重
        if self.opt.mypretrain is not None:
            self.load_pretrain()

        # 打印训练信息
        print("Training model named:", self.opt.model_name)
        print("Models and tensorboard events files are saved to:  ", self.opt.log_dir)
        print("Training is using:  ", self.device)

        # 数据集字典：KITTI原始数据集和KITTI里程计数据集
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        # 根据配置选择数据集类
        self.dataset = datasets_dict[self.opt.dataset]

        # 定义数据集文件路径（训练/验证文件列表）
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        # 读取训练和验证文件列表
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        # TODO test
        test_filenames = readlines(fpath.format("test"))
        # 图像扩展名：根据配置选择png或jpg
        img_ext = '.png' if self.opt.png else '.jpg'

        # 计算训练样本数量和总训练步数
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # 初始化训练数据集
        train_dataset = self.dataset(
            self.opt.data_path,  # 数据路径
            train_filenames,  # 训练文件列表
            self.opt.height, self.opt.width,  # 图像高宽
            self.opt.frame_ids,  # 帧ID列表
            4,  # 缩放因子（可能用于数据增强）
            is_train=True,  # 训练模式
            img_ext=img_ext)  # 图像扩展名
        # 初始化训练数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,  # 批大小
            True,  # 打乱数据
            num_workers=self.opt.num_workers,  # 工作进程数
            pin_memory=True,  # 锁存内存
            persistent_workers=True,
            drop_last=True)  # 丢弃最后一个不完整的批次
        # 初始化验证数据集
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height, self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,  # 验证模式
            img_ext=img_ext)
        # 初始化验证数据加载器
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        # 创建验证迭代器
        self.val_iter = iter(self.val_loader)

        #TODO 创建test数据集
        test_dataset = self.dataset(
            self.opt.data_path,
            test_filenames,
            self.opt.height, self.opt.width,
            [0],
            4,
            is_train=False,  # 测试模式
            img_ext=img_ext)
        # 初始化测试数据加载器
        self.test_loader = DataLoader(
            test_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=False)
        # 创建测试迭代器
        self.test_iter = iter(self.test_loader)

        # TODO 初始化日志写入器（训练和验证）
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # 如果不禁用SSIM，初始化SSIM损失计算器
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # 初始化深度反投影和3D投影层（不同尺度）
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            # 计算当前尺度下的图像高宽
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # 初始化深度反投影层（将深度图转换为3D点云）
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            # 初始化3D投影层（将3D点云投影到图像平面）
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        # 深度评估指标名称
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # 打印数据集信息
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        # === 新增：初始化最佳指标记录 ===
        self.best_abs_rel = float('inf')  # 初始设为无穷大
        # 保存配置参数
        self.save_opts()

    def prepare_teacher(self):
        """准备教师模型：包括 Encoder 和 Depth Decoder"""
        self.teacher_models = {}

        # 1. 初始化教师 Encoder
        self.teacher_models["encoder"] = LiteMono(model="lite-mono", height=self.opt.height, width=self.opt.width)
        # 2. 初始化教师 Decoder
        self.teacher_models["depth"] = DepthDecoder(self.teacher_models["encoder"].num_ch_enc,scales=range(3))

        # 3. 加载预训练权重 (假设你的pth文件里同时包含了 encoder 和 depth 的权重)
        # 通常 Lite-Mono 的权重文件是一个字典，包含 'encoder' 和 'depth' 两个 key，或者键名有前缀
        checkpoint = {}
        encoder_path = "./pretrained/lite_mono_encoder.pth"
        decoder_path = "./pretrained/lite_mono_depth.pth"
        checkpoint["encoder"] = torch.load(encoder_path, map_location="cpu")
        checkpoint["depth"] = torch.load(decoder_path, map_location="cpu")
        print(f'\n\n已加载teacher model of encoder path: {encoder_path},  decoder path: {decoder_path}')

        # 加载 encoder 权重
        # 注意：有时候pth里的key是 "encoder.conv1..."，需要处理一下前缀，或者直接 load_state_dict
        # 这里假设 checkpoint 直接对应模型结构，如果不对应需要做 key 匹配
        # self.teacher_models["encoder"].load_state_dict(checkpoint["encoder"])
        # self.teacher_models["depth"].load_state_dict(checkpoint["depth"])

        self.teacher_models["encoder"].load_state_dict(checkpoint["encoder"], strict=False)
        self.teacher_models["depth"].load_state_dict(checkpoint["depth"], strict=False)

        # 4. 移动到 GPU 并冻结参数
        for k in ["encoder", "depth"]:
            self.teacher_models[k].to(self.device)
            self.teacher_models[k].eval()  # 开启验证模式
            for param in self.teacher_models[k].parameters():
                param.requires_grad = False  # 关闭梯度

    def prepare_teacher_adapter(self):
        """准备教师模型：包括 Encoder 和 Depth Decoder"""
        self.teacher_models = {}

        # 1. 初始化教师 Encoder
        self.teacher_models["encoder"] = LiteMono(model="lite-mono", height=self.opt.height, width=self.opt.width)
        # 2. 初始化教师 Decoder
        self.teacher_models["depth"] = DepthDecoder(self.teacher_models["encoder"].num_ch_enc, scales=range(3))

        self.teacher_encoder_out_channels_group = self.teacher_models["encoder"].num_ch_enc

        # 3. 加载预训练权重 (假设你的pth文件里同时包含了 encoder 和 depth 的权重)
        # 通常 Lite-Mono 的权重文件是一个字典，包含 'encoder' 和 'depth' 两个 key，或者键名有前缀
        checkpoint = {}
        encoder_path = "./pretrained/lite_mono_encoder.pth"
        decoder_path = "./pretrained/lite_mono_depth.pth"
        checkpoint["encoder"] = torch.load(encoder_path, map_location="cpu")
        checkpoint["depth"] = torch.load(decoder_path, map_location="cpu")
        print(f'\n\n已加载teacher model of encoder path: {encoder_path},  decoder path: {decoder_path}')

        # 加载 encoder 权重
        # 注意：有时候pth里的key是 "encoder.conv1..."，需要处理一下前缀，或者直接 load_state_dict
        # 这里假设 checkpoint 直接对应模型结构，如果不对应需要做 key 匹配
        # self.teacher_models["encoder"].load_state_dict(checkpoint["encoder"])
        # self.teacher_models["depth"].load_state_dict(checkpoint["depth"])

        self.teacher_models["encoder"].load_state_dict(checkpoint["encoder"], strict=False)
        self.teacher_models["depth"].load_state_dict(checkpoint["depth"], strict=False)

        # 4. 移动到 GPU 并冻结参数
        for k in ["encoder", "depth"]:
            self.teacher_models[k].to(self.device)
            self.teacher_models[k].eval()  # 开启验证模式
            for param in self.teacher_models[k].parameters():
                param.requires_grad = False  # 关闭梯度

    def set_train(self):
        """将所有模型设置为训练模式
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """将所有模型设置为评估模式
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """运行整个训练流程
        """
        self.epoch = 0  # 初始化epoch
        self.step = 0  # 初始化全局步数
        self.start_time = time.time()  # 记录开始时间
        # 遍历每个epoch
        for self.epoch in range(self.opt.num_epochs):
            # 运行单个epoch，获取epoch统计指标
            epoch_metrics = self.run_epoch()
            # 打印当前epoch的指标
            self.print_epoch_metrics(epoch_metrics)

            # 每隔指定频率保存模型
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """运行单个epoch的训练和验证，新增epoch指标统计逻辑
        Returns:
            dict: 包含当前epoch的统计指标（平均损失、深度指标、耗时等）
        """
        # ===================== 新增：初始化epoch统计变量 =====================
        epoch_start_time = time.time()  # 记录epoch开始时间
        epoch_losses = []  # 累加每个批次的总损失（全量，非仅日志批次）
        # 初始化深度指标累加器（KITTI标准指标）
        depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        depth_metrics_accum = {metric: [] for metric in depth_metric_names}
        has_depth_gt = False  # 标记当前epoch是否有深度真值

        print("Training")
        self.set_train()  # 设置模型为训练模式

        # 更新学习率调度器
        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        # 遍历训练数据加载器中的批次
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()  # 记录操作开始时间

            # 处理批次数据，得到输出和损失
            outputs, losses = self.process_batch(inputs)

            # 清零优化器梯度
            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            # 反向传播计算梯度
            losses["loss"].backward()
            # 更新参数
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            # 计算当前批次处理时间
            duration = time.time() - before_op_time

            # ===================== 新增：累加当前批次总损失 =====================
            epoch_losses.append(losses["loss"].cpu().item())

            # ===================== 新增：计算并累加深度指标（全量批次） =====================
            if "depth_gt" in inputs:
                has_depth_gt = True
                # 不管是否满足日志条件，都计算深度指标|指标都在losses里面
                self.compute_depth_losses(inputs, outputs, losses)
                # 累加深度指标到epoch级列表 | losses里面有总损失，以及这一轮的指标
                for metric in depth_metric_names:
                    if metric in losses:  # 确保指标已计算
                        depth_metrics_accum[metric].append(losses[metric])

            # 日志记录频率：前20000步按指定频率，之后每2000步记录一次
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            # 如果满足日志条件
            if early_phase or late_phase:
                # 记录时间信息
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # 记录日志（深度指标已提前计算）
                self.log("train", inputs, outputs, losses)
                # 进行验证
                self.val()
            self.step += 1

        # ===================== 新增：统计epoch级指标 =====================
        epoch_metrics = {}
        # 1. 计算epoch耗时
        epoch_metrics["epoch_time"] = time.time() - epoch_start_time
        # 2. 计算平均总损失
        epoch_metrics["avg_total_loss"] = np.mean(epoch_losses) if epoch_losses else "N/A"
        # 3. 计算深度指标均值（仅当有深度真值时）
        if has_depth_gt:
            for metric in depth_metric_names:
                if depth_metrics_accum[metric]:  # 确保有数据
                    epoch_metrics[metric] = np.mean(depth_metrics_accum[metric])
                else:
                    epoch_metrics[metric] = "N/A"
        else:
            epoch_metrics["depth_info"] = "无深度真值，未计算深度指标"

        return epoch_metrics  # 返回epoch统计指标

    def print_epoch_metrics(self, epoch_metrics):
        current_epoch = self.epoch  # 从0开始
        total_epochs = self.opt.num_epochs - 1
        # 打印深度指标（仅当有数据时）
        log_str = "深度评估指标（KITTI）:\n"
        depth_metrics = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        if "de/abs_rel" in epoch_metrics and epoch_metrics["de/abs_rel"] != "N/A":
            log_str += f"Epoch [{current_epoch}/{total_epochs}] | "
            for metric in depth_metrics:
                val = epoch_metrics[metric]
                if val != "N/A":
                    # 误差类指标保留4位，精度类（a1/a2/a3）保留3位
                    fmt = ".4f" if "rel" in metric or "rmse" in metric else ".3f"
                    log_str += f"  {metric.split('/')[-1]}: {val:{fmt}} | "
            log_str = log_str.rstrip(" | ") + "\n"  # 移除最后一个分隔符
        elif "depth_info" in epoch_metrics:
            log_str += f"深度指标: {epoch_metrics['depth_info']}\n"

        # 打印到控制台
        print(log_str)
        # 写入到model_name_train_result.txt文件
        if hasattr(self.opt, "model_name") and self.opt.model_name:
            file_name = f"{self.opt.model_name}_train_result.txt"
        else:
            file_name = "lite-mono_train_result.txt"  # 默认文件名

        # 可选：将文件保存到日志目录（self.log_dir），若不需要可直接用file_name
        save_path = os.path.join(self.opt.log_dir, self.opt.model_name, file_name)
        # 确保文件所在目录存在（避免目录不存在报错）
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # 以追加模式写入文件（a+：不存在则创建，存在则追加）
        with open(save_path, "a+", encoding="utf-8") as f:
            # 写入当前epoch的指标，可添加时间戳（可选）
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"[{time_str}] {log_str}\n")  # 加时间戳便于追溯；若不需要时间戳，直接f.write(log_str + "\n")

    # ===================== train_v1 or run_epoch_v1 =====================
    """新增在每次epoch结束后，记录验证指标到txt文件"""
    def run_epoch_v1(self):
        """运行单个epoch的训练和验证
        """
        print("Training")
        self.set_train()  # 设置模型为训练模式
        # 更新学习率调度器
        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        # 遍历训练数据加载器中的批次
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()  # 记录操作开始时间
            # 处理批次数据，得到输出和损失
            # outputs, losses = self.process_batch(inputs)
            # outputs, losses = self.process_batch_distillation(inputs)
            # [structure_kd, kd]
            if self.opt.use_kd == 'kd':
                outputs, losses = self.process_batch_distillation(inputs)
            elif self.opt.use_kd == 'structure_kd':
                outputs, losses = self.process_batch_structure_distillation(inputs)
            else:
                outputs, losses = self.process_batch(inputs)

            # 清零优化器梯度
            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            # 反向传播计算梯度
            losses["loss"].backward()
            # 更新参数
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            # 计算当前批次处理时间
            duration = time.time() - before_op_time

            # 日志记录频率：前20000步按指定频率，之后每2000步记录一次
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            # 如果满足日志条件
            if early_phase or late_phase:
                # 记录时间信息
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses["loss/distill"].cpu().data)
                # 如果输入包含深度真值，计算深度损失指标
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                # 记录日志
                self.log("train", inputs, outputs, losses)
                # 进行验证
                self.val()
            self.step += 1
            # ✅ 完整释放：新增gc.collect()，强制Python垃圾回收
            del inputs, outputs, losses
            torch.cuda.empty_cache()  # 清理 GPU 无用显存，还给显卡；
            import gc  # 清理 CPU 无用内存 + 碎片，还给 Windows 系统；
            gc.collect()

    def train_v1(self):
        """运行整个训练流程
        """
        self.epoch = 0  # 初始化epoch
        self.step = 0  # 初始化全局步数
        self.start_time = time.time()  # 记录开始时间
        # 遍历每个epoch
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch_v1()  # 运行单个epoch
            # 一个 Epoch 跑完后，进行全量验证
            if self.epoch>=self.opt.val_of_epochs:
                self.val_full_metrics_v2()
                # self.test_full_metrics_v2()
            # 每隔指定频率保存模型
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

        print("Training finished after {} epochs,".format(self.epoch))

    def val_full_metrics_v2(self):
        """
        运行全量验证集评估，计算标准指标，并保存最佳模型
        """
        print("\n" + "-" * 20 + " Running Full Validation " + "-" * 20)
        self.set_eval()  # 切换到评估模式
        # 初始化指标累加器
        errors = []
        val_losses = []
        has_depth_gt = False
        # KITTI 深度估计的标准指标顺序
        depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        depth_metrics_accum = {metric: [] for metric in depth_metric_names}
        # 1. 遍历验证集
        with torch.no_grad():
            # for batch_idx, inputs in enumerate(tqdm(self.val_loader, desc="Validating", ncols=80, unit="batch")):
            # for batch_idx, inputs in enumerate(self.val_loader):
            for batch_idx, inputs in enumerate(tqdm(self.val_loader, desc="📊 Validating (batch)")):
                # 处理批次数据
                outputs, losses = self.process_batch(inputs)
                # 累加损失
                val_losses.append(losses["loss"].cpu().item())
                if "depth_gt" not in inputs:
                    continue
                has_depth_gt = True
                self.compute_depth_losses(inputs, outputs, losses)
                # 计算误差 (compute_depth_errors )
                # 返回: [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
                # 累加深度指标
                for metric in depth_metric_names:
                    if metric in losses:
                        depth_metrics_accum[metric].append(losses[metric])

                # # 可选：打印进度
                # if batch_idx % 10 == 0:
                #     print(f"Validation batch {batch_idx}/{len(self.val_loader)}")
                # 释放内存
                del inputs, outputs, losses
        # 计算平均指标
        val_metrics = {}
        val_metrics["avg_total_loss"] = np.mean(val_losses) if val_losses else "N/A"
        # 计算深度指标均值
        if has_depth_gt:
            for metric in depth_metric_names:
                if depth_metrics_accum[metric]:
                    val_metrics[metric] = np.mean(depth_metrics_accum[metric])
                else:
                    val_metrics[metric] = "N/A"
        else:
            val_metrics["depth_info"] = "无深度真值，未计算深度指标"
        # 取出深度指标值"abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        all_metrics = [val_metrics[e] for e in depth_metric_names]

        all_metrics = np.array(all_metrics)
        # 格式化输出
        metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        print("\n" + "=" * 80)
        print(f"Epoch {self.epoch} Full Validation Results:")
        header = "{:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(*metric_names)
        values = "{:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.3f} | {:10.3f} | {:10.3f}".format(*all_metrics)
        print(header)
        print(values)
        print("=" * 80 + "\n")

        # 8. 写入文件
        self.log_val_results(all_metrics)

        # 9. 保存最佳模型 (Best Model Saving)
        current_abs_rel = all_metrics[0]  # abs_rel 是最重要的指标
        if current_abs_rel < self.best_abs_rel:
            print(f"Update Best Model! Previous Best: {self.best_abs_rel:.4f} -> New Best: {current_abs_rel:.4f}")
            self.best_abs_rel = current_abs_rel
            self.save_model(is_best=True)  # 需要修改一下 save_model 支持 is_best

        # 恢复训练模式
        self.set_train()

    def compute_errors(self, gt, pred):
        """计算KITTI标准7个深度评估指标，和你测试代码完全一致"""
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def batch_post_process_disparity(self, l_disp, r_disp):
        """视差后处理：左右视图融合，提升精度，和你测试代码完全一致"""
        _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
        r_mask = l_mask[:, :, ::-1]
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def test_full_metrics_v2(self):
        """
        【极简版】仅计算+打印KITTI测试集7个核心深度指标，和val_full_metrics_v2格式完全一致
        无推理速度、无参数量、无FLOPs、无耗时统计，只保留纯指标评估逻辑
        """
        print("\n" + "=" * 80)
        print(f"Epoch {self.epoch} | Running Full KITTI Test Set Evaluation (Eigen Split)")
        print("=" * 80)
        self.set_eval()  # 切换到评估模式，关闭BN/Dropout
        # KITTI深度评估标准阈值
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        # 初始化指标累加器，和验证函数完全一致
        depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        depth_metrics_accum = {metric: [] for metric in depth_metric_names}
        has_depth_gt = True

        # 无梯度推理核心逻辑
        with torch.no_grad():
            # 遍历测试集dataloader，进度条保留，方便看进度
            for batch_idx, inputs in enumerate(
                    tqdm(self.test_loader, desc="📊 Testing (batch)", ncols=80, unit="batch")):
                input_color = inputs[("color", 0, 0)].to(self.device)

                # 【可选】视差后处理：左右翻转拼接，配置文件开关控制，不打印任何信息
                if self.opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # 纯推理，无计时、无参数量计算
                output = self.depth_decoder(self.encoder(input_color))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

                # 视差转numpy
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                # 视差后处理（可选）
                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                # 获取真值深度+裁剪+计算指标（核心逻辑）
                gt_depth = inputs["depth_gt"].cpu().numpy()
                for i in range(pred_disp.shape[0]):
                    pred_disp_i = pred_disp[i]
                    gt_depth_i = gt_depth[i, 0]  # 真值深度维度 [B,1,H,W]

                    # 调整预测视差尺寸和真值一致
                    h_gt, w_gt = gt_depth_i.shape
                    pred_disp_i = cv2.resize(pred_disp_i, (w_gt, h_gt))
                    pred_depth_i = 1.0 / pred_disp_i  # 视差转深度

                    # Eigen Split 标准裁剪（KITTI必需）
                    mask = np.logical_and(gt_depth_i > MIN_DEPTH, gt_depth_i < MAX_DEPTH)
                    crop = np.array([0.40810811 * h_gt, 0.99189189 * h_gt,
                                     0.03594771 * w_gt, 0.96405229 * w_gt]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                    # 过滤有效区域
                    pred_depth_i = pred_depth_i[mask]
                    gt_depth_i = gt_depth_i[mask]

                    # 中位数缩放（单目深度必备，无打印）
                    if not self.opt.disable_median_scaling:
                        ratio = np.median(gt_depth_i) / np.median(pred_depth_i)
                        pred_depth_i *= ratio

                    # 限制深度范围，防止异常值
                    pred_depth_i[pred_depth_i < MIN_DEPTH] = MIN_DEPTH
                    pred_depth_i[pred_depth_i > MAX_DEPTH] = MAX_DEPTH

                    # 计算7个标准指标，并累加
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.compute_errors(gt_depth_i, pred_depth_i)
                    batch_metrics = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
                    for metric_name, metric_val in zip(depth_metric_names, batch_metrics):
                        depth_metrics_accum[metric_name].append(metric_val)

                # 释放显存
                del inputs, output, pred_disp

        # ========== 计算平均指标，和 val_full_metrics_v2 完全一致 ==========
        test_metrics = {}
        if has_depth_gt:
            for metric in depth_metric_names:
                test_metrics[metric] = np.mean(depth_metrics_accum[metric]) if depth_metrics_accum[metric] else "N/A"
        else:
            test_metrics["depth_info"] = "无深度真值，未计算深度指标"

        # 取出指标值
        all_metrics = [test_metrics[e] for e in depth_metric_names]
        all_metrics = np.array(all_metrics)

        # ========== 纯指标打印，格式和 val_full_metrics_v2 一模一样！无任何多余内容 ==========
        metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        print("\n" + "-" * 80)
        print(f"Epoch {self.epoch} Full Test Set Results (Eigen Split):")
        header = "{:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(*metric_names)
        values = "{:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.3f} | {:10.3f} | {:10.3f}".format(*all_metrics)
        print(header)
        print(values)
        print("-" * 80 + "\n")

        # 写入日志 + 保存最佳模型（和验证逻辑一致，无多余打印）
        self.log_val_results(all_metrics)
        current_abs_rel = all_metrics[0]
        if current_abs_rel < self.best_abs_rel:
            print(f"🔥 Update Best Model! Previous Best: {self.best_abs_rel:.4f} -> New Best: {current_abs_rel:.4f}")
            self.best_abs_rel = current_abs_rel
            self.save_model(is_best=True)

        # 恢复训练模式
        self.set_train()
        print("=" * 80 + "\n")
        return all_metrics

    def val_full_metrics(self):
        """
        运行全量验证集评估，计算标准指标，并保存最佳模型
        """
        print("\n" + "-" * 20 + " Running Full Validation " + "-" * 20)
        self.set_eval()  # 切换到评估模式
        # 初始化指标累加器
        errors = []
        # KITTI 深度估计的标准指标顺序
        # metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        # 1. 遍历验证集
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                # 移动数据到设备
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                # 检查是否有深度真值
                if "depth_gt" not in inputs:
                    continue
                # 2. 前向传播 (只跑 Encoder 和 Depth Decoder)
                # 注意：验证时通常使用未增强的图像 ("color", 0, 0)
                features = self.models["encoder"](inputs["color", 0, 0])
                outputs = self.models["depth"](features)
                # 3. 后处理预测结果
                # 取出 scale=0 的预测深度 (sigmoid输出)
                pred_disp = outputs[("disp", 0)]
                # 上采样到与 GT 相同的尺寸 (KITTI 原始尺寸通常是 375x1242)
                # 使用 bilinear 插值，align_corners=False
                _, depth_pred = disp_to_depth(pred_disp, self.opt.min_depth, self.opt.max_depth)
                depth_pred = F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False)

                # 4. 准备真值和掩码
                depth_gt = inputs["depth_gt"]
                mask = depth_gt > 0

                # 5. 应用 Garg/Eigen Crop (这是单目深度估计学术界的标准操作)
                # 裁剪区域：Top: 153, Bottom: 371, Left: 44, Right: 1197
                crop_mask = torch.zeros_like(mask)
                crop_mask[:, :, 153:371, 44:1197] = 1
                mask = mask * crop_mask

                # 6. 计算指标 (逐张图片计算)
                depth_pred = depth_pred.detach()
                depth_gt = depth_gt.detach()
                batch_size = depth_gt.shape[0]
                for i in range(batch_size):
                    # 提取当前图片的有效像素
                    gt_i = depth_gt[i][mask[i]]
                    pred_i = depth_pred[i][mask[i]]

                    if len(gt_i) == 0:
                        continue
                    # === Median Scaling (自监督单目的关键) ===
                    # 因为自监督不知道绝对尺度，所以用中值对齐
                    scale_factor = torch.median(gt_i) / (torch.median(pred_i) + 1e-8)
                    pred_i = pred_i * scale_factor
                    # 截断预测值范围 (0-80m 是 KITTI 标准)
                    pred_i = torch.clamp(pred_i, min=1e-3, max=80)
                    # 计算误差 (compute_depth_errors 来自 utils.py)
                    # 返回: [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
                    current_errors = compute_depth_errors(gt_i, pred_i)
                    # errors.append(current_errors.cpu().numpy())
                    errors.append([e.item() for e in current_errors])
                # 可选：打印进度
                if batch_idx % 10 == 0:
                    print(f"Validation batch {batch_idx}/{len(self.val_loader)}")
        # 7. 统计全量平均指标
        if len(errors) > 0:
            mean_errors = np.array(errors).mean(0)
        else:
            print("Warning: No valid validation data found.")
            self.set_train()
            return

        # 格式化输出
        metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        print("\n" + "=" * 80)
        print(f"Epoch {self.epoch} Full Validation Results:")
        header = "{:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(*metric_names)
        values = "{:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.3f} | {:10.3f} | {:10.3f}".format(*mean_errors)
        print(header)
        print(values)
        print("=" * 80 + "\n")

        # 8. 写入文件
        self.log_val_results(mean_errors)

        # 9. 保存最佳模型 (Best Model Saving)
        current_abs_rel = mean_errors[0]  # abs_rel 是最重要的指标
        if current_abs_rel < self.best_abs_rel:
            print(f"Update Best Model! Previous Best: {self.best_abs_rel:.4f} -> New Best: {current_abs_rel:.4f}")
            self.best_abs_rel = current_abs_rel
            self.save_model(is_best=True)  # 需要修改一下 save_model 支持 is_best

        # 恢复训练模式
        self.set_train()

    def log_val_results(self, mean_errors):
        """将验证结果写入 txt 文件"""
        file_name = "val_results_log.txt"
        save_path = os.path.join(self.log_path, file_name)

        with open(save_path, "a+") as f:
            # 如果是第一行，写入表头
            if f.tell() == 0:
                f.write("Epoch, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3\n")

            line = f"{self.epoch}, " + ", ".join([f"{x:.5f}" for x in mean_errors]) + "\n"
            f.write(line)

    def save_model(self, is_best=False):
        """将模型权重保存到磁盘
        """

        if is_best:
            # 如果是最佳模型，保存在 models/best_model 文件夹
            save_folder = os.path.join(self.log_path, "models", "best_model")
        else:
            # 常规保存
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        # # 保存文件夹路径：日志路径/models/weights_{epoch}
        # save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 保存深度相关模型
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            # 对于编码器，额外保存输入尺寸和是否使用立体视觉（预测时需要）
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # 保存姿态相关模型
        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        # 保存优化器状态
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        # 保存姿态优化器状态（如果使用）
        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    # ===================== train_v1 or run_epoch_v1 =====================
    def process_batch(self, inputs):
        """将一个批次数据传入网络，生成图像和损失
        """
        # 将输入数据移动到指定设备
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # 如果姿态模型类型为"shared"（与深度编码器共享）
        if self.opt.pose_model_type == "shared":
            # 所有增强后的图像拼接在一起，输入到编码器
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            # 按批次分割特征（每个帧对应一组特征）
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            # 为每个帧ID保存特征
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            # 使用参考帧（0）的特征计算深度
            outputs = self.models["depth"](features[0])
        else:
            #TODO-here 否则，仅将参考帧（0）输入到深度编码器
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            # 计算深度
            outputs = self.models["depth"](features)

        # 如果使用预测掩码，计算掩码
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 如果使用姿态网络，预测姿态
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # 生成预测的重投影图像
        self.generate_images_pred(inputs, outputs)
        # 计算损失
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def process_batch_distillation(self, inputs):
        """将一个批次数据传入网络，生成图像和损失
        """
        # print('使用蒸馏')
        # 将输入数据移动到指定设备s
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # ====================================================
        # [新增] 教师模型推理 (Knowledge Distillation)
        # ====================================================
        teacher_disp = None
        # 只有在训练阶段且定义了教师模型时才执行
        if hasattr(self, "teacher_models"):
            with torch.no_grad():
                # 老师只需要看当前帧 (Frame 0)
                # 保持和学生一样的输入 (color_aug)
                teacher_img = inputs[("color_aug", 0, 0)]
                # 1. 老师提取特征
                teacher_features = self.teacher_models["encoder"](teacher_img)
                # 2. 老师解码深度
                teacher_outputs = self.teacher_models["depth"](teacher_features)
                # 获取最高分辨率的视差图 (Scale 0) 作为“标准答案”
                # .detach() 很重要，确保不传梯度给老师
                teacher_disp = teacher_outputs[("disp", 0)].detach()

        # 如果姿态模型类型为"shared"（与深度编码器共享）
        if self.opt.pose_model_type == "shared":
            # 所有增强后的图像拼接在一起，输入到编码器
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            # 按批次分割特征（每个帧对应一组特征）
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            # 为每个帧ID保存特征
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            # 使用参考帧（0）的特征计算深度
            outputs = self.models["depth"](features[0])
        else:
            #TODO-here 否则，仅将参考帧（0）输入到深度编码器
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            # 计算深度
            outputs = self.models["depth"](features)

        # 如果使用预测掩码，计算掩码
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 如果使用姿态网络，预测姿态
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # 生成预测的重投影图像
        self.generate_images_pred(inputs, outputs)
        # 计算损失
        # ====================================================
        # [修改] 计算损失
        # ====================================================
        # 1.先计算原始的自监督损失 (Photo + Smoothness)
        losses = self.compute_losses(inputs, outputs)
        # 2. 计算蒸馏损失 (Distillation Loss) 并加到总 Loss 中
        if teacher_disp is not None:
            # 取出学生预测的视差图 (Scale 0)
            student_disp = outputs[("disp", 0)]
            # 计算 L1 Loss (也可以用 MSE Loss)
            # 注意：有时候需要对齐尺寸，但通常 Scale 0 是一样的
            distill_loss = torch.abs(student_disp - teacher_disp).mean()
            # 记录到日志字典中
            losses["loss/distill"] = distill_loss
            # 将蒸馏损失加权添加到总损失 (权重系数 0.1 可调)
            # 你的架构是从零训练，建议初期权重稍微大一点，比如 0.1 或 0.5
            losses["loss"] += 0.1 * distill_loss

        return outputs, losses

    def process_batch_structure_distillation(self, inputs):
        """将一个批次数据传入网络，生成图像和损失
        """
        # print('使用蒸馏')
        # 将输入数据移动到指定设备s
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # ====================================================
        # [新增] 教师模型推理 (Knowledge Distillation)
        # ====================================================
        # 只有在训练阶段且定义了教师模型时才执行
        if hasattr(self, "teacher_models"):
            with torch.no_grad():
                # 老师只需要看当前帧 (Frame 0)
                # 保持和学生一样的输入 (color_aug)
                teacher_img = inputs[("color_aug", 0, 0)]
                # 1. 老师提取特征
                teacher_features = self.teacher_models["encoder"](teacher_img)
                # 2. 老师解码深度
                teacher_outputs = self.teacher_models["depth"](teacher_features)
                # 获取最高分辨率的视差图 (Scale 0) 作为“标准答案”
                # .detach() 很重要，确保不传梯度给老师
                teacher_disp = teacher_outputs[("disp", 0)].detach()

        # 如果姿态模型类型为"shared"（与深度编码器共享）
        if self.opt.pose_model_type == "shared":
            # 所有增强后的图像拼接在一起，输入到编码器
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            # 按批次分割特征（每个帧对应一组特征）
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            # 为每个帧ID保存特征
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            # 使用参考帧（0）的特征计算深度
            outputs = self.models["depth"](features[0])
        else:
            #TODO-here 否则，仅将参考帧（0）输入到深度编码器
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            # 计算深度
            outputs = self.models["depth"](features)

        # 如果使用预测掩码，计算掩码
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 如果使用姿态网络，预测姿态
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # 生成预测的重投影图像
        self.generate_images_pred(inputs, outputs)
        # 计算损失
        # ====================================================
        # [修改] 计算损失
        # ====================================================
        # 1.先计算原始的自监督损失 (Photo + Smoothness)
        losses = self.compute_losses(inputs, outputs)
        # 2. 计算蒸馏损失 (Distillation Loss) 并加到总 Loss 中
        if teacher_disp is not None:
            # 带温度的KL散度（软标签蒸馏）
            T = 4.0  # 温度系数，T越大软标签越平滑（论文常用4-6）
            # 教师输出软化
            student_disp = outputs[("disp", 0)]
            # 计算 L1 Loss (也可以用 MSE Loss)
            # 注意：有时候需要对齐尺寸，但通常 Scale 0 是一样的
            output_distill_loss = F.l1_loss(student_disp / T, teacher_disp / T)

            # distill_loss = torch.abs(student_disp/ T - teacher_disp/ T).mean()


            # （2）中间层特征蒸馏损失：结构化特征对齐（论文中的SFD）
            feature_distill_loss = 0.0
            # 对齐师生的多尺度特征（如teacher_features和student_features各有4个尺度）
            for t_feat, s_feat in zip(teacher_features, features):
                # 对齐特征的维度（若师生特征维度不同，用1×1卷积转换）
                if t_feat.shape[1] != s_feat.shape[1]:
                    t_feat = torch.nn.Conv2d(t_feat.shape[1], s_feat.shape[1], 1).to("cuda")(t_feat)
                # 结构化特征损失：MSE（或用余弦相似度）
                feature_distill_loss += F.mse_loss(s_feat, t_feat)
            # 平均多尺度特征的损失
            feature_distill_loss /= len(teacher_features)
            # ---------------------- 总损失：任务损失 + 蒸馏损失 ----------------------
            alpha = 0.1  # 输出蒸馏损失的权重
            beta = 0.3  # 特征蒸馏损失的权重（论文中更重要）
            distill_loss = alpha * output_distill_loss + beta * feature_distill_loss
            # 记录到日志字典中
            losses["loss/distill"] = distill_loss
            losses["loss"] +=  distill_loss

        return outputs, losses

    def process_batch_structure_distillation_adapter(self, inputs):
        """将一个批次数据传入网络，生成图像和损失
        """
        # print('使用蒸馏')
        # 将输入数据移动到指定设备s
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # ====================================================
        # [新增] 教师模型推理 (Knowledge Distillation)
        # ====================================================
        # 只有在训练阶段且定义了教师模型时才执行
        if hasattr(self, "teacher_models"):
            with torch.no_grad():
                # 老师只需要看当前帧 (Frame 0)
                # 保持和学生一样的输入 (color_aug)
                teacher_img = inputs[("color_aug", 0, 0)]
                # 1. 老师提取特征
                teacher_features = self.teacher_models["encoder"](teacher_img)
                # 2. 老师解码深度
                teacher_outputs = self.teacher_models["depth"](teacher_features)
                # 获取最高分辨率的视差图 (Scale 0) 作为“标准答案”
                # .detach() 很重要，确保不传梯度给老师
                teacher_disp = teacher_outputs[("disp", 0)].detach()

        # 如果姿态模型类型为"shared"（与深度编码器共享）
        if self.opt.pose_model_type == "shared":
            # 所有增强后的图像拼接在一起，输入到编码器
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            # 按批次分割特征（每个帧对应一组特征）
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            # 为每个帧ID保存特征
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            # 使用参考帧（0）的特征计算深度
            outputs = self.models["depth"](features[0])
        else:
            #TODO-here 否则，仅将参考帧（0）输入到深度编码器
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            # 计算深度
            outputs = self.models["depth"](features)

        # 如果使用预测掩码，计算掩码
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 如果使用姿态网络，预测姿态
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # 生成预测的重投影图像
        self.generate_images_pred(inputs, outputs)
        # 计算损失
        # ====================================================
        # [修改] 计算损失
        # ====================================================
        # 1.先计算原始的自监督损失 (Photo + Smoothness)
        losses = self.compute_losses(inputs, outputs)
        # 2. 计算蒸馏损失 (Distillation Loss) 并加到总 Loss 中
        if teacher_disp is not None:
            # 带温度的KL散度（软标签蒸馏）
            T = 4.0  # 温度系数，T越大软标签越平滑（论文常用4-6）
            # 教师输出软化
            student_disp = outputs[("disp", 0)]
            # 计算 L1 Loss (也可以用 MSE Loss)
            # 注意：有时候需要对齐尺寸，但通常 Scale 0 是一样的
            output_distill_loss = F.l1_loss(student_disp / T, teacher_disp / T)

            # distill_loss = torch.abs(student_disp/ T - teacher_disp/ T).mean()


            # （2）中间层特征蒸馏损失：结构化特征对齐（论文中的SFD）
            feature_distill_loss = 0.0
            # 对齐师生的多尺度特征（如teacher_features和student_features各有4个尺度）
            # for t_feat, s_feat in zip(teacher_features, features):
            #     # 对齐特征的维度（若师生特征维度不同，用1×1卷积转换）
            #     if t_feat.shape[1] != s_feat.shape[1]:
            #         t_feat = torch.nn.Conv2d(t_feat.shape[1], s_feat.shape[1], 1).to("cuda")(t_feat)
            #
            #
            #     # 结构化特征损失：MSE（或用余弦相似度）
            #     feature_distill_loss += F.mse_loss(s_feat, t_feat)
            for i, (t_feat, s_feat) in enumerate(zip(teacher_features, features)):
                # 1. 使用预定义好的、可学习的适配器处理老师特征
                # 注意：self.adapters[i] 已经在 GPU 上且权重是固定的（或者正在被优化的）
                aligned_t_feat = self.adapters[i](t_feat)

                # 2. 计算损失
                feature_distill_loss += F.mse_loss(s_feat, aligned_t_feat)
            # 平均多尺度特征的损失
            feature_distill_loss /= len(teacher_features)
            # ---------------------- 总损失：任务损失 + 蒸馏损失 ----------------------
            alpha = 0.1  # 输出蒸馏损失的权重
            beta = 0.3  # 特征蒸馏损失的权重（论文中更重要）
            distill_loss = alpha * output_distill_loss + beta * feature_distill_loss
            # 记录到日志字典中
            losses["loss/distill"] = distill_loss
            losses["loss"] +=  distill_loss

        return outputs, losses

    def predict_poses(self, inputs, features):
        """预测单目序列中输入帧之间的姿态
        """
        outputs = {}
        # 如果姿态输入帧数为2（ pairwise 模式）
        if self.num_pose_frames == 2:
            # 为每个源帧通过姿态网络单独计算姿态

            # 选择姿态网络的输入特征
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in
                              self.opt.frame_ids}  # 姿态网络与深度网络共享编码器，直接用深度编码器提取的多帧特征（features）作为输入；
            else:  # 用原始增强图像（color_aug）作为输入
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            # 遍历除参考帧外的其他帧
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":  # 排除立体帧"s"
                    # 保持时间顺序：负帧ID在前，正帧ID在后
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    # 根据姿态模型类型处理输入
                    if self.opt.pose_model_type == "separate_resnet":
                        # 姿态编码器处理拼接的输入
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        # 直接拼接输入
                        pose_inputs = torch.cat(pose_inputs, 1)

                    # 预测轴角和平移
                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # TODO 根据帧ID是否为负，决定是否反转变换矩阵; 这儿取的是axisangle第二个维度是 “预测的位姿数量”
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # 输入所有帧到姿态网络，同时预测所有姿态
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                # 拼接除立体帧外的所有增强图像
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                # 如果是separate_resnet，通过姿态编码器处理
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            # 如果是shared类型，使用各帧的特征
            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            # 预测轴角和平移
            axisangle, translation = self.models_pose["pose"](pose_inputs)

            # 为每个源帧保存姿态变换
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    # TODO新 版 PyTorch 中DataLoader迭代器不再支持.next()方法，需要将代码中的self.val_iter.next()改为 Python 内置的next(self.val_iter)即可解决
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """为批次生成扭曲（重投影）的彩色图像，保存到outputs字典
        利用模型预测的视差（深度）和相机姿态，将其他帧（如前一帧、后一帧、双目帧）的图像 “扭曲”（重投影）到参考帧的视角
        ，生成重投影图像，用于后续通过光度损失（photometric loss）监督模型学习。
        """
        for scale in self.opt.scales:
            # 获取当前尺度的视差图
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                # v1多尺度模式：源尺度与当前尺度相同
                source_scale = scale
            else:
                # 否则将视差图上采样到原始尺寸，源尺度为0
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # TODO 将视差转换为深度（min_depth和max_depth限制范围）
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # 保存当前尺度的深度图
            outputs[("depth", 0, scale)] = depth

            # 遍历除参考帧外的其他帧
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                # 获取变换矩阵T：立体帧使用预定义的stereo_T，其他帧使用预测的姿态
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # 如果是posecnn模型，调整平移（来自论文https://arxiv.org/abs/1712.00175）
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    # 计算逆深度及其均值
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    # 用平均逆深度缩放平移，并计算变换矩阵
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # TODO 反投影：将深度图转换为相机坐标系下的3D点
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])  # inv_K为内参矩阵的逆
                # TODO 投影：将3D点投影到目标帧的图像平面，得到像素坐标
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)  # K为内参矩阵

                # 保存采样坐标
                outputs[("sample", frame_id, scale)] = pix_coords

                # 用网格采样生成重投影图像

                # TODO input：源图像（如代码中的
                # inputs[("color", frame_id, source_scale)]）；
                # grid：坐标网格，形状为(batch_size, height, width, 2)，每个元素(x, y)
                # 表示目标图像中对应位置的像素在源图像中的坐标（需归一化到[-1, 1]
                # 范围）

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],  # 源帧图像
                    outputs[("sample", frame_id, scale)],  # 采样坐标
                    padding_mode="border", align_corners=True)  # 边界填充模式

                # 如果不禁用自动掩码，保存源帧原始图像（用于掩码计算）
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """计算预测图像和目标图像之间的重投影损失
        """
        # 计算绝对差异
        abs_diff = torch.abs(target - pred)
        # 计算L1损失（通道维度平均）
        l1_loss = abs_diff.mean(1, True)

        # 如果禁用SSIM，重投影损失仅为L1损失
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            # 否则，重投影损失为SSIM损失（85%）+ L1损失（15%）
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """计算批次的重投影损失和平滑度损失
        """

        losses = {}
        total_loss = 0  # 总损失

        # 遍历所有尺度
        for scale in self.opt.scales:
            loss = 0  # 当前尺度的损失
            reprojection_losses = []  # 重投影损失列表

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0  # 源尺度为0

            # 获取当前尺度的视差图、参考帧图像、目标图像
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # 计算每个源帧的重投影损失
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]  # 重投影图像
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            # 拼接所有源帧的重投影损失
            reprojection_losses = torch.cat(reprojection_losses, 1)

            # 如果不禁用自动掩码
            if not self.opt.disable_automasking:
                # 计算身份重投影损失（直接使用源帧图像作为预测）
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                # 如果平均重投影损失，取均值
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # 否则保存所有损失，后续取最小值
                    identity_reprojection_loss = identity_reprojection_losses

            # 如果使用预测掩码
            elif self.opt.predictive_mask:
                # 获取预测的掩码
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    # 上采样掩码到原始尺寸
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                # 用掩码加权重投影损失
                reprojection_losses *= mask

                # 添加掩码的正则化损失（促使掩码接近1）
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            # 计算重投影损失（平均或保留原始维度）
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # TODO 组合身份损失和重投影损失（用于自动掩码）
            if not self.opt.disable_automasking:
                # 添加微小随机数打破平局
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            # 取最小损失（自动掩码逻辑：选择重投影或身份中较小的损失）
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            # 记录身份选择掩码（哪些像素使用了身份损失）|idxs > identity_reprojection_loss.shape[1] - 1： 即 “最小值是否来自重投影损失”；
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            # 累加重投影损失
            loss += to_optimise.mean()

            # 计算视差平滑度损失
            mean_disp = disp.mean(2, True).mean(3, True)  # 视差均值 [B,1,H,W]->[B,1,1,W]->[B,1,1,1]
            norm_disp = disp / (mean_disp + 1e-7)  # 归一化视差|消除不同批次 / 图像的全局视差大小差异（比如有的图像视差整体大，有的整体小），让平滑度损失只关注 “视差的相对变化”
            smooth_loss = get_smooth_loss(norm_disp, color)  # 基于图像梯度的平滑损失

            # 累加平滑度损失（按尺度衰减）
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            # 累加当前尺度损失到总损失
            total_loss += loss
            # 记录当前尺度的损失
            losses["loss/{}".format(scale)] = loss

        # 总损失取所有尺度的平均
        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """计算深度评估指标，用于训练过程中的监控

        注意：这不是精确的评估，因为它在整个批次上平均，仅用于验证性能指示
        """
        # 获取预测的深度图（尺度0）
        depth_pred = outputs[("depth", 0, 0)]
        # 上采样到KITTI原始尺寸（375x1242）并限制范围
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()  # detach不参与梯度计算

        # 获取深度真值
        depth_gt = inputs["depth_gt"]
        # 真值掩码（仅考虑真值有效的区域）
        mask = depth_gt > 0

        # 应用Garg/Eigen裁剪（KITTI评估标准区域）
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1  # 裁剪区域|裁剪区域：行153~371，列44~1197
        mask = mask * crop_mask  # 结合真值掩码和裁剪掩码

        # 提取掩码区域的真值和预测值
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        # 缩放预测深度，使中值与真值中值一致（消除尺度歧义）|模型只能学习到像素间的相对深度关系，无法直接预测绝对尺度（比如真实深度是 10 米，模型可能输出 5 米，所有深度都缩放了 2 倍
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        # 再次限制预测深度范围
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        # 计算深度误差指标
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        # 记录每个指标
        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
        # return depth_errors

    def log_time(self, batch_idx, duration, loss, loss_distill=None):
        """向终端打印日志信息
        """
        # 计算每秒处理的样本数
        samples_per_sec = self.opt.batch_size / duration
        # 计算已用时间
        time_sofar = time.time() - self.start_time
        # 估计剩余训练时间
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        # 日志格式字符串
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | loss/distill: {:.5f} | time elapsed: {} | time left: {}"
        # 打印日志（当前epoch、学习率、批次索引、样本率、损失、已用时间、剩余时间）
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss, loss_distill,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """向tensorboard事件文件写入日志
        """
        writer = self.writers[mode]  # 获取对应模式（train/val）的写入器
        # 记录损失值
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # 最多记录4个样本的图像
        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:  # 遍历所有尺度
                for frame_id in self.opt.frame_ids:  # 遍历所有帧
                    # 记录原始图像
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    # 记录尺度0的预测重投影图像（非参考帧）
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                # 记录视差图（归一化后）
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # 如果使用预测掩码，记录掩码
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                # 如果使用自动掩码，记录身份选择掩码
                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """将配置参数保存到磁盘，以便记录实验设置
        """
        models_dir = os.path.join(self.log_path, "models")
        # 创建目录（如果不存在）
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        # 复制配置参数字典
        to_save = self.opt.__dict__.copy()

        # 写入JSON文件
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def load_pretrain(self):
        # 处理预训练模型路径（展开用户目录）
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        # 获取当前编码器的状态字典
        model_dict = self.models["encoder"].state_dict()
        # 加载预训练权重
        pretrained_dict = torch.load(path)['model']
        # 过滤预训练权重：仅保留与当前模型匹配且不以'norm'开头的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        # 更新模型字典
        model_dict.update(pretrained_dict)
        # 加载更新后的状态字典
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """从磁盘加载模型权重
        """
        # 处理加载路径
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        # 断言路径存在
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # 加载指定的模型
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            # 如果是姿态相关模型
            if n in ['pose_encoder', 'pose']:
                model_dict = self.models_pose[n].state_dict()
                pretrained_dict = torch.load(path)
                # 过滤匹配的参数
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_pose[n].load_state_dict(model_dict)
            else:
                # 深度相关模型
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # 加载优化器状态
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
