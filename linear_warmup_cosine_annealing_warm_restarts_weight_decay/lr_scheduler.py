import math
import torch
from typing import Optional

from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models import AlexNet


class WarmUpScheduler(_LRScheduler):
    """
    Args:
        optimizer: [torch.optim.Optimizer] only pass if using as astand alone lr_scheduler
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        eta_min: float = 0.0,
        last_epoch=-1,
        max_lr: Optional[float] = 0.1,
        warmup_steps: Optional[int] = 0,
    ):

        if warmup_steps != 0:
            assert warmup_steps >= 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.step_in_cycle = last_epoch
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup

        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]

        else:
            return [base_lr + (self.max_lr - base_lr) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

        else:
            self.step_in_cycle = self.epoch

        self.max_lr = self.base_max_lr
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnealingWarmRestartsWeightDecay(_LRScheduler):
    """
       Helper class for chained scheduler not to used directly. this class is synchronised with
       previous stage i.e.  WarmUpScheduler (max_lr, T_0, T_cur etc) and is responsible for
       CosineAnealingWarmRestarts with weight decay
       """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mul: float = 1.,
        eta_min: float = 0.001,
        last_epoch=-1,
        max_lr: Optional[float] = 0.1,
        gamma: Optional[float] = 1.,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        super(CosineAnealingWarmRestartsWeightDecay, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mul

        # since warmup steps must be < T_0 and if epoch count > T_0 we just apply cycle count for weight decay
        if self.epoch >= self.T_0:
            if self.T_mul == 1.:
                self.T_cur = self.epoch % self.T_0
                self.cycle = self.epoch // self.T_0
            else:
                n = int(math.log((self.epoch / self.T_0 * (self.T_mul - 1) + 1), self.T_mul))
                self.cycle = n
                self.T_cur = self.epoch - int(self.T_0 * (self.T_mul**n - 1) / (self.T_mul - 1))
                self.T_i = self.T_0 * self.T_mul**(n)

        # base condition that applies original implementation for cosine cycles for details visit:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        else:
            self.T_i = self.T_0
            self.T_cur = self.epoch

        # this is where weight decay is applied
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ChainedScheduler(_LRScheduler):
    """
    Driver class
        Args:
        T_0: First cycle step size, Number of iterations for the first restart.
        T_mul: multiplicative factor Default: -1., A factor increases T_i after a restart
        eta_min: Min learning rate. Default: 0.001.
        max_lr: warmup's max learning rate. Default: 0.1. shared between both schedulers
        warmup_steps: Linear warmup step size. Number of iterations to complete the warmup
        gamma: Decrease rate of max learning rate by cycle. Default: 1.0 i.e. no decay
        last_epoch: The index of last epoch. Default: -1

    Usage:

        ChainedScheduler without initial warmup and weight decay:

            scheduler = ChainedScheduler(
                            optimizer,
                            T_0=20,
                            T_mul=2,
                            eta_min = 1e-5,
                            warmup_steps=0,
                            gamma = 1.0
                        )

        ChainedScheduler with weight decay only:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch=-1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 0,
                            gamma: Optional[float] = 0.9
                        )

        ChainedScheduler with initial warm up and weight decay:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch = -1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 10,
                            gamma: Optional[float] = 0.9
                        )
    Example:
        >>> model = AlexNet(num_classes=2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
        >>> scheduler = ChainedScheduler(
        >>>                 optimizer,
        >>>                 T_0 = 20,
        >>>                 T_mul = 1,
        >>>                 eta_min = 0.0,
        >>>                 gamma = 0.9,
        >>>                 max_lr = 1.0,
        >>>                 warmup_steps= 5 ,
        >>>             )
        >>> for epoch in range(100):
        >>>     optimizer.step()
        >>>     scheduler.step()

    Proper Usage:
        https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mul: float = 1.0,
        eta_min: float = 0.001,
        last_epoch=-1,
        max_lr: Optional[float] = 1.0,
        warmup_steps: Optional[int] = 5,
        gamma: Optional[float] = 0.95,
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        if warmup_steps != 0:
            assert warmup_steps < T_0
            # 默认warmup_epochs = 1
            warmup_steps = warmup_steps + 1  # directly refers to epoch account for 0 off set

        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        self.last_epoch = last_epoch

        self.cosine_scheduler1 = WarmUpScheduler(
            optimizer,
            eta_min=self.eta_min,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
        )
        self.cosine_scheduler2 = CosineAnealingWarmRestartsWeightDecay(
            optimizer,
            T_0=self.T_0,
            T_mul=self.T_mul,
            eta_min=self.eta_min,
            max_lr=self.max_lr,
            gamma=self.gamma,
        )

    def get_lr(self):
        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                return self.cosine_scheduler1.get_lr()
        if self.epoch >= self.warmup_steps:
            return self.cosine_scheduler2.get_lr()

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1

        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                self.cosine_scheduler1.step()
                self.last_epoch = self.epoch

        if self.epoch >= self.warmup_steps:
            self.cosine_scheduler2.step()
            self.last_epoch = self.epoch

if __name__ == '__main__':
    # lr =  [0.0001, 5.0e-6, 31, 0.0001, 1.0e-5, 31] # fine_tune
    # lr =  [5e-4, 5.0e-6, 35, 5e-4, 1.0e-5, 35]
    # lr = [5.0e-4, 5.0e-6, 38, 5.0e-4, 1.0e-5, 38]
    lr = [5.0e-4, 5.0e-6, 31, 5.0e-4, 1.0e-5, 31]
    enc_max_lr, enc_eta_min, enc_T0 = lr[0], lr[1], lr[2]
    dec_max_lr, dec_eta_min, dec_T0 = lr[3], lr[4], lr[5]

    # ===================== 1. 初始化模型 (模拟Lite-Mono的Encoder+Decoder分模块) =====================
    model = AlexNet(num_classes=2)
    # 分2个参数组：encoder参数 + decoder参数 (和Lite-Mono完全一致)
    param_groups = [
        {"params": model.features.parameters(), "lr": enc_max_lr},  # 对应Encoder
        {"params": model.classifier.parameters(), "lr": dec_max_lr}  # 对应Decoder
    ]
    # ===================== 2. 初始化优化器 =====================
    model_optimizer = optim.AdamW(param_groups, weight_decay=1e-2)  # weight_decay可按需修改
    # ===================== 3. 初始化调度器  =====================
    model_lr_scheduler = ChainedScheduler(
        model_optimizer,
        T_0=int(enc_T0),  # 初始余弦周期
        T_mul=1,  # 周期倍增因子，1=固定周期
        eta_min=enc_eta_min,  # 基础最小学习率
        last_epoch=-1,  # 初始epoch
        max_lr=enc_max_lr,  # 基础最大学习率
        warmup_steps=3,  # 0=关闭预热，>0=开启预热(如5)
        gamma=0.9  # 每个周期的max_lr衰减因子
    )
    # ===================== 4. 核心：遍历epoch收集学习率 =====================
    EPOCHS = enc_T0  # 绘制100个epoch的学习率曲线，可修改
    lr_encoder_list = []  # 收集编码器的学习率
    lr_decoder_list = []  # 收集解码器的学习率
    epoch_list = list(range(EPOCHS))
    for epoch in range(EPOCHS):
        # 记录当前的两个参数组的学习率
        lr_encoder = model_optimizer.param_groups[0]['lr']
        lr_decoder = model_optimizer.param_groups[1]['lr']
        lr_encoder_list.append(lr_encoder)
        lr_decoder_list.append(lr_decoder)

        # 学习率调度器步进
        model_lr_scheduler.step()
    # ===================== 5. 绘制学习率曲线 (美化版，一键出图) =====================
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(epoch_list, lr_encoder_list, color='#FF6B6B', linewidth=2.0,
             label=f'Encoder LR (max={enc_max_lr:.4f}, min={enc_eta_min:.6f}, T0={enc_T0})')
    # plt.plot(epoch_list, lr_decoder_list, color='#4ECDC4', linewidth=2.0, linestyle='--',
    #          label=f'Decoder LR (max={dec_max_lr:.4f}, min={dec_eta_min:.6f}, T0={dec_T0})')

    # 图表美化
    plt.title('Lite-Mono ChainedScheduler LR Curve (CosineAnnealingWarmRestarts + Gamma Decay)', fontsize=14,
              pad=20)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.xlim(0, EPOCHS)
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.legend(loc='upper right', fontsize=10)
    plt.tick_params(labelsize=10)

    # 可选：保存图片到本地
    # plt.savefig('lr_curve.png', bbox_inches='tight', facecolor='white')
    plt.savefig(f'LearningRate_Curve_max{enc_max_lr}_min{enc_eta_min}_{enc_T0}.png', bbox_inches='tight', dpi=300)
    # 显示图片
    plt.show()

    # ===================== 6. 打印关键信息 =====================
    print(f"训练周期数: {EPOCHS}")
    print(f"编码器初始LR: {enc_max_lr:.4f}, 最小LR: {enc_eta_min:.6f}, 余弦周期: {enc_T0}")
    print(f"解码器初始LR: {dec_max_lr:.4f}, 最小LR: {dec_eta_min:.6f}, 余弦周期: {dec_T0}")
    print(f"Gamma衰减因子: {0.9}, Warmup步数: {0}")









