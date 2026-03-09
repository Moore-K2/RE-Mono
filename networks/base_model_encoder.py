import contextlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from networks.modules import *


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):
    # 生成随机深度概率（线性递增）
    dp_rates = [x.item() for x in torch.linspace(0, d['drop_path_rate'], sum(d['depth']))]
    print(f'\ndepth:{d["depth"]}, drop_path_rate:{d["drop_path_rate"]}')
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    drop_cur = 0

    gd, gw = d['depth_multiple'], d['width_multiple']
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        gd = 1.0
        n = n_ = max(round(n * gd), 1) if n > 1 else n # depth
        if m in(Conv, DWConv, DilatedConv, LGFI, C3P, GAM):
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in (DilatedConv, LGFI):
                assert c1 == c2, f'DilatedConv & LGFI {c1} != {c2}'
                args = args[1:]
                if m is DilatedConv:
                    args[4] = dp_rates[drop_cur]  # DilatedConv:dim, k, dilation=1, stride=1, drop_path
                    drop_cur += 1
                elif m is LGFI:
                    args[1] = dp_rates[drop_cur]  # LGFI:dim, drop_path
                    drop_cur += 1
        elif m is nn.Identity:
            c1, c2 = ch[f], ch[f]  # Identity层保持通道数不变
        elif m is AvgPool:
            c1, c2 = ch[f], ch[f]
        elif m is Concat:
            c2 =sum(ch[x] for x in f )
        else:
            c2 = ch[f]
        m_ = m(*args) # module

        t = str(m)[8:-2].replace('__main__.', '')  # m odule type
        num_of_params = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, num_of_params  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{num_of_params:10.0f}  {t:<40}{str(args):<30}')
        # TODO 只有当f不是-1时，才添加到save中
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def model_info(model, verbose=False, imgsz=[192, 640]):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try:  # FLOPs
        import thop
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = torch.zeros((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 192x640 GFLOPs
    except Exception:
        fs = ''

    name = Path(model.yaml_file).stem.replace('lite_mono', 'lite_mono') if hasattr(model, 'yaml_file') else 'Model'
    print(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

class BaseModel(nn.Module):
    def __init__(self, cfg="lite_mono.yaml", ch=3, height=192, width=640):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        self.num_ch_enc = np.array(self.yaml['num_ch_enc'])  # 编码器各阶段输出通道数
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.save_features_ls = self.yaml['features']
        print(f'\ntrain height:{height}, width:{width}, 编码器输出特征通道数:{self.num_ch_enc}')
        # TODO self.save存储所有不是-1的层索引
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist

        # Init weights, biases
        self.apply(self._init_weights)
        model_info(self) # cal params and flops

    def forward(self, x):
        """
        Forward pass that returns features from specified layers.
        Returns:
            dict: Dictionary containing features from layers [12, 18, 30]
                 keys: 'feat_11', 'feat_17', 'feat_29'
        """
        features = []  # 返回各阶段输出特征
        y = [] # 存储中间特征图
        x = (x - 0.45) / 0.225  # 输入标准化（均值0.45，标准差0.225）
        for i, m in enumerate(self.model):
            if m.f != -1: # 如果不是从前一层获取输入
                # [[-1, 2, 8], 1, Concat, [1]]为例子，让j==-1,取x=[x(即为前一层的输出)]
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # 只要是self.save中的层，就存储到y中，便于后期融合
            y.append(x if m.i in self.save else None)
            if m.i in self.save_features_ls:
                features.append(x)
        return features

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

Model = BaseModel

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # m = BaseModel('./models/lite_gam_mono.yaml').to(device)
    m = BaseModel('./models/lite_mono.yaml').to(device)
    # im = torch.rand(1, 3, 192, 640).to(device)
    # out = m(im)
    # print(out[0].shape)
    pass
