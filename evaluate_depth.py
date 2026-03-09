# 1. 导入Python2兼容Python3的核心特性，避免语法冲突
# absolute_import：强制绝对导入，避免与本地模块重名；division：强制真除法；print_function：强制print为函数形式
from __future__ import absolute_import, division, print_function
# 2. 导入os模块，用于文件路径拼接、目录查询等系统文件操作
import os
# 3. 导入OpenCV库，用于图像尺寸调整、翻转等计算机视觉处理
import cv2
# 4. 导入NumPy库，用于高效数组运算、矩阵操作和数据存储
import numpy as np
# 5. 导入PyTorch核心库，用于张量运算、模型构建与加载
import torch
# 6. 从PyTorch数据工具中导入DataLoader，用于批量加载数据集
from torch.utils.data import DataLoader
# 7. 从自定义layers模块导入disp_to_depth函数，用于视差图转换为深度图
from layers import disp_to_depth
# 8. 从自定义utils模块导入readlines函数，用于读取文本文件中的每行内容（加载文件名列表）
from utils import readlines
# 9. 从自定义options模块导入LiteMonoOptions，用于解析模型评估的配置参数
from options import LiteMonoOptions
# 10. 导入自定义datasets模块，用于加载KITTI数据集
import datasets
# 11. 导入自定义networks模块，用于构建LiteMono编码器和深度解码器
import networks
# 12. 导入time模块，用于计时（统计模型推理速度）
import time
# 13. 从thop库导入clever_format，用于格式化计算量（FLOPs）和参数量（Params）的显示格式
from thop import clever_format
# 14. 从thop库导入profile，用于统计模型的计算量和参数量
from thop import profile
from networks.base_model_encoder import Model

# 15. 设置OpenCV的线程数为0，关闭多线程加速
# 作用：在Unix系统（OpenCV 3.3.1）上可将评估速度提升5倍，避免多线程与DataLoader线程冲突
cv2.setNumThreads(0)

# 16. 拼接splits目录的路径：当前脚本所在目录 + "splits"
# splits目录用于存储数据集划分文件（test_files.txt）和真实深度图（gt_depths.npz）
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


# 17. 定义profile_once函数，用于一次性统计编码器+解码器的计算量和参数量
# 输入：encoder（编码器模型）、decoder（解码器模型）、x（输入张量）
def profile_once(encoder, decoder, x):
    # 18. 提取输入张量的第0个样本，并增加一个批次维度（变为[1, C, H, W]），用于单独统计编码器
    x_e = x[0, :, :, :].unsqueeze(0)
    # 19. 将处理后的输入传入编码器，得到编码器输出（为统计解码器做准备）
    x_d = encoder(x_e)
    # 20. 统计编码器的计算量（flops_e）和参数量（params_e），verbose=False关闭详细日志输出
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    # 21. 统计解码器的计算量（flops_d）和参数量（params_d）
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    # 22. 格式化编码器+解码器的总计算量和总参数量，保留3位小数
    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    # 23. 格式化编码器的计算量和参数量，保留3位小数
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    # 24. 格式化解码器的计算量和参数量，保留3位小数
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    # 25. 返回格式化后的总/编码器/解码器的计算量和参数量
    return flops, params, flops_e, params_e, flops_d, params_d


# 26. 定义time_sync函数，用于获取精确的同步时间（适配GPU推理计时）
def time_sync():
    # 27. 如果CUDA可用（GPU环境），先执行CUDA同步操作
    # 作用：等待GPU上所有张量运算完成后再计时，避免异步运算导致的计时误差
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 28. 返回当前系统时间（秒级）
    return time.time()


# 29. 定义compute_errors函数，用于计算预测深度图与真实深度图之间的各项误差指标
# 输入：gt（真实深度图，numpy数组）、pred（预测深度图，numpy数组）
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    # 30. 计算阈值因子：取（真实深度/预测深度）和（预测深度/真实深度）中的较大值
    # 作用：用于评估预测深度与真实深度的偏差比例
    thresh = np.maximum((gt / pred), (pred / gt))
    # 31. 计算a1指标：阈值小于1.25的样本占比（偏差在25%以内）
    a1 = (thresh < 1.25     ).mean()
    # 32. 计算a2指标：阈值小于1.25²的样本占比（偏差在56.25%以内）
    a2 = (thresh < 1.25 ** 2).mean()
    # 33. 计算a3指标：阈值小于1.25³的样本占比（偏差在95.31%以内）
    a3 = (thresh < 1.25 ** 3).mean()

    # 34. 计算RMSE（均方根误差）：先计算差值的平方，再求均值，最后开平方根
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # 35. 计算RMSE_log（对数域均方根误差）：先取对数再计算差值平方，求均值后开平方根
    # 作用：对深度值的高低差异不敏感，更公平评估整体误差
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # 36. 计算abs_rel（绝对相对误差）：绝对误差与真实深度的比值的均值
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    # 37. 计算sq_rel（平方相对误差）：误差平方与真实深度的比值的均值
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    # 38. 返回所有误差指标
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# 39. 定义batch_post_process_disparity函数，用于对视差图进行后处理（来自Monodepthv1）
# 输入：l_disp（左视差图）、r_disp（右视差图，由原图水平翻转后预测得到）
def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1"""
    # 40. 获取视差图的批次大小（此处批次为1）、高度、宽度
    _, h, w = l_disp.shape
    # 41. 计算平均视差图：左视差图与右视差图的平均值
    m_disp = 0.5 * (l_disp + r_disp)
    # 42. 生成网格坐标：l为宽度方向的归一化坐标（0到1），高度方向同理
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # 43. 生成左侧掩码：对图像左侧区域进行加权（边缘区域侧重左视差图）
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    # 44. 生成右侧掩码：将左侧掩码水平翻转（边缘区域侧重右视差图）
    r_mask = l_mask[:, :, ::-1]
    # 45. 融合视差图：右侧掩码加权左视差图 + 左侧掩码加权右视差图 + 中间区域加权平均视差图
    # 作用：提升视差图的平滑性和准确性，减少边缘误差
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# 46. 定义evaluate函数，核心评估函数：加载模型、预测深度、计算误差
# 输入：opt（解析后的配置参数）
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    # 47. 定义深度最小值和最大值，用于裁剪异常深度值
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # 48. 判断是否使用外部预计算的视差图：若为None，则自行加载模型预测
    if opt.ext_disp_to_eval is None:

        # 49. 展开模型权重文件夹的路径（支持~表示用户主目录）
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        # 50. 断言：检查权重文件夹是否存在，若不存在则抛出异常
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        # 51. 打印日志：提示正在加载的权重文件夹路径
        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # 52. 读取测试文件名列表：拼接splits目录、评估数据集划分、test_files.txt文件
        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        # 53. 拼接编码器权重路径：权重文件夹 + encoder.pth
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        # 54. 拼接解码器权重路径：权重文件夹 + depth.pth
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        # 55. 加载编码器权重文件（字典格式，包含模型参数和输入尺寸等信息）
        encoder_dict = torch.load(encoder_path)
        # 56. 加载解码器权重文件
        decoder_dict = torch.load(decoder_path)

        # 57. 构建KITTI原始数据集实例
        # 参数说明：数据路径、文件名列表、输入高度、输入宽度、帧索引、缩放因子、非训练模式
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        # 58. 构建DataLoader：批量加载数据集
        # 参数说明：数据集、批次大小16、不打乱、工作线程数、锁页内存、不丢弃最后一个不完整批次
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # 59. 构建LiteMono编码器模型：传入模型类型、输入高度、输入宽度
        if opt.cfg_first:
            # TOdO 使用自定义的编码器
            encoder = Model(opt.cfg, height=encoder_dict['height'], width=encoder_dict['width'])
        else:
            encoder = networks.LiteMono(model=opt.model,
                                        height=encoder_dict['height'],
                                        width=encoder_dict['width'])


        # 60. 构建深度解码器模型：传入编码器输出通道数、预测尺度范围
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        # 61. 获取编码器的默认状态字典（模型结构参数）
        model_dict = encoder.state_dict()
        # 62. 获取解码器的默认状态字典
        depth_model_dict = depth_decoder.state_dict()
        # 63. 加载编码器权重：只加载与模型结构匹配的参数（避免键名不匹配报错）
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # 64. 加载解码器权重：只加载与模型结构匹配的参数
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        # 65. 将编码器移至GPU（CUDA）设备
        encoder.cuda()
        # 66. 将编码器设置为评估模式：关闭Dropout、BatchNorm等训练相关层
        encoder.eval()
        # 67. 将解码器移至GPU设备
        depth_decoder.cuda()
        # 68. 将解码器设置为评估模式
        depth_decoder.eval()

        # 69. 初始化空列表，用于存储所有预测视差图
        pred_disps = []

        # 70. 打印日志：提示当前预测的图像尺寸
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        # 71. 开启无梯度上下文：禁用自动求导，节省内存并提升推理速度
        with torch.no_grad():
            # 72. 遍历DataLoader中的每个批次数据
            for data in dataloader:
                # 73. 获取输入彩色图像：键为("color", 0, 0)表示第0帧、原始视角，移至GPU
                input_color = data[("color", 0, 0)].cuda()

                # 74. 判断是否启用后处理：若启用，则需要对每张图像做两次前向传播（原图+水平翻转图）
                if opt.post_process:
                    # 75. 拼接原图和水平翻转图：在批次维度拼接（总批次变为2*原批次）
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # 76. 统计当前批次的模型计算量和参数量
                flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                # 77. 记录推理开始时间（同步GPU后）
                t1 = time_sync()
                # 78. 模型前向传播：编码器输出传入解码器，得到预测结果
                output = depth_decoder(encoder(input_color))
                # 79. 记录推理结束时间（同步GPU后）
                t2 = time_sync()

                # 80. 将解码器输出的视差图转换为深度图：提取尺度0的视差图，传入最小/最大深度
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                # 81. 将预测视差图移至CPU，并转换为numpy数组：提取第0个通道（深度通道）
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                # 82. 判断是否启用后处理：若启用，则对预测视差图进行融合处理
                if opt.post_process:
                    # 83. 计算原批次大小：总批次//2
                    N = pred_disp.shape[0] // 2
                    # 84. 对视差图进行后处理：传入原图预测结果和翻转图预测结果（翻转回原方向）
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                # 85. 将当前批次的预测视差图添加到列表中
                pred_disps.append(pred_disp)

        # 86. 将所有批次的预测视差图拼接为一个大的numpy数组（形状：[总样本数, H, W]）
        pred_disps = np.concatenate(pred_disps)

    # 87. 若指定了外部视差图路径，则直接加载预计算的视差图
    else:
        # 88. 打印日志：提示正在加载外部视差图
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        # 89. 加载外部视差图文件（.npy格式）
        pred_disps = np.load(opt.ext_disp_to_eval)

        # 90. 判断是否需要将eigen划分转换为benchmark划分：若需要，则加载对应索引并筛选
        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    # 91. 判断是否保存预测视差图：若需要，则拼接保存路径并保存
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        # 92. 打印日志：提示保存路径
        print("-> Saving predicted disparities to ", output_path)
        # 93. 保存视差图为.npy格式
        np.save(output_path, pred_disps)

    # 94. 判断是否禁用评估：若禁用，则打印日志并退出程序
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # 95. 拼接真实深度图路径：splits目录 + 评估划分 + gt_depths.npz
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # 96. 加载真实深度图：支持兼容旧格式，编码格式为latin1，允许加载pickle对象
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    # 97. 打印日志：提示开始评估
    print("-> Evaluating")
    # 98. 打印日志：提示使用单目评估（采用中位数缩放策略）
    print("   Mono evaluation - using median scaling")

    # 99. 初始化空列表：用于存储每个样本的误差指标
    errors = []
    # 100. 初始化空列表：用于存储每个样本的缩放比例
    ratios = []

    # 101. 遍历所有预测样本（按样本索引遍历）
    for i in range(pred_disps.shape[0]):
        # 102. 获取当前样本的真实深度图
        gt_depth = gt_depths[i]
        # 103. 获取真实深度图的高度和宽度
        gt_height, gt_width = gt_depth.shape[:2]

        # 104. 获取当前样本的预测视差图
        pred_disp = pred_disps[i]
        # 105. 将预测视差图调整为真实深度图的尺寸（使用OpenCV的resize函数）
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # 106. 视差图转换为深度图：深度 = 1 / 视差（单目深度估计的基本转换关系）
        pred_depth = 1 / pred_disp

        # 107. 判断评估划分是否为eigen：eigen划分需要裁剪图像区域（去除无效区域）
        if opt.eval_split == "eigen":
            # 108. 生成掩码：筛选真实深度在[MIN_DEPTH, MAX_DEPTH]范围内的有效像素
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            # 109. 定义裁剪区域：基于图像尺寸的归一化坐标，转换为整数像素索引
            # 裁剪规则：去除顶部~40.8%和底部~0.8%的区域，左右两侧各去除~3.6%的区域
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            # 110. 生成裁剪掩码：初始化全0数组，在裁剪区域内设置为1
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            # 111. 合并掩码：有效深度掩码 + 裁剪区域掩码（只保留裁剪区域内的有效深度像素）
            mask = np.logical_and(mask, crop_mask)

        # 112. 非eigen划分：掩码为真实深度大于0的像素（有效深度像素）
        else:
            mask = gt_depth > 0

        # 113. 应用掩码：提取预测深度图中的有效像素
        pred_depth = pred_depth[mask]
        # 114. 应用掩码：提取真实深度图中的有效像素
        gt_depth = gt_depth[mask]

        # 115. 对预测深度图进行缩放：乘以配置的深度缩放因子
        pred_depth *= opt.pred_depth_scale_factor
        # 116. 判断是否禁用中位数缩放：若不禁用，则计算并应用中位数缩放
        if not opt.disable_median_scaling:
            # 117. 计算缩放比例：真实深度中位数 / 预测深度中位数
            # 作用：消除单目深度估计的尺度歧义（单目模型无法预测绝对尺度，只能预测相对尺度）
            ratio = np.median(gt_depth) / np.median(pred_depth)
            # 118. 将缩放比例添加到列表中
            ratios.append(ratio)
            # 119. 应用缩放比例：调整预测深度图的尺度
            pred_depth *= ratio

        # 120. 裁剪预测深度图：将小于MIN_DEPTH的像素设为MIN_DEPTH
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        # 121. 裁剪预测深度图：将大于MAX_DEPTH的像素设为MAX_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        # 122. 计算当前样本的误差指标，并添加到误差列表中
        errors.append(compute_errors(gt_depth, pred_depth))

    # 123. 若不禁用中位数缩放，则统计缩放比例的中位数和标准差
    if not opt.disable_median_scaling:
        # 124. 将缩放比例列表转换为numpy数组
        ratios = np.array(ratios)
        # 125. 计算缩放比例的中位数
        med = np.median(ratios)
        # 126. 打印日志：输出缩放比例的中位数和标准差
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    # 127. 计算所有样本的平均误差：对误差列表按列求均值（每个指标的平均值）
    mean_errors = np.array(errors).mean(0)

    # 128. 打印日志：输出误差指标的表头（格式化对齐）
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    # 129. 打印日志：输出平均误差值（格式化对齐，保留3位小数，适配LaTex表格格式）
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    # 130. 打印日志：输出模型的计算量和参数量（总/编码器/解码器）
    print("\n  " + ("flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    # 131. 打印日志：提示评估完成
    print("\n-> Done!")


# 132. 程序入口：当脚本直接运行时执行（若被导入则不执行）
if __name__ == "__main__":
    # 133. 创建LiteMonoOptions实例，用于解析命令行参数
    options = LiteMonoOptions()
    # 134. 解析命令行参数，并传入evaluate函数开始评估
    opt = options.parse()
    opt.data_path = r'E:\Moore\datasets\kitti_data'
    opt.cfg_first = True
    opt.load_weights_folder = r'E:\AI_proj\Lite-Mono\tmp\liteMono_no_pretrain_cfg\models\best_model'

    evaluate(opt)