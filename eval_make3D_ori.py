# make3d 数据集评估， 以Monodepth2为例
from tqdm import tqdm

from layers import disp_to_depth
import networks
import cv2
import os
import torch
import scipy.misc
from scipy import io
import numpy as np
from networks import BiDepthDecoder, BiDepthDecoder2, LiteMono, ReMono,\
    mpvit_small,VitDepthDecoder,DepthDecoder, ReMonov2


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


def main(model_name='lite_mono'):
    # load_weights_folder = './models/mono_resnet50_640x192'
    load_weights_folder = f'./model/{model_name}'
    main_path = f'E:\datasets\make3d_test'
    print(f'测试模型：{model_name}, 测试数据集：{main_path}')

    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    # 初始化网络
    if model_name in ['lite_mono', 'lite_mono_wp', 'lite_mono_GAMD']:
        encoder = LiteMono()
        depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    elif model_name in ['re_mono', 're_mono_w29_p', 're_mono_w14_wp']:
        encoder = ReMono()
        depth_decoder = BiDepthDecoder(encoder.num_ch_enc, scales=range(3))
    else:
        print(f'<UNKnown>{model_name}')
    # ===================== 核心修复1：加载encoder权重 + 双过滤（解决所有key匹配问题） =====================
    encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    model_dict = encoder.state_dict()
    # 过滤规则：1.保留模型本身有的key 2.过滤total_ops/total_params无效统计字段
    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in model_dict and not any(x in k for x in ["total_ops", "total_params"])})

    # ===================== 核心修复2：加载depth decoder权重 + 过滤无效字段 =====================
    depth_decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))
    depth_decoder.load_state_dict({k: v for k, v in depth_decoder_dict.items()
                                   if not any(x in k for x in ["total_ops", "total_params"])})

    # encoder.cuda()
    encoder.eval()
    # depth_decoder.cuda()
    depth_decoder.eval()

    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()
    test_filenames = map(lambda x: x[4:], test_filenames)

    depths_gt = []
    images = []
    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = int(1704 / 2)
    depth_new_height = 21
    for filename in test_filenames:
        mat = io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
        depths_gt.append(mat["Position3DGrid"][:, :, 3])

        image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
        image = image[int((2272 - color_new_height) / 2):int((2272 + color_new_height) / 2), :, :]
        images.append(image[:, :, ::-1])
    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = map(lambda x: x[int((55 - 21) / 2):int((55 + 21) / 2), :], depths_gt)

    depths_gt_cropped = list(depths_gt_cropped)
    errors = []
    with torch.no_grad():
        for i in tqdm(range(len(images))):
            input_color = images[i]
            input_color = cv2.resize(input_color / 255.0, (640, 192), interpolation=cv2.INTER_NEAREST)  # <----1
            input_color = torch.tensor(input_color, dtype=torch.float).permute(2, 0, 1)[None, :, :, :]
            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 0.1, 100)  # <---2
            pred_disp = pred_disp.squeeze().cpu().numpy()
            depth_gt = depths_gt_cropped[i]
            depth_pred = 1 / pred_disp
            depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(depth_gt > 0, depth_gt < 70)
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= np.median(depth_gt) / np.median(depth_pred)
            depth_pred[depth_pred > 70] = 70
            errors.append(compute_errors(depth_gt, depth_pred))
        mean_errors = np.mean(errors, 0)

    print(("{:>8} | " * 4).format("abs_rel", "sq_rel", "rmse", "rmse_log"))
    print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))

if __name__ == '__main__':
    # model_name = 'lite_mono'
    # model_name = 're_mono'
    # model_name = 're_mono_w29_p'
    model_name = 'lite_mono_GAMD'
    # model_name = 're_mono_w14_wp' # wp
    # model_name = 'lite_mono_wp'
    main(model_name)
