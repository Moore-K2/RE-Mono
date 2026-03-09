# from __future__ import absolute_import, division, print_function
#
# from options import LiteMonoOptions
# from trainer_dist import TrainerDist
#
# options = LiteMonoOptions()
# opts = options.parse()
from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from options_parse_yaml import LiteMonoOptionsYaml
from trainer_dist import TrainerDist

#TODO 解决windows多进程问题
import multiprocessing
multiprocessing.freeze_support()


# options = LiteMonoOptions()
# opts = options.parse()

# 用yaml配置文件训练
options = LiteMonoOptionsYaml()
opts = options.parse()


if __name__ == "__main__":
    # 在创建trainer之前修改参数
    # opts.cfg = "./models/re_gam_mono.yaml"
    # opts.model_name = "re_gam_mono"
    # print(opts)
    opts.data_path = r'E:\datasets\kitti_data'
    # opts.model_name = "ReMono_LGSA_bidecoder_ks_2_continue"
    opts.model_name = "ReMono_LGSA_bidecoder_ks_2_continue"
    opts.split = 'eigen_zhou'
    # opts.split = 'eigen_test'
    opts.batch_size = 12
    opts.num_epochs = 20
    opts.val_of_epochs = 3
    opts.cfg_first = False
    # opts.mypretrain = '/home/RM_luo/Documents/Lite-Mono/pretrained/lite-mono-pretrain.pth'
    opts.load_weights_folder = r'E:\AI_proj\Lite-Mono\tmp\ReMono_LGSA_bidecoder_ks_2\models\best_model'
    # opts.mypretrain = 'null'
    opts.depth_encoder = "ReMono"
    # opts.depth_decoder = "BiDepthDecoder2"
    opts.depth_decoder = "BiDepthDecoder"
    opts.use_kd = 'structure_kd' # # choose:[structure_kd, kd]c

    opts.lr=[0.0005, 5.0e-4, opts.num_epochs, 0.0005, 1e-6,  opts.num_epochs]

    trainer = TrainerDist(opts)

    trainer.train_v1()
