from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from options_parse_yaml import LiteMonoOptionsYaml
from trainer import Trainer

#TODO 解决windows多进程问题
import multiprocessing
multiprocessing.freeze_support()


# options = LiteMonoOptions()
# opts = options.parse()

# 用yaml配置文件训练
options = LiteMonoOptionsYaml()
opts = options.parse()


def main():
    # 在创建trainer之前修改参数
    # opts.cfg = "./models/re_gam_mono.yaml"
    # opts.model_name = "re_gam_mono"
    # print(opts)
    opts.data_path = r'E:\datasets\kitti_data'
    opts.split = 'eigen_test'
    # opts.split = 'eigen_zhou_val'
    opts.batch_size = 4

    trainer = Trainer(opts)
    # trainer.train()
    trainer.train_v1()


if __name__ == "__main__":
    main()
