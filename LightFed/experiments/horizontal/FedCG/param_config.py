import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed
METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=100)

    parser.add_argument('--I', type=int, default=1, help='synchronization interval, 本地训练步数')

    parser.add_argument('--batch_size', type=int, default=64)  # 128 训练的batch

    parser.add_argument('--gen_batch_size', type=int, default=64)  # 128 训练的batch


    # 多少个通信轮次评估一次全局模型
    # 默认1表示每个通信轮次结束后都要评估模型
    parser.add_argument('--eval_step_interval', type=int, default=5)
    # 每个通信轮次完成后，评估全局模型时的全量数据的dataloader的批次大小
    # 可以适当大一点
    parser.add_argument('--eval_batch_size', type=int, default=256)

    # 每个通信轮次完成后，评估全局模型时是否使用全量的测试集（训练集的评估固定只取一个批次，批次大小为：eval_batch_size）
    # 如果为True，则全量评估测试集，否则只评估一个批次，批次的大小为：eval_batch_size
    parser.add_argument('--eval_on_full_test_data', type=lambda s: s == 'true', default=True)

    parser.add_argument('--weight_decay', type=float, default=0.0)

    ### 本地更新的相关参数
    ## 学习率
    parser.add_argument('--lr_lm', type=float, default=0.05, help='learning rate of local model')
    parser.add_argument('--lr_gen', type=float, default=0.0003, help='learning rate of generator')
    parser.add_argument('--lr_dis', type=float, default=0.0003, help='learning rate of generator')

    ## 权重
    parser.add_argument('--gamma', type=float, default=1.0, help='parameter for L_{kl,1}^i')

    ##
    parser.add_argument("--I_lm", type=int, default=20, help="inner iterations for augmenting update local model by generator")
    parser.add_argument("--I_gen_dis", type=int, default=1, help="inner iterations for augmenting update local model by global classifier")
    parser.add_argument("--temp_c", type=int, default=10, help="Distillation temperature")


    ##服务器端更新
    parser.add_argument('--lr_gen_gc', type=float, default=0.001, help='learning rate of local model')
    parser.add_argument("--I_gen_gc", type=int, default=20, help="inner iterations for augmenting update local model by generator")
    parser.add_argument("--temp_s", type=int, default=10, help="Distillation temperature")

    parser.add_argument("--distill_aggregation", type=bool, default=False, help="Use embedding layer in generator network")
    parser.add_argument('--distill_aggregation_type', type=str, default='fedcg', choices=['None', 'fedcg', 'proposed'])

    parser.add_argument("--embedding", type=bool, default=False, help="Use embedding layer in generator network")

    # 模型权重衰减系数，只对权重参数进行衰减，偏置参数不加L2正则
    parser.add_argument('--generator_model_type', type=str, default='generator', choices=['generator'])

    parser.add_argument('--model_type', type=str, default='Lenet', choices=['Lenet', 'ResNet_18', 'ResNet_20'])

    parser.add_argument('--data_set', type=str, default='FMNIST', choices=['MNIST', 'EMNIST', 'FMNIST', 'CIFAR-10', 'CIFAR-100'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet_balanced',
                        choices=['iid', 'non_iid_dirichlet_unbalanced', 'non_iid_dirichlet_balanced']) #'non_iid_dirichlet',

    parser.add_argument('--non_iid_alpha', type=float, default=0.1)  # 在进行non_iid_dirichlet数据划分时需要, 该值越大表明数据越均匀

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='FedCG')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()

    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    if os.path.exists(ff):
        print(f"output file existed, skip task")
        exit(0)

    args.data_distributer = _get_data_distributer(args)

    # args.weight_matrix = _get_weight_matrix(args)

    return args

def _get_data_distributer(args):
    set_seed(args.seed + 5363)
    # set_seed(args.seed + 5364)
    return DataDistributer(args)

# def _get_weight_matrix(args):
#     n = args.client_num
#     wm = np.zeros(shape=(n, n))
#     for i in range(n):
#         wm[i][(i - 1 + n) % n] = 1 / 3
#         wm[i][i] = 1 / 3
#         wm[i][(i + 1 + n) % n] = 1 / 3
#
#     assert np.allclose(wm, wm.T)
#     return wm
