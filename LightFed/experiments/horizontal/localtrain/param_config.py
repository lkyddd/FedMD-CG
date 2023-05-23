import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=200)

    parser.add_argument('--I', type=int, default=20, help='synchronization interval')

    parser.add_argument('--batch_size', type=int, default=64)  # 128 

    parser.add_argument('--eval_step_interval', type=int, default=5)


    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--eval_on_full_test_data', type=lambda s: s == 'true', default=True)

    parser.add_argument('--lr_lm', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='Lenet', choices=['Lenet', 'ResNet_18', 'ResNet_20'])

    parser.add_argument('--data_set', type=str, default='FMNIST',
                        choices=['MNIST', 'FMNIST', 'EMNIST', 'CIFAR-10', 'CIFAR-100'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet_balanced',
                        choices=['iid', 'non_iid_dirichlet_unbalanced', 'non_iid_dirichlet_balanced'])  # 'non_iid_dirichlet',

    parser.add_argument('--non_iid_alpha', type=float, default=0.1)  # 

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='Fedlocal')

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
