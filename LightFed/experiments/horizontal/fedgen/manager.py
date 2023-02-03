import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from experiments.models.model import model_pull, generator_model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg, NumericAvgAgg
from lightfed.tools.funcs import consistent_hash, formula, save_pkl, set_seed, model_size, grad_False, grad_True
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param, DiversityLoss
from torch import nn
import torch.nn.functional as F

from trainer import ClientTrainer
from collections import OrderedDict
import time
from param_config import METRICS
MIN_SAMPLES_PER_LABEL=1

class ServerManager(BaseServer):
    # 现在的server_role的作用是控制实验的初始化和进程
    # 以及进行测试global模型的test
    # 因为decentralized method的目标还是训练一个全局模型后续的测试还是需要全局模型
    # 不再进行梯度或者是模型参数的aggregation
    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        del self.super_params['data_distributer']
        del self.super_params['log_level']
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num
        self.selected_client_num = args.selected_client_num
        self.comm_round = args.comm_round
        self.I = args.I
        self.diversity_loss_type = args.diversity_loss_type  ###是否在diversity loss中考虑类别信息


        self.eval_step_interval = args.eval_step_interval
        self.eval_on_full_test_data = args.eval_on_full_test_data

        ###训练生成器的相关参数
        self.I_gen = args.I_gen
        self.gen_batch_size = args.gen_batch_size
        self.lr_gen = args.lr_gen
        self.lambda_kl_s = args.lambda_kl_s
        self.lambda_cls = args.lambda_cls
        self.lambda_div = args.lambda_div
        self.temp_s = args.temp_s
        self.weight_decay = args.weight_decay

        self.full_train_dataloader = args.data_distributer.get_train_dataloader()  ##训练数据 计算train_loss
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()    ##测试数据 计算test_loss
        self.init_loss_fn()

        set_seed(args.seed + 657)

        self.model, self.global_classifier = model_pull(args, g_classifer=True)  # 用于全局模型性能的评估
        self.generator_model = generator_model_pull(args).to(self.device)
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        # if not os.path.exists(f"{path}/model_save/{args.model_type}.pth"):
        #     torch.save(self.model, f"{path}/model_save/{args.model_type}.pth")

        if not os.path.exists(f"{path}/model_save/{args.generator_model_type}_{args.data_set}.pth"):
            torch.save(self.generator_model, f"{path}/model_save/{args.generator_model_type}_{args.data_set}.pth")

        self.global_params = get_cpu_param(self.model.state_dict())
        self.global_classifier_params = get_cpu_param(self.global_classifier.state_dict())
        self.generator_params = get_cpu_param(self.generator_model.state_dict())
        self.latent_layer_idx = None
        self.init_optimizer()
        torch.cuda.empty_cache()

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]
        self.unique_labels = args.data_distributer.class_num
        self.qualified_labels = [i for i in range(self.unique_labels)]
        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}
        # self.global_label_counts = {label: 1 for label in range(self.unique_labels)}
        self.label_counts_collect = {}
        self.client_classifier_collect = {}

        self.global_params_aggregator = ModelStateAvgAgg()
        self.global_classifier_params_aggregator = ModelStateAvgAgg()

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.client_eval_info = []  ##不需要, 因此在运行过程中都为空
        self.global_train_eval_info = []  ##需要

        self.unfinished_client_num = -1

        self.label_num = []

        self.step = -1

    def start(self):
        logging.info("start...")
        self.next_step()

    def init_loss_fn(self):
        self.NLL = nn.NLLLoss(reduce=False).to(self.device)
        self.KL = nn.KLDivLoss().to(self.device)  # ,log_target=True)
        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.diversity_loss = DiversityLoss(metric='l2').to(self.device)  ##这里可以进行修改
        self.dist_loss = nn.MSELoss().to(self.device)

    def init_optimizer(self):
        self.gen_optimizer = torch.optim.Adam(params=self.generator_model.parameters(), lr=self.lr_gen, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
        # self.gen_optimizer = torch.optim.SGD(params=self.generator_model.parameters(), lr=self.lr_gen, weight_decay=self.weight_decay)
        self.gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.gen_optimizer, gamma=0.98)


    def end(self):
        logging.info("end...")

        self.super_params['device'] = self.super_params['device'].type

        ff = f"{self.app_name}-{consistent_hash(self.super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_train_eval_info': pd.DataFrame(self.global_train_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'comm_load':self.comm_load}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self.selected_clients = self._new_train_workload_arrage()  ##随机
        self.unfinished_client_num = self.selected_client_num
        self.global_params_aggregator.clear()

        if self.step > 0:
            self.global_classifier_params = get_cpu_param(self.global_classifier.state_dict())
            self.generator_params = get_cpu_param(self.generator_model.state_dict())
        for client_id in self.selected_clients:
            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(step=self.step, global_classifier_params=self.global_classifier_params, generator_params=self.generator_params,
                                       label_num=self.label_num,
                                       latent_layer_idx=self.latent_layer_idx)

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client

    # def update_label_counts(self, counter_dict):
    #     for label in counter_dict:
    #         self.global_label_counts[int(label)] += counter_dict[label]
    #
    # def clean_up_counts(self):
    #     del self.global_label_counts
    #     self.global_label_counts = {label:1 for label in range(self.unique_labels)}


    def update_label_counts_collect(self, client_id, counter_dict):
        self.label_counts_collect[client_id] = counter_dict

    def clean_up_counts_collect(self):
        del self.label_counts_collect
        self.label_counts_collect = {}

    def update_client_classifier_collect(self, client_id, local_classifier):
        grad_False(local_classifier)
        self.client_classifier_collect[client_id] = local_classifier

    def clean_up_client_classifier_collect(self):
        del self.client_classifier_collect
        self.client_classifier_collect = {}

    def get_label_weights(self):
        label_num = []
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.label_counts_collect:
                weights.append(self.label_counts_collect[user][label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_num.append(np.array(weights))
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        label_num = np.array(label_num).reshape((self.unique_labels, -1))
        return label_weights, label_num, qualified_labels

    def random_choice_y(self, batch_size, label_num):
        if len(label_num) > 0:
            _label_num = label_num.sum(axis=1)
            label_pop = _label_num / sum(_label_num)
            label_cumpop = np.cumsum(label_pop)
            label_cumpop = np.insert(label_cumpop, 0, 0.0)
            r_bz = np.random.random(batch_size)
            y = []
            for r in r_bz:
                for i in range(len(label_cumpop) - 1):
                    if r >= label_cumpop[i] and r <= label_cumpop[i + 1]:
                        y.append(i)
        else:
            y = np.random.choice(self.qualified_labels, batch_size)
        return y


    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     local_model,
                                     local_classifier,
                                     label_count,
                                     eval_info):
        logging.debug(f"train comm. round of client_id:{client_id} comm. round:{step} was finished")
        assert self.step == step
        self.client_eval_info.append(eval_info)

        weight = self.local_sample_numbers[client_id]
        self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)
        local_model_params = get_cpu_param(local_model.state_dict())
        local_classifier_params = get_cpu_param(local_classifier.state_dict())

        self.global_params_aggregator.put(local_model_params, weight)
        self.global_classifier_params_aggregator.put(local_classifier_params, weight)

        if self.comm_load[client_id] == 0:
            self.comm_load[client_id] = model_size(local_classifier_params) / 1024 / 1024  ##单位为MB

        self.update_label_counts_collect(client_id, label_count)
        self.update_client_classifier_collect(client_id, local_classifier)

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}
            ##获得全局模型，用于查看性能
            self.global_params = self.global_params_aggregator.get_and_clear()
            client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
            print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
            self.model.load_state_dict(self.global_params, strict=True)
            if self.I == 0:
                if self.step % self.eval_step_interval == 0:
                    self._set_global_eval_info()
            else:
                self._set_global_eval_info()

            ###training Generator
            self.global_classifier_params = self.global_classifier_params_aggregator.get_and_clear()
            self.global_classifier.load_state_dict(self.global_classifier_params, strict=True)

            self.label_weights, self.label_num, self.qualified_labels = self.get_label_weights()

            grad_False(self.global_classifier)
            self.timestamp = time.time()
            L_KL, L_CLS, L_DIV, LOSS = 0, 0, 0, 0
            for i in range(self.I_gen):
                self.generator_model.train()
                self.global_classifier.eval()

                self.generator_model.zero_grad(set_to_none=True)
                self.gen_optimizer.zero_grad(set_to_none=True)

                y = self.random_choice_y(self.gen_batch_size, self.label_num)
                y_input = torch.LongTensor(y)
                gen_result = self.generator_model(y_input)

                ##### get losses ####
                if self.diversity_loss_type == 'div0':
                    L_div = self.diversity_loss(noises=gen_result['eps'].to(self.device), layer=gen_result['output'])  # encourage different outputs
                elif self.diversity_loss_type == 'div1':
                    L_div = self.diversity_loss(noises=gen_result['eps_y'].to(self.device), layer=gen_result['output'])
                elif self.diversity_loss_type == 'div2':
                    L_div = self.diversity_loss(noises=gen_result['eps'].to(self.device), layer=gen_result['output'], y_input=gen_result['y_input'].to(self.device),
                                                diversity_loss_type='div2')

                ######### get teacher loss ############
                L_cls = 0
                L_kl_logit = 0
                for id_, user_idx in enumerate(self.client_classifier_collect):
                    self.cache_classifier = self.client_classifier_collect[user_idx]
                    self.cache_classifier.eval()
                    weight = self.label_weights[y][:, id_]#
                    gen_cc_result = self.cache_classifier(gen_result['output'], logit=True)

                    L_cls_ = torch.mean( \
                        self.NLL(gen_cc_result['output'], y_input.to(self.device)) * torch.tensor(weight, dtype=torch.float32).to(self.device))
                    L_cls += L_cls_
                    L_kl_logit += gen_cc_result['logit'] * torch.tensor(weight.reshape(-1, 1), dtype=torch.float32).to(self.device)

                ######### get student loss ############
                gen_gc_result = self.global_classifier(gen_result['output'], logit=True)
                gen_gc_softmax_temp = F.softmax(gen_gc_result['logit'] / self.temp_s, dim=1).clone().detach()
                L_kl = self.KL(F.log_softmax(L_kl_logit / self.temp_s, dim=1), gen_gc_softmax_temp)

                if self.lambda_kl_s > 0:
                    loss = -self.lambda_kl_s * L_kl + self.lambda_cls * L_cls  + self.lambda_div * L_div
                else:
                    loss = self.lambda_cls * L_cls  + self.lambda_div * L_div
                L_KL += L_kl
                L_CLS += L_cls
                L_DIV += L_div
                LOSS += loss

                loss.backward()
                self.gen_optimizer.step()
            curr_time = time.time()
            train_time = curr_time - self.timestamp

            L_KL = L_KL.detach().cpu().numpy() / self.I_gen
            L_CLS = L_CLS.detach().cpu().numpy() / self.I_gen
            L_DIV = L_DIV.detach().cpu().numpy() / self.I_gen
            LOSS = LOSS.detach().cpu().numpy() / self.I_gen
            info = "Generator: L_CLS= {:.4f}, L_KL= {:.4f}, L_DIV = {:.4f}".format(L_CLS, L_KL, L_DIV)
            # info_gen = {'comm. round': self.step, 'client_id': 'server',
            #             'L_CLS': round(L_CLS, 4), 'L_KL': round(L_KL, 4), 'L_DIV': round(L_DIV, 4),
            #             # 'train_time': round(train_time, 4)}
            self.server_train_test_res.update(L_KL=L_KL, L_CLS=L_CLS, L_DIV=L_DIV, LOSS=LOSS, train_time=train_time)
            # print(info)

            # self.gen_lr_scheduler.step()
            logging.debug(f"train comm. round:{step} is finished")
            self.global_train_eval_info.append(self.server_train_test_res)
            self.server_train_test_res = {}
            self.clean_up_counts_collect()
            self.clean_up_client_classifier_collect()
            self.next_step()


    def _set_global_eval_info(self):
        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.full_train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.global_params,
        #                             device=self.device,
        #                             eval_full_data=False)
        # eval_info.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.CE,
                                    device=self.device,
                                    eval_full_data=self.eval_on_full_test_data)
        torch.cuda.empty_cache()
        self.server_train_test_res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{self.server_train_test_res}")

class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I = args.I
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.generator_model_type = args.generator_model_type
        self.data_set = args.data_set

        self.trainer = ClientTrainer(args, self.client_id)
        self.step = 0

    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def classifier_to_model(self, global_classifier_params):
        if self.data_set in ['FMNIST', 'EMNIST', 'CIFAR-10']:
            model_name = self.trainer.model.named_parameter_layers
        classifier_name = list(global_classifier_params.keys())
        i, j = 0, 0
        for name, param in self.trainer.model.named_parameters():
            if self.data_set in ['FMNIST', 'EMNIST', 'CIFAR-10']:
                if 'fc' in model_name[i]:
                    param.data = global_classifier_params[classifier_name[j]].clone().detach()
                    j += 1
            else:
                if 'fc' in name:
                    param.data = global_classifier_params[classifier_name[j]].clone().detach()
                    j += 1
            i += 1
        self.trainer.model.to(self.device)

    def model_to_classifier(self):
        if self.data_set in ['FMNIST', 'EMNIST', 'CIFAR-10']:
            model_name = self.trainer.model.named_parameter_layers
        i, j = 0, 0
        class_layers = []
        for name, param in self.trainer.model.named_parameters():
            if self.data_set in ['FMNIST', 'EMNIST', 'CIFAR-10']:
                if 'fc' in model_name[i]:
                    class_layers.append(param.data.clone().detach())
                    j += 1
            else:
                if 'fc' in name:
                    class_layers.append(param.data.clone().detach())
                    j += 1
            i += 1
        k = 0
        for name, param in self.trainer.classifier.named_parameters():
            param.data = class_layers[k]
            k += 1
        self.trainer.classifier.to(self.device)

    def fed_client_train_step(self, step, global_classifier_params, generator_params, label_num, latent_layer_idx):
        self.step = step
        logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        # self.trainer.model = torch.load(f"{path}/model_save/{self.model_type}.pth")

        self.trainer.res = {'communication round': step, 'client_id': self.client_id}

        if self.step > 0:
            self.classifier_to_model(global_classifier_params)  ##将global classifier的参数替换model中对应的层

        self.trainer.generator_model = torch.load(f"{path}/model_save/{self.generator_model_type}_{self.data_set}.pth")
        self.trainer.generator_model.load_state_dict(generator_params, strict=True)

        self.timestamp = time.time()
        # 算法第9行:获取梯度
        self.trainer.train_locally_step(self.I, self.step, label_num, latent_layer_idx)
        curr_timestamp = time.time()
        train_time = curr_timestamp - self.timestamp

        self.model_to_classifier()
        self.finish_train_step(self.trainer.model, self.trainer.classifier, self.trainer.label_counts, train_time)
        self.trainer.clean_up_counts()
        self.trainer.clear()
        torch.cuda.empty_cache()

    def finish_train_step(self, model, classifier, label_counts, train_time):
        self.trainer.get_eval_info(self.step, train_time)
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model,
                                          classifier,
                                          label_counts,
                                          self.trainer.res)