import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull, generator_model_pull
from lightfed.tools.funcs import set_seed, grad_False, grad_True
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters, DiversityLoss
from torch import nn
import torch.nn.functional as F
from collections import Counter
import numpy as np



class ClientTrainer:
    def __init__(self, args, client_id):
        self.comm_round = args.comm_round
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.gen_batch_size = args.gen_batch_size
        self.weight_decay = args.weight_decay

        self.lr_lm = args.lr_lm
        self.lr_gen = args.lr_gen
        self.lr_dis = args.lr_dis

        self.gamma = args.gamma

        self.I_lm = args.I_lm
        self.I_gen_dis = args.I_gen_dis
        self.temp_c = args.temp_c

        self.unique_labels = args.data_distributer.class_num
        self.qualified_labels = [i for i in range(self.unique_labels)]
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)

        # for _, label in self.train_dataloader.dataset:
        #     self.label_counts[label] += 1

        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.init_loss_fn()

        self.cache_dataset = []
        self.cache_noise_dataset = []
        self.res = {}

        set_seed(args.seed + 657)

        self.model, self.classifier = model_pull(args, g_classifer=True)
        _, self.discriminator = model_pull(args, discriminator=True)
        self.generator_model = generator_model_pull(args).to(self.device)

        self.lm_optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm,
                                            weight_decay=self.weight_decay)
        # self.lm_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr_lm, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        # self.lr_lm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.lm_optimizer, gamma=0.98)

        self.gen_optimizer = torch.optim.Adam(params=self.generator_model.parameters(), lr=self.lr_gen,
                                              betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        self.dis_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr_dis,
                                              betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        # self.gen_optimizer = torch.optim.SGD(params=self.generator_model.parameters(), lr=self.lr_gen, weight_decay=self.weight_decay)
        # self.dis_optimizer = torch.optim.SGD(params=self.discriminator.parameters(), lr=self.lr_dis, weight_decay=self.weight_decay)

    def init_loss_fn(self):
        self.MSE = nn.MSELoss()
        self.KL = nn.KLDivLoss(reduction="batchmean") #
        self.CE = nn.CrossEntropyLoss()
        self.diversity_loss = DiversityLoss(metric='l2').to(self.device)  ##这里可以进行修改

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

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

    def _get_grad_(self):
        grad = OrderedDict()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for name, weight in self.model.named_parameters():
                _g = weight.grad.detach()
                if 'bias' not in name:  # 不对偏置项进行正则化
                    _g += (weight * self.weight_decay).detach()
                grad[name] = _g
        return grad

    def update_label_counts(self, counter_dict):
        for label in counter_dict:
            self.label_counts[int(label)] += counter_dict[label]

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def ploy_increase_lr_scheduler(self, epoch, inca=3, init_lr=0.1):
        lr = init_lr * (epoch/(self.comm_round-1))**inca
        return lr

    def y_select_in_dataset(self, y):
        for buff_data in self.cache_dataset:
            try:
                if (y - buff_data[1]).detach().cpu().numpy().sum() == 0:
                    return True
            except:
                return True
        return False

    def y_select_in_noise(self, y):
        for buff_data in self.cache_noise_dataset:
            if (y - buff_data[1]).sum() == 0:
                return True
        return False

    def clear(self):
        # self.generator_model = None
        # self.gen_optimizer = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step, label_num, latent_layer_idx):
        """算法的第5行
        """
        self.step = step
        self.label_num = label_num
        self.latent_layer_idx = latent_layer_idx
        # self.clean_up_counts()


        ##这个可以后续在调整
        # generative_alpha = self.exp_lr_scheduler(self.step, decay=0.98, init_lr=self.generative_alpha)
        # generative_beta = self.exp_lr_scheduler(self.step, decay=0.98, init_lr=self.generative_beta)

        self._gamma = self.ploy_increase_lr_scheduler(self.step, init_lr=self.gamma)

        ###sample y and generate z
        for tau in range(I):
            ##阶段1：利用生成器去增强本地训练
            self.update_local_model()
            for _, y in self.cache_dataset:
                eps = torch.rand((y.shape[0], self.generator_model.noise_dim))
                self.cache_noise_dataset.append((eps, y))
            self.update_generator_discriminator()
            self.cache_dataset, self.cache_noise_dataset = [], []

    def update_local_model(self, ):
        self.model.train()
        ##冻结模型，即在更新的时候不要计算其梯度，节省训练时间
        grad_False(self.generator_model)

        L_CE, L_MSE_F, LOSS = 0, 0, 0
        for _ in range(self.I_lm):
            self.model.zero_grad(set_to_none=True)
            self.lm_optimizer.zero_grad(set_to_none=True)

            ##batch数据
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            if not self.y_select_in_dataset(y):
                self.cache_dataset.append((x,y))
            model_result = self.model(x, logit=True, latent_feature=True)
            L_ce = self.CE(model_result['logit'], y)

            ###sample y and generate z
            if self.step:
                latent_feature = model_result['latent_features'][0]

                self.generator_model.eval()
                gen_result = self.generator_model(y.clone().detach().cpu())
                L_mse_f = self.MSE(latent_feature, gen_result['output'].clone().detach())

                loss = L_ce + self._gamma * L_mse_f

                L_CE += L_ce
                L_MSE_F = L_mse_f
                LOSS += loss
            else:
                loss = L_ce
                L_CE += L_ce
                LOSS += loss
            loss.backward()
            self.lm_optimizer.step()
        L_CE = L_CE.detach().cpu().numpy() / self.I_lm
        LOSS = LOSS.detach().cpu().numpy() / self.I_lm
        try:
            L_MSE_F = L_MSE_F.detach().cpu().numpy() / self.I_lm
        except:
            L_MSE_F = 0
        info = '\nclient_i={}, comm={}, L_CE={:.4f}, L_MSE_F={:.4f}'.format(
            self.client_id, self.step, L_CE, L_MSE_F)
        # info = '\nclient_i={}, comm={}, L_CE={:.4f}, L_CE_G={:.4f}, L_KL_F={:.4f}, L_MSE_F={:.4f}'.format(self.client_id, self.step, L_CE, L_CE_G, L_KL_F, L_MSE_F)
        # print(info)
        self.res.update(m_L_CE=L_CE, m_L_MSE_F=L_MSE_F, m_LOSS=LOSS)
        # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        self.model.zero_grad(set_to_none=True)
        self.lm_optimizer.zero_grad(set_to_none=True)
        grad_True(self.generator_model)

        for _, y in self.cache_dataset:
            counter_dict = dict(Counter(y.cpu().numpy()))
            self.update_label_counts(counter_dict)

    def update_generator_discriminator(self, ):
        self.model.eval()
        ##冻结模型，即在更新的时候不要计算其梯度，节省训练时间
        grad_False(self.model)

        L_CE_G, L_DISC, L_GEN = 0, 0, 0
        for _ in range(self.I_gen_dis):
            ##batch数据
            for data, noise in zip(self.cache_dataset, self.cache_noise_dataset):
                x, y = data[0], data[1]
                loss_disc = self.update_discriminator(x, y, noise)
                loss_gen, L_ce_g = self.update_generator(y, noise)

                L_DISC += loss_disc
                L_GEN += loss_gen
                L_CE_G += L_ce_g
        len_ = len(self.cache_dataset)
        L_DISC = L_DISC.detach().cpu().numpy() / self.I_gen_dis / len_
        L_GEN = L_GEN.detach().cpu().numpy() / self.I_gen_dis / len_
        L_CE_G = L_CE_G.detach().cpu().numpy() / self.I_gen_dis / len_

        # info = '\nGenerator: client_i={}, comm={}, L_CE_G={:.4f}, L_KL={:.4f}, L_MSE_G={:.4f}, L_DIV={:.4f}'.format(self.client_id, self.step, L_CE_G, L_KL, L_MSE_G, L_DIV)
        info = '\nGenerator: client_i={}, comm={}, L_DISC={:.4f}, L_GEN={:.4f}, L_CE_G={:.4f}'.format(
            self.client_id, self.step, L_DISC, L_GEN, L_CE_G)
        # print(info)
        self.res.update(dg_L_CE_G=L_CE_G, dg_L_GEN=L_GEN, dg_L_DISC=L_DISC)
        # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        self.generator_model.zero_grad(set_to_none=True)
        self.gen_optimizer.zero_grad(set_to_none=True)

        self.discriminator.zero_grad(set_to_none=True)
        self.dis_optimizer.zero_grad(set_to_none=True)
        grad_True(self.model)
        grad_True(self.generator_model)

    def update_discriminator(self, x, y, noise):
        self.discriminator.train()
        self.discriminator.zero_grad(set_to_none=True)
        self.dis_optimizer.zero_grad(set_to_none=True)
        grad_True(self.discriminator)
        grad_False(self.generator_model)

        model_result = self.model(x, logit=True, latent_feature=True)
        latent_feature = model_result['latent_features'][0].clone().detach()

        ##get generator output(latent representation) of the same label
        gen_result = self.generator_model(y.clone().detach().cpu(), eps=noise[0], eps_=True)

        model_dis_result = self.discriminator(latent_feature, logit=True)
        gen_dis_result = self.discriminator(gen_result['output'], logit=True)

        loss_disc = -torch.mean(torch.log(model_dis_result['logit']) + torch.log(1 - gen_dis_result['logit']))

        loss_disc.backward()
        self.dis_optimizer.step()
        grad_True(self.generator_model)
        return loss_disc

    def update_generator(self, y, noise):
        self.generator_model.train()
        self.generator_model.zero_grad(set_to_none=True)
        self.gen_optimizer.zero_grad(set_to_none=True)
        grad_True(self.generator_model)
        grad_False(self.discriminator)

        ##get generator output(latent representation) of the same label
        gen_result = self.generator_model(y.clone().detach().cpu(), eps=noise[0], eps_=True)
        gen_dis_result = self.discriminator(gen_result['output'], logit=True)

        gen_model_result = self.model(gen_result['output'], start_layer_idx=self.latent_layer_idx, logit=True)
        L_ce_g = self.CE(gen_model_result['logit'], y)

        loss_gen = -torch.mean(torch.log(gen_dis_result['logit']))

        loss_gen.backward()
        self.gen_optimizer.step()
        grad_True(self.discriminator)
        return loss_gen, L_ce_g


    def get_eval_info(self, step, train_time=None):
        self.res.update(train_time=train_time)

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.test_dataloader,
                                    criterion=self.CE,
                                    device=self.device)
        self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

