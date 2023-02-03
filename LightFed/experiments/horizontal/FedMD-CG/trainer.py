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
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.gen_batch_size = args.gen_batch_size
        self.weight_decay = args.weight_decay
        self.comm_round = args.comm_round

        self.lr_lm = args.lr_lm
        self.lr_gg = args.lr_gg

        self.lambda_kl_F = args.lambda_kl_F
        self.lambda_ce_G = args.lambda_ce_G
        self.lambda_mse_F = args.lambda_mse_F

        self.lambda_ce_G_g = args.lambda_ce_G_g
        self.lambda_kl = args.lambda_kl
        self.lambda_mse_G = args.lambda_mse_G
        self.lambda_div = args.lambda_div

        self.I_lm = args.I_lm
        self.I_gg = args.I_gg
        self.temp_c = args.temp_c
        self.diversity_loss_type = args.diversity_loss_type

        self.unique_labels = args.data_distributer.class_num
        self.qualified_labels = [i for i in range(self.unique_labels)]
        self.label_counts = {label: 0 for label in range(self.unique_labels)}

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
        self.generator_model = generator_model_pull(args).to(self.device)
        self.EMA_generator_model = generator_model_pull(args).to(self.device)

        self.lm_optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        self.gen_optimizer = torch.optim.Adam(params=self.generator_model.parameters(), lr=self.lr_gg, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        # self.lr_lm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.lm_optimizer, gamma=0.98)

        # self.gen_optimizer = torch.optim.SGD(params=self.generator_model.parameters(), lr=self.lr_gg, weight_decay=self.weight_decay)

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
        # self.lm_optimizer = None
        # self.gen_optimizer = None
        self.res = {}
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step, label_num, latent_layer_idx):
        """算法的第5行
        """
        self.step = step
        self.label_num = label_num
        self.latent_layer_idx = latent_layer_idx

        self._lambda_kl_F = self.ploy_increase_lr_scheduler(self.step, init_lr=self.lambda_kl_F)
        self._lambda_ce_G = self.ploy_increase_lr_scheduler(self.step, init_lr=self.lambda_ce_G)
        self._lambda_mse_F = self.ploy_increase_lr_scheduler(self.step, init_lr=self.lambda_mse_F)
        if self.client_id == 0:
            print('lambda_kl_F:{:.4f}, lambda_mse_F:{:.4f}'.format(self._lambda_kl_F, self._lambda_mse_F))

        # self.clean_up_counts()
        ##这个可以后续在调整
        # generative_alpha = self.exp_lr_scheduler(self.step, decay=0.98, init_lr=self.generative_alpha)
        # generative_beta = self.exp_lr_scheduler(self.step, decay=0.98, init_lr=self.generative_beta)

        ###sample y and generate z
        for tau in range(I):
            ##阶段1：利用生成器去增强本地训练
            self.update_local_model()
            for _, y in self.cache_dataset:
                eps = torch.rand((y.shape[0], self.generator_model.noise_dim))
                self.cache_noise_dataset.append((eps, y))
            self.update_generator()
            self.cache_noise_dataset, self.cache_noise_dataset = [], []

    def update_local_model(self, ):
        self.model.train()
        ##冻结模型，即在更新的时候不要计算其梯度，节省训练时间
        grad_False(self.EMA_generator_model)
        L_CE, L_CE_G, L_KL_F, L_MSE_F, LOSS = 0, 0, 0, 0, 0

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
                model_logsoftmax_temp = F.log_softmax(model_result['logit'] / self.temp_c, dim=1)
                latent_feature = model_result['latent_features'][0]

                self.EMA_generator_model.eval()
                gen_result = self.EMA_generator_model(y.clone().detach().cpu())

                L_mse_f = self.MSE(latent_feature, gen_result['output'].clone().detach())

                gen_model_result = self.model(gen_result['output'], start_layer_idx=self.latent_layer_idx, logit=True)
                gen_model_softmax_temp = F.softmax(gen_model_result['logit'] / self.temp_c, dim=1).clone().detach()
                L_kl_f = self.KL(model_logsoftmax_temp, gen_model_softmax_temp)

                sam_y = self.random_choice_y(self.gen_batch_size, self.label_num)
                sam_y_input = torch.LongTensor(sam_y)
                gen_result_sam_y = self.EMA_generator_model(sam_y_input)
                gen_model_result_sam_y = self.model(gen_result_sam_y['output'], start_layer_idx=self.latent_layer_idx, logit=True)
                L_ce_g = self.CE(gen_model_result_sam_y['logit'], sam_y_input.to(self.device))

                loss = L_ce + self._lambda_kl_F * L_kl_f + self._lambda_ce_G * L_ce_g + self._lambda_mse_F * L_mse_f #
                # loss = L_ce + self.lambda_mse_F * L_mse_f #

                L_CE += L_ce
                L_CE_G += L_ce_g
                L_KL_F += L_kl_f
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
            L_CE_G = L_CE_G.detach().cpu().numpy() / self.I_lm
            L_KL_F = L_KL_F.detach().cpu().numpy() / self.I_lm
            L_MSE_F = L_MSE_F.detach().cpu().numpy() / self.I_lm
        except:
            L_CE_G = 0
            L_KL_F = 0
            L_MSE_F = 0
        info = '\nlocal_model: client_i={}, comm={}, L_CE={:.4f}, L_CE_G={:.4f}, L_KL_F={:.4f}, L_MSE_F={:.4f}'.format(self.client_id, self.step, L_CE, L_CE_G, L_KL_F, L_MSE_F)
        # print(info)
        self.res.update(m_L_CE=L_CE, m_L_CE_G=L_CE_G, m_L_KL_F=L_KL_F, m_L_MSE_F=L_MSE_F, m_LOSS=LOSS)
        # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        self.model.zero_grad(set_to_none=True)
        self.lm_optimizer.zero_grad(set_to_none=True)

        for _, y in self.cache_dataset:
            counter_dict = dict(Counter(y.cpu().numpy()))
            self.update_label_counts(counter_dict)


    def update_generator(self, ):
        self.generator_model.train()
        self.model.eval()
        ##冻结模型，即在更新的时候不要计算其梯度，节省训练时间
        grad_True(self.generator_model)
        grad_False(self.model)
        L_CE_G, L_KL, L_MSE_G, L_DIV, LOSS = 0, 0, 0, 0, 0
        for _ in range(self.I_gg):
            self.generator_model.zero_grad(set_to_none=True)
            self.gen_optimizer.zero_grad(set_to_none=True)
            ##batch数据
            for data, noise in zip(self.cache_dataset, self.cache_noise_dataset):
                x, y = data[0], data[1]

                model_result = self.model(x, logit=True, latent_feature=True)
                model_softmax = F.softmax(model_result['logit'], dim=1).clone().detach()
                latent_feature = model_result['latent_features'][0].clone().detach()

                ##get generator output(latent representation) of the same label
                gen_result = self.generator_model(y.clone().detach().cpu(), eps=noise[0], eps_=True)

                if self.diversity_loss_type == 'div0':
                    L_div = self.diversity_loss(noises=gen_result['eps'].to(self.device), layer=gen_result['output'])  # encourage different outputs
                elif self.diversity_loss_type == 'div1':
                    L_div = self.diversity_loss(noises=gen_result['eps_y'].to(self.device), layer=gen_result['output'])
                elif self.diversity_loss_type == 'div2':
                    L_div = self.diversity_loss(noises=gen_result['eps'].to(self.device), layer=gen_result['output'], y_input=gen_result['y_input'].to(self.device), diversity_loss_type='div2')
                elif self.diversity_loss_type == 'None':
                    L_div = self.diversity_loss(noises=gen_result['eps'].to(self.device), layer=gen_result['output'])

                gen_model_result = self.model(gen_result['output'], start_layer_idx=self.latent_layer_idx, logit=True, latent_feature=True)
                gen_model_logsoftmax = F.log_softmax(gen_model_result['logit'], dim=1)

                L_ce_g = self.CE(gen_model_result['logit'], y)
                L_kl =  self.KL(gen_model_logsoftmax, model_softmax)  # p*log(q)
                L_mse_g = self.MSE(gen_result['output'], latent_feature)
                if self.diversity_loss_type == 'None':
                    loss = self.lambda_ce_G_g * L_ce_g + self.lambda_kl * L_kl + self.lambda_mse_G * L_mse_g
                else:
                    loss = self.lambda_ce_G_g * L_ce_g + self.lambda_kl * L_kl + self.lambda_mse_G * L_mse_g + self.lambda_div * L_div
                # loss = self.lambda_kl * L_kl + self.lambda_mse_G * L_mse_g + self.lambda_div * L_div

                L_CE_G += L_ce_g
                L_KL += L_kl
                L_MSE_G += L_mse_g
                L_DIV += L_div
                LOSS += loss

                loss.backward()
                self.gen_optimizer.step()
        len_ = len(self.cache_dataset)
        L_CE_G = L_CE_G.detach().cpu().numpy() / self.I_gg / len_
        L_KL = L_KL.detach().cpu().numpy() / self.I_gg / len_
        L_MSE_G = L_MSE_G.detach().cpu().numpy() / self.I_gg / len_
        L_DIV = L_DIV.detach().cpu().numpy() / self.I_gg / len_
        LOSS = LOSS.detach().cpu().numpy() / self.I_gg / len_

        info = '\nGenerator: client_i={}, comm={}, L_CE_G={:.4f}, L_KL={:.4f}, L_MSE_G={:.4f}, L_DIV={:.4f}'.format(self.client_id, self.step, L_CE_G, L_KL, L_MSE_G, L_DIV)
        # print(info)
        self.res.update(g_L_CE_G=L_CE_G, g_L_KL=L_KL, g_L_MSE_G=L_MSE_G, g_L_DIV=L_DIV, g_LOSS=LOSS)
        # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        self.generator_model.zero_grad(set_to_none=True)
        self.gen_optimizer.zero_grad(set_to_none=True)
        grad_True(self.model)


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

