import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed, grad_False, grad_True
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters
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

        self.lr_lm = args.lr_lm
        self.lambda_ce_1 = args.lambda_ce_1
        self.lambda_kl = args.lambda_kl
        self.I_lm = args.I_lm
        self.temp_c = args.temp_c

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.unique_labels = args.data_distributer.class_num
        self.qualified_labels = [i for i in range(self.unique_labels)]
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

        # for _, label in self.train_dataloader.dataset:
        #     self.label_counts[label] += 1

        self.init_loss_fn()

        self.cache_dataset = [] ##用于计算没有重复batch的数据集
        self.res = {}

        set_seed(args.seed + 657)
        self.model, self.classifier = model_pull(args, g_classifer=True)
        self.generator_model = None

    def init_loss_fn(self):
        self.MSE = nn.MSELoss()
        self.KL = nn.KLDivLoss(reduction="batchmean") #
        self.CE = nn.CrossEntropyLoss()

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

    def y_select_in_dataset(self, y):
        for buff_data in self.cache_dataset:
            try:
                if (y - buff_data[1]).detach().cpu().numpy().sum() == 0:
                    return True
            except:
                return True
        return False

    def clear(self):
        # self.model = None
        self.generator_model = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step, label_num, latent_layer_idx):
        """算法的第5行
        """
        # self.clean_up_counts()
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
        # generative_alpha = self.exp_lr_scheduler(step, decay=0.98, init_lr=self.generative_alpha)
        # generative_beta = self.exp_lr_scheduler(step, decay=0.98, init_lr=self.generative_beta)
        ##get generator output(latent representation) of the same label

        grad_False(self.generator_model)
        for _ in range(I):
            L_CE, L_CE_1, L_KL, LOSS = 0, 0, 0, 0
            self.model.train()
            self.generator_model.eval()
            for _ in range(self.I_lm):
                self.model.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)

                ##batch数据
                x, y = next(self.train_batch_data_iter)
                x = x.to(self.device)
                y = y.to(self.device)
                if not self.y_select_in_dataset(y):
                    self.cache_dataset.append((x, y))
                model_result = self.model(x, logit=True)
                L_ce = self.CE(model_result['logit'], y)

                ###sample y and generate z
                if step:
                    model_logsoftmax_temp = F.log_softmax(model_result['logit'] / self.temp_c, dim=1)

                    gen_result = self.generator_model(y.clone().detach().cpu())
                    gen_model_result = self.model(gen_result['output'], start_layer_idx=latent_layer_idx, logit=True)
                    gen_model_softmax_temp = F.softmax(gen_model_result['logit'] / self.temp_c, dim=1).clone().detach()
                    L_kl = self.KL(model_logsoftmax_temp, gen_model_softmax_temp)  #p*log(q)

                    sampled_y = self.random_choice_y(self.gen_batch_size, label_num)
                    sampled_y_int = torch.LongTensor(sampled_y)
                    gen_result_sam_y = self.generator_model(sampled_y_int)

                    gen_model_result_sam_y = self.model(gen_result_sam_y['output'], start_layer_idx=latent_layer_idx, logit=True)
                    L_ce_1 = self.CE(gen_model_result_sam_y['logit'], sampled_y_int.to(self.device))

                    # this is to further balance oversampled down-sampled synthetic data
                    loss = L_ce + self.lambda_ce_1 * L_ce_1 + self.lambda_kl * L_kl

                    L_CE += L_ce
                    L_CE_1 += L_ce_1
                    L_KL += L_kl
                    LOSS += loss
                else:
                    loss = L_ce
                    L_CE += L_ce
                    LOSS += loss
                loss.backward()
                optimizer.step()

            L_CE = L_CE.detach().cpu().numpy() / self.I_lm
            LOSS = LOSS.detach().cpu().numpy() / self.I_lm
            try:
                L_CE_1 = L_CE_1.detach().cpu().numpy() / self.I_lm
                L_KL = L_KL.detach().cpu().numpy() / self.I_lm
            except:
                L_CE_1 = 0
                L_KL = 0

            info = '\nlocal_model: client_i={}, comm={}, L_CE={:.4f}, L_CE_1={:.4f}, L_KL={:.4f}'.format(self.client_id, step, L_CE, L_CE_1, L_KL)
            self.res.update(m_L_CE=L_CE,  m_L_CE_1=L_CE_1,  m_L_KL_=L_KL, m_LOSS=LOSS)
            # print(info)
            self.model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        for _, y in self.cache_dataset:
            counter_dict = dict(Counter(y.cpu().numpy()))
            self.update_label_counts(counter_dict)
        self.cache_dataset = []


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
