import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
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
        self.weight_decay = args.weight_decay

        self.lr_lm = args.lr_lm
        self.I_lm = args.I_lm

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)

        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.CE = nn.CrossEntropyLoss()

        set_seed(args.seed + 657)

        _ ,self.classifier = model_pull(args, l_classifer=True)
        self.model = None


    def clear(self):
        self.model = None
        self.lm_optimizer = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step):
        self.step = step
        self.lm_optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        # self.lm_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr_lm, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        # self.lr_lm_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.lm_optimizer, gamma=0.98)

        ###sample y and generate z
        for tau in range(I):
            self.update_local_model()

    def update_local_model(self, ):
        self.model.train()
        LOSS = 0
        for _ in range(self.I_lm):
            self.model.zero_grad(set_to_none=True)
            self.lm_optimizer.zero_grad(set_to_none=True)

            ##batch
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            model_result = self.model(x, logit=True)
            loss = self.CE(model_result['logit'], y)
            LOSS += loss
            loss.backward()
            self.lm_optimizer.step()
        LOSS = LOSS.detach().cpu().numpy() / self.I_lm
        self.res.update(m_LOSS=LOSS)
        info = '\nclient_i={}, comm={}, LOSS={:.4f}'.format(self.client_id, self.step, LOSS)
        # print(info)
        # logging.debug(f"train_locally_step for step: {tau}")
        # lr_scheduler.step(step)
        self.model.zero_grad(set_to_none=True)
        self.lm_optimizer.zero_grad(set_to_none=True)

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
