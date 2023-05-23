import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters
from torch import nn


class ClientTrainer:
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.lr_lm = args.lr_lm

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.res = {}

        set_seed(args.seed + 657)
        self.model, _ = model_pull(args)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr_lm, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, lr_lm=0.98)

    def clear(self):
        # self.model = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I):
        self.model.train()
        LOSS = 0
        for tau in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()

            ##batch
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)

            model_result = self.model(x, logit=True)
            loss = self.criterion(model_result['logit'], y)
            loss.backward()
            self.optimizer.step()
            LOSS += loss
            # logging.debug(f"train_locally_step for step: {tau}")
        LOSS = LOSS.detach().cpu().numpy() / I
        self.res.update(m_LOSS=LOSS)
        self.model.zero_grad(set_to_none=True)


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
                                    criterion=self.criterion,
                                    model_params=None,
                                    device=self.device)
        self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
