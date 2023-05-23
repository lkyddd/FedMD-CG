import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets as vision_datasets
from lightfed.tools.funcs import save_pkl
from lightfed.tools.funcs import set_seed
from experiments.models.model import GENERATORCONFIGS



class TransDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.dataset)


class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class DataDistributer:
    def __init__(self, args, dataset_dir=None, cache_dir=None):
        if dataset_dir is None:
            dataset_dir = os.path.abspath(os.path.join(__file__, "../../../../dataset"))

        if cache_dir is None:
            cache_dir = f"{dataset_dir}/cache_data"

        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.args = args
        self.client_num = args.client_num
        self.batch_size = args.batch_size

        self.data_set = args.data_set

        self.class_num = None
        self.x_shape = None
        self.client_train_dataloaders = []
        self.client_noise_dataset = []
        self.full_noise_dataset = None
        self.full_label = None
        self.client_test_dataloaders = []
        self.train_dataloaders = None
        self.test_dataloaders = None

        self.raw_train_data = None
        self.raw_test_data = None

        self.raw_train_data_dataloaders = None
        self.raw_test_data_dataloaders = None


        _dataset_load_func = getattr(self, f'_load_{args.data_set.replace("-","_")}')
        _dataset_load_func() 
        
    def get_client_train_dataloader(self, client_id):
        return self.client_train_dataloaders[client_id]

    def get_client_noise_dataset(self, client_id):
        return self.client_noise_dataset[client_id]

    def get_client_test_dataloader(self, client_id):
        return self.client_test_dataloaders[client_id]


    def get_train_dataloader(self):
        return self.train_dataloaders

    def get_full_noise_label(self):
        return self.full_noise_dataset, self.full_label


    def get_test_dataloader(self):
        return self.test_dataloaders

    def get_raw_train_data(self):
        return self.raw_train_data

    def get_raw_test_data(self):
        return self.raw_test_data

    def get_raw_train_dataloader(self):
        return self.raw_train_data_dataloaders

    def get_raw_test_dataloader(self):
        return self.raw_test_data_dataloaders



    def _load_MNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=True, download=True, transform=transform)
        test_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=False, download=True, transform=transform)

        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=False)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=False)

        if self.args.data_partition_mode == 'None':
            return

        client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)
            self.client_train_dataloaders.append(_train_dataloader)

            cache_noise_dataset = []
            i = 0
            for _, y in _train_dataloader:
                eps = torch.rand((y.shape[0], GENERATORCONFIGS[self.data_set][-1]))
                cache_noise_dataset.append((eps, y))
                if i == 0:
                    full_client_noise = eps
                    full_client_y = y.numpy()
                else:
                    full_client_noise = np.vstack((full_client_noise, eps))
                    full_client_y = np.hstack((full_client_y, y))
                i += 1

            if client_id == 0:
                self.full_noise_dataset = full_client_noise
                self.full_label = full_client_y
            else:
                self.full_noise_dataset = np.vstack((self.full_noise_dataset, full_client_noise))
                self.full_label = np.hstack((self.full_label, full_client_y))

            self.client_noise_dataset.append(cache_noise_dataset)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_FMNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])


        raw_train_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=True, download=True)
        raw_test_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=False, download=True)

        self.raw_train_data = raw_train_dataset
        self.raw_test_data = raw_test_dataset

        train_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=True, download=True,
                                                     transform=transform)
        test_dataset = vision_datasets.FashionMNIST(root=f"{self.dataset_dir}/FMNIST", train=False, download=True, transform=transform)

        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=False)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=False)

        if self.args.data_partition_mode == 'None':
            return

        client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        # client_train_datasets = self._split_dataset(train_dataset, test_dataset)
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)
            self.client_train_dataloaders.append(_train_dataloader)

            cache_noise_dataset = []
            i = 0
            for _, y in _train_dataloader:
                eps = torch.rand((y.shape[0], GENERATORCONFIGS[self.data_set][-1]))
                cache_noise_dataset.append((eps, y))
                if i == 0:
                    full_client_noise = eps
                    full_client_y = y.numpy()
                else:
                    full_client_noise = np.vstack((full_client_noise, eps))
                    full_client_y = np.hstack((full_client_y, y))
                i += 1
            if client_id == 0:
                self.full_noise_dataset = full_client_noise
                self.full_label = full_client_y
            else:
                self.full_noise_dataset = np.vstack((self.full_noise_dataset, full_client_noise))
                self.full_label = np.hstack((self.full_label, full_client_y))

            self.client_noise_dataset.append(cache_noise_dataset)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_EMNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        transform_DLG = transforms.Compose([transforms.ToTensor()])
        raw_train_dataset_DLG = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=True,
                                               download=False, transform=transform_DLG)
        raw_test_dataset_DLG = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=False,
                                              download=False, transform=transform_DLG)

        self.raw_train_data_dataloaders = DataLoader(dataset=raw_train_dataset_DLG, batch_size=self.args.eval_batch_size,
                                                     drop_last=True, shuffle=False)
        self.raw_test_data_dataloaders = DataLoader(dataset=raw_test_dataset_DLG, batch_size=self.args.eval_batch_size,
                                                    drop_last=True, shuffle=False)


        raw_train_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=True,
                                               download=False)
        raw_test_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=False,
                                              download=False)

        self.raw_train_data = raw_train_dataset
        self.raw_test_data = raw_test_dataset

        train_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=True, download=False, transform=transform)
        test_dataset = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=False, download=False, transform=transform)
        # train_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/EMNIST", train=True, download=False, transform=transform)
        # test_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/EMNIST", train=False, download=False, transform=transform)

        ###train data
        if len(train_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=False)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=False)



        if self.args.data_partition_mode == 'None':
            return

        client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        # client_train_datasets = self._split_dataset(train_dataset, test_dataset)
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            _test_dataset = client_test_datasets[client_id]
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_train_dataset), shuffle=False)
            self.client_train_dataloaders.append(_train_dataloader)

            cache_noise_dataset = []
            i = 0
            for _, y in _train_dataloader:
                eps = torch.rand((y.shape[0], GENERATORCONFIGS[self.data_set][-1]))
                cache_noise_dataset.append((eps, y))
                if i == 0:
                    full_client_noise = eps
                    full_client_y = y.numpy()
                else:
                    full_client_noise = np.vstack((full_client_noise, eps))
                    full_client_y = np.hstack((full_client_y, y))
                i += 1

            if client_id == 0:
                self.full_noise_dataset = full_client_noise
                self.full_label = full_client_y
            else:
                self.full_noise_dataset = np.vstack((self.full_noise_dataset, full_client_noise))
                self.full_label = np.hstack((self.full_label, full_client_y))

            self.client_noise_dataset.append(cache_noise_dataset)
            self.client_test_dataloaders.append(_test_dataloader)


    def _load_CIFAR_10(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)

        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4), 
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

        raw_train_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=True, download=True)
        raw_test_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=False, download=True)

        self.raw_train_data = raw_train_dataset
        self.raw_test_data = raw_test_dataset

        train_dataset = TransDataset(raw_train_dataset, train_transform)
        test_dataset = TransDataset(raw_test_dataset, test_transform)
        ###train data
        if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        # test data
        if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size,
                                            drop_last=drop_last_train, shuffle=False)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size,
                                           drop_last=drop_last_test, shuffle=False)

        if self.args.data_partition_mode == 'None':
            return

        raw_client_train_datasets, raw_client_test_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        # raw_client_train_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        for client_id in range(self.client_num):
            _raw_train_dataset = raw_client_train_datasets[client_id]
            _raw_test_dataset = raw_client_test_datasets[client_id]
            _train_dataset = TransDataset(_raw_train_dataset, train_transform)
            _test_dataset = TransDataset(_raw_test_dataset, test_transform)
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            self.client_train_dataloaders.append(_train_dataloader)

            cache_noise_dataset = []
            i = 0
            for _, y in _train_dataloader:
                eps = torch.rand((y.shape[0], GENERATORCONFIGS[self.data_set][-1]))
                cache_noise_dataset.append((eps, y))
                if i == 0:
                    full_client_noise = eps
                    full_client_y = y.numpy()
                else:
                    full_client_noise = np.vstack((full_client_noise, eps))
                    full_client_y = np.hstack((full_client_y, y))
                i += 1

            if client_id == 0:
                self.full_noise_dataset = full_client_noise
                self.full_label = full_client_y
            else:
                self.full_noise_dataset = np.vstack((self.full_noise_dataset, full_client_noise))
                self.full_label = np.hstack((self.full_label, full_client_y))

            self.client_noise_dataset.append(cache_noise_dataset)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_CIFAR_100(self):
        self.class_num = 100
        self.x_shape = (3, 32, 32)

        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        raw_train_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=True, download=True)
        raw_test_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=False, download=True)

        train_dataset = TransDataset(raw_train_dataset, train_transform)
        test_dataset = TransDataset(raw_test_dataset, test_transform)
        ###train data
        if len(train_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_train = True
        else:
            drop_last_train = False
        #test data
        if len(test_dataset.dataset.targets) % self.args.eval_batch_size == 1:
            drop_last_test = True
        else:
            drop_last_test = False
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_train, shuffle=False)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, drop_last=drop_last_test, shuffle=False)


        if self.args.data_partition_mode == 'None':
            return

        raw_client_train_datasets, raw_client_test_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        # raw_client_train_datasets  = self._split_dataset(raw_train_dataset, raw_test_dataset)
        for client_id in range(self.client_num):
            _raw_train_dataset = raw_client_train_datasets[client_id]
            _raw_test_dataset = raw_client_test_datasets[client_id]
            _train_dataset = TransDataset(_raw_train_dataset, train_transform)
            _test_dataset = TransDataset(_raw_test_dataset, test_transform)
            if self.args.batch_size > 0:
                if len(_train_dataset) >= self.args.batch_size:
                    if len(_train_dataset.dataset.data_list) % self.args.batch_size == 1:
                        drop_last_train = True
                    else:
                        drop_last_train = False
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, drop_last=drop_last_train, shuffle=False)
                elif len(_train_dataset) < self.args.batch_size:
                    _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)

                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            elif self.args.batch_size == 0:
                _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=len(_train_dataset), shuffle=False)
                _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=len(_test_dataset), shuffle=False)

            self.client_train_dataloaders.append(_train_dataloader)

            cache_noise_dataset = []
            i = 0
            for _, y in _train_dataloader:
                eps = torch.rand((y.shape[0], GENERATORCONFIGS[self.data_set][-1]))
                cache_noise_dataset.append((eps, y))
                if i == 0:
                    full_client_noise = eps
                    full_client_y = y.numpy()
                else:
                    full_client_noise = np.vstack((full_client_noise, eps))
                    full_client_y = np.hstack((full_client_y, y))
                i += 1

            if client_id == 0:
                self.full_noise_dataset = full_client_noise
                self.full_label = full_client_y
            else:
                self.full_noise_dataset = np.vstack((self.full_noise_dataset, full_client_noise))
                self.full_label = np.hstack((self.full_label, full_client_y))

            self.client_noise_dataset.append(cache_noise_dataset)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_CELEBA(self):
        train_transforms = transforms.ToTensor()
        test_transforms = transforms.ToTensor()

        train_dataset = vision_datasets.CelebA(root=f"{self.dataset_dir}/CELEBA", split='train', target_type='attr', download=True)
        test_dataset = vision_datasets.CelebA(root=f"{self.dataset_dir}/CELEBA", split='test', target_type='attr', download=True)


    def _split_dataset(self, train_dataset, test_dataset):
        if self.args.data_partition_mode == 'iid':
            partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1/self.client_num)
            client_train_datasets = self._split_dataset_iid(train_dataset, partition_proportions)

        elif self.args.data_partition_mode == 'non_iid_dirichlet_unbalanced':
            client_train_datasets = self._split_dataset_dirichlet_unbalanced(train_dataset, self.client_num, alpha=self.args.non_iid_alpha)

        elif self.args.data_partition_mode == 'non_iid_dirichlet_balanced':
            client_train_datasets = self._split_dataset_dirichlet_balanced(train_dataset, self.client_num, alpha=self.args.non_iid_alpha)
        else:
            raise Exception(f"unknow data_partition_mode:{self.args.data_partition_mode}")

        partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1 / self.client_num)
        client_test_datasets = self._split_dataset_iid(test_dataset, partition_proportions)

        return client_train_datasets, client_test_datasets

    def _split_dataset_dirichlet_unbalanced(self, train_dataset, n_nets, alpha=0.01):
        y_train = train_dataset.targets

        min_size = 0
        K = len(train_dataset.class_to_idx)
        try:
            N = y_train.shape[0]
        except:
            y_train = np.array(y_train)
            N = y_train.shape[0]
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                pp_list = []
                for p, idx_j in zip(proportions, idx_batch):
                    pp = p * (len(idx_j) < N / n_nets)
                    pp_list.append(pp)
                proportions = np.array(pp_list)
                # proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch0 = []
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions)):
                    idx_batch0.append(idx_j + idx.tolist())
                idx_batch = idx_batch0
                # idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

        client_data_list = [[] for _ in range(self.client_num)]
        for client_id, client_data in zip(net_dataidx_map, client_data_list):
            for id in net_dataidx_map[client_id]:
                client_data.append(train_dataset[id])
        client_datasets = []
        for client_data in client_data_list:
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets

    def _split_dataset_dirichlet_balanced(self, train_dataset, n_nets, alpha=0.5):

        y_train = train_dataset.targets
        # y_train = torch.zeros(len(train_dataset), dtype=torch.long)
        #
        # for a in range(len(train_dataset)):
        #     y_train[a] = (train_dataset[a][1])

        K = len(train_dataset.class_to_idx)
        try:
            N = y_train.shape[0]
        except:
            y_train = np.array(y_train)
            N = y_train.shape[0]
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}
        assigned_ids = []
        idx_batch = [[] for _ in range(n_nets)]
        num_data_per_client = int(N / n_nets)
        for i in range(n_nets):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = proportions[k]
            weights[assigned_ids] = 0.0 #            sum(weights)
            try:
                idx_batch[i] = (torch.multinomial(weights, num_data_per_client, replacement=False)).tolist()
            except:
                idx_batch[i] = [i for i in range(N) if i not in assigned_ids]

            assigned_ids += idx_batch[i]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        client_data_list = [[] for _ in range(self.client_num)]
        for client_id, client_data in zip(net_dataidx_map, client_data_list):
            for id in net_dataidx_map[client_id]:
                client_data.append(train_dataset[id])
        client_datasets = []
        for client_data in client_data_list:
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets

    def _split_dataset_iid(self, dataset, partition_proportions):
        data_labels = dataset.targets
        class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                      for y in range(self.class_num)]

        client_idcs = [[] for _ in range(self.client_num)]

        for c, fracs in zip(class_idcs, partition_proportions):
            np.random.shuffle(c)
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i].extend(list(idcs))

        client_data_list = [[] for _ in range(self.client_num)]
        for client_id, client_data in zip(client_idcs, client_data_list):
            for id in client_id:
                client_data.append(dataset[id])
        client_datasets = []
        for client_data in client_data_list:
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets

