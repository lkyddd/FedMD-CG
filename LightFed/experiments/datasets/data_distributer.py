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
        self.x_shape = None  # 每个样本x的shape
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



        # _cache_file_name = f"{self.cache_dir}/{self.args.data_set}_seed_{self.args.seed}_client_num_{self.client_num}_{self.args.data_partition_mode}"
        # if self.args.data_partition_mode == 'non_iid_dirichlet':
        #     _cache_file_name += f"_{self.args.non_iid_alpha}"
        # _cache_file_name += f"{self.args.device.type}.pkl"

        # if os.path.exists(_cache_file_name):
        #     self.class_num, self.x_shape, \
        #         self.client_train_dataloaders, self.client_test_dataloaders, \
        #         self.train_dataloaders, self.test_dataloaders = load_pkl(_cache_file_name)
        #     return

        # 由数据集名字拼接出加载数据集的函数名
        # 数据集名字中的减号会被替换为下划线，因为函数名不能有下划线
        _dataset_load_func = getattr(self, f'_load_{args.data_set.replace("-","_")}')
        _dataset_load_func()  # 调用数据集加载函数

        # if 'MNIST' in args.data_set:
        #     self._load_MNIST()
        # elif 'FMNIST' in args.data_set:
        #     self._load_FMNIST()
        # elif 'CIFAR-10' in args.data_set:
        #     self._load_CIFAR_10()
        # elif 'CIFAR-100' in args.data_set:
        #     self._load_CIFAR_100()


        # save_pkl((self.class_num, self.x_shape,
        #           self.client_train_dataloaders, self.client_test_dataloaders,
        #           self.train_dataloaders, self.test_dataloaders),
        #          _cache_file_name)

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

    def _load_FMNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        # transform = transforms.Compose([transforms.ToTensor()])

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

        ###用于训练的原始图片
        transform_DLG = transforms.Compose([transforms.ToTensor()])
        raw_train_dataset_DLG = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=True,
                                               download=False, transform=transform_DLG)
        raw_test_dataset_DLG = vision_datasets.EMNIST(root=f"{self.dataset_dir}/EMNIST", split="digits", train=False,
                                              download=False, transform=transform_DLG)

        self.raw_train_data_dataloaders = DataLoader(dataset=raw_train_dataset_DLG, batch_size=self.args.eval_batch_size,
                                                     drop_last=True, shuffle=False)
        self.raw_test_data_dataloaders = DataLoader(dataset=raw_test_dataset_DLG, batch_size=self.args.eval_batch_size,
                                                    drop_last=True, shuffle=False)


        ##原始图片
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

        # 只开RandomCrop和RandomHorizontalFlip是标准的CIFA-10数据扩展
        train_transform = transforms.Compose([
            # transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.RandomCrop(size=32, padding=4),  # 填充后裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ToTensor(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 颜色变化。亮度
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
        # partition_proportions中的第i行是第i个类别的数据被划分到各个客户端的比例
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
        # y_train = torch.zeros(len(train_dataset), dtype=torch.long)
        # print(y_train.dtype)
        # for a in range(len(train_dataset)):
        #     y_train[a] = (train_dataset[a][1])

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
            # 打乱数据集
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
            # 打乱数据集
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets

    def _split_dataset_iid(self, dataset, partition_proportions):
        data_labels = dataset.targets
        # 记录每个K个类别对应的样本下标
        class_idcs = [list(np.argwhere(np.array(data_labels) == y).flatten())
                      for y in range(self.class_num)]

        #记录N个client分别对应样本集合的索引
        client_idcs = [[] for _ in range(self.client_num)]

        for c, fracs in zip(class_idcs, partition_proportions):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            np.random.shuffle(c)
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i].extend(list(idcs))

        client_data_list = [[] for _ in range(self.client_num)]
        for client_id, client_data in zip(client_idcs, client_data_list):
            for id in client_id:
                client_data.append(dataset[id])
        # label_partition = []
        # for dtset in client_data_list:
        #     label_partition.append([lb[1] for lb in dtset])
        #
        # save_pkl(label_partition, file_path=f"{os.path.dirname(__file__)}/label_partition/{self.args.data_set}_{self.args.non_iid_alpha}")
        client_datasets = []
        for client_data in client_data_list:
            # 打乱数据集
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=128)  # 128

        parser.add_argument('--data_set', type=str, default='CIFAR-10')  # BBC-TEXT, AG-NEWS暂时无法使用

        parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet',
                            choices=['iid', 'non_iid_dirichlet'])

        parser.add_argument('--non_iid_alpha', type=float, default=0.5)  # 在进行non_iid_dirichlet数据划分时需要, 该值越大表明数据月均匀

        parser.add_argument('--client_num', type=int, default=10)

        parser.add_argument('--device', type=torch.device, default='cuda')

        parser.add_argument('--seed', type=int, default=0)

        parser.add_argument('--app_name', type=str, default='DecentLaM')

        args = parser.parse_args(args=[])

        return args

    args = get_args()
    dd = DataDistributer(args)


##################################################################

# label_partition = []
        # for dtset in client_data_list:
        #     label_partition.append([lb[1] for lb in dtset])
        #
        # save_pkl(label_partition, file_path=f"{os.path.dirname(__file__)}/label_partition/{self.args.data_set}_{self.args.non_iid_alpha}")

# 老代码
##################################################################

# from lightfed.tools.extract_dataset_CIFAR10 import Dataset_CIFAR10
# from lightfed.tools.extract_dataset_CIFAR100 import Dataset_CIFAR100
# from transformers import BertTokenizer
# from sklearn.datasets import load_svmlight_file

# ---------------- 加载：COVERTYPE数据集 ---------------  # 训练集：400,000；测试集：181,012；特征维数：54
# class Deal_Dataset_COVERTYPE:
#     def __init__(self, folder, is_train):
#         data = np.genfromtxt(GzipFile(filename=folder+'/covtype.data.gz'), delimiter=',')
#         np.random.shuffle(data)
#         data_X = data[:, :-1]
#         data_Y = data[:, -1].astype(np.int32)-1
#         if is_train:
#             self.train_set = data_X[:400000, ]
#             self.train_labels = data_Y[:400000, ]
#         else:
#             self.train_set = data_X[400000:, ]
#             self.train_labels = data_Y[400000:, ]

#     def __getitem__(self, index):
#         feature, target = self.train_set[index], int(self.train_labels[index])
#         return feature, target

#     def __len__(self):
#         return len(self.train_set)
# ----------------------------------------------------


# -------------------- 加载：A9A数据集 ------------------   # 训练集：32,561；测试集：16,281；特征维数：122
# class Deal_Dataset_A9A:
#     def __init__(self, file_path, is_train):
#         self.train_set, self.train_labels = load_svmlight_file(file_path)
#         if is_train:
#             self.train_set = self.train_set.todense().A[:, :-1]
#         else:
#             self.train_set = self.train_set.todense().A
#         self.train_labels[self.train_labels == -1.0] = 0
#         self.train_labels[self.train_labels == 1.0] = 1

#     def __getitem__(self, index):
#         feature, target = self.train_set[index], int(self.train_labels[index])
#         return feature, target

#     def __len__(self):
#         return len(self.train_set)
# ----------------------------------------------------


# -------------------- 加载：W8A数据集 ------------------  # 训练集：49,749；测试集：14,951；特征维数：300
# class Deal_Dataset_W8A(Dataset):
#     def __init__(self, file_path):
#         self.train_set, self.train_labels = load_svmlight_file(file_path)
#         self.train_set = self.train_set.todense().A
#         self.train_labels[self.train_labels == -1.0] = 0
#         self.train_labels[self.train_labels == 1.0] = 1

#     def __getitem__(self, index):
#         feature, target = self.train_set[index], int(self.train_labels[index])
#         return feature, target

#     def __len__(self):
#         return len(self.train_set)
# ----------------------------------------------------


# -------------------- 加载：FEMNIST数据集 ------------------  # 训练集：341,873；测试集：40,832；特征维数：784
# class Deal_Dataset_FEMNIST:
#     def __init__(self, data_file_path='', transform=None):
#         self.train_set, self.train_labels, self.users_index = torch.load(data_file_path)
#         self.transform = transform

#     def __getitem__(self, index):
#         img, target = self.train_set[index], int(self.train_labels[index])
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target

#     def __len__(self):
#         return len(self.train_set)
# ----------------------------------------------------


# # -------------------- 加载：BBC-TEXT数据集 ------------------  # 训练集：1,780；测试集：445
# class Deal_Dataset_BBCTEXT:
#     def __init__(self, df, is_train):
#         tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#         labels = {'business': 0,
#                   'entertainment': 1,
#                   'sport': 2,
#                   'tech': 3,
#                   'politics': 4
#                   }
#         df_train, df_test = np.split(df, (int(0.8*len(df)),))
#         if is_train:
#             text_data = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df_train['text']]
#             self.train_set = []
#             for val in text_data:
#                 self.train_set.append(torch.cat((val['input_ids'], val['attention_mask']), 0))
#             self.train_labels = [labels[label] for label in df_train['category']]
#         else:
#             text_data = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df_test['text']]
#             self.train_set = []
#             for val in text_data:
#                 self.train_set.append(torch.cat((val['input_ids'], val['attention_mask']), 0))
#             self.train_labels = [labels[label] for label in df_test['category']]
#
#     def __getitem__(self, index):
#         feature, target = self.train_set[index], int(self.train_labels[index])
#         return feature, target
#
#     def __len__(self):
#         return len(self.train_set)
# # ----------------------------------------------------
#
#
# # -------------------- 加载：AG-NEWS数据集 ------------------  # 训练集：120,000；测试集：7,600
# class Deal_Dataset_AGNEWS:
#     def __init__(self, df):
#         tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#         labels = {1: 0,
#                   2: 1,
#                   3: 2,
#                   4: 3
#                   }
#         text_data = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df['text']]
#         self.train_set = []
#         for val in text_data:
#             self.train_set.append(torch.cat((val['input_ids'], val['attention_mask']), 0))
#         self.train_labels = [labels[label] for label in df['category']]
#
#     def __getitem__(self, index):
#         feature, target = self.train_set[index], int(self.train_labels[index])
#         return feature, target
#
#     def __len__(self):
#         return len(self.train_set)
# # ----------------------------------------------------


# elif args.data_set == 'CIFAR_100':
#     # CIFAR-100中有两类标签：分100类的精准标签（超类），分20类的粗略标签（子类）
#     # 设置'set_class_num_for_cifar_100'为100（默认值）或20来选择目标分类数
#     set_class_num_for_cifar_100 = 100
#     self.class_num = set_class_num_for_cifar_100
#     self.feature_size = 3072
#     self.channel_size = 3
#     train_dataset = Deal_Dataset_CIFAR100(path + '/dataset/CIFAR-100/cifar100_data', is_train=True,
#                                           classes=set_class_num_for_cifar_100,
#                                           transform=transforms.Compose([transforms.ToTensor(),
#                                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
#     test_dataset = Deal_Dataset_CIFAR100(path + '/dataset/CIFAR-100/cifar100_data', is_train=False,
#                                          classes=set_class_num_for_cifar_100,
#                                          transform=transforms.Compose([transforms.ToTensor(),
#                                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
#     self._init_train_dataloader_(train_dataset)
#     self._init_test_dataloader_(test_dataset)

# elif args.data_set == 'COVERTYPE':
#     self.class_num = 7
#     self.feature_size = 54
#     self.channel_size = 0
#     train_dataset = Deal_Dataset_COVERTYPE(path + '/dataset/COVERTYPE/covtype_data', is_train=True)
#     test_dataset = Deal_Dataset_COVERTYPE(path + '/dataset/COVERTYPE/covtype_data', is_train=False)
#     self._init_train_dataloader_(train_dataset)
#     self._init_test_dataloader_(test_dataset)

# elif args.data_set == 'A9A':
#     self.class_num = 2
#     self.feature_size = 122
#     self.channel_size = 0
#     train_dataset = Deal_Dataset_A9A(path + '/dataset/A9A/a9a_data/a9a.txt', is_train=True)
#     test_dataset = Deal_Dataset_A9A(path + '/dataset/A9A/a9a_data/a9a.t', is_train=False)
#     self._init_train_dataloader_(train_dataset)
#     self._init_test_dataloader_(test_dataset)

# elif args.data_set == 'W8A':
#     self.class_num = 2
#     self.feature_size = 300
#     self.channel_size = 0
#     train_dataset = Deal_Dataset_W8A(path + '/dataset/W8A/w8a_data/w8a.txt')
#     test_dataset = Deal_Dataset_W8A(path + '/dataset/W8A/w8a_data/w8a.t')
#     self._init_train_dataloader_(train_dataset)
#     self._init_test_dataloader_(test_dataset)

# elif args.data_set == 'FEMNIST':
#     self.class_num = 10
#     self.feature_size = 784
#     self.channel_size = 1
#     train_dataset = Deal_Dataset_FEMNIST(data_file_path=path + '/dataset/FEMNIST/femnist_data/training.pt',
#                                          transform=)
#     test_dataset = Deal_Dataset_FEMNIST(data_file_path=path + '/dataset/FEMNIST/femnist_data/test.pt',
#                                         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
#     self._init_train_dataloader_(train_dataset)
#     self._init_test_dataloader_(test_dataset)
# else:
#     raise Exception(f"unkonw data_set: {args.data_set}")
