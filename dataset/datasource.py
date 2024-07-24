import numpy as np
import random
import torch
import torchvision as tv
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid, cifar_extr_noniid
from dataset.mnist_noniid import get_dataset_mnist_extr_noniid, mnist_extr_noniid
# Given each user euqal number of samples if possible. If not, the last user
# gets whatever is left after other users had their shares


def DataLoaders(n_devices, dataset_name, n_classes, nsamples, log_dirpath, mode="non-iid", batch_size=32, rate_unbalance=1.0, dataloader_workers=1):
    if mode == "non-iid":
        if dataset_name == "mnist":
            return get_data_noniid_mnist(n_devices,
                                         n_classes,
                                         nsamples,
                                         log_dirpath,
                                         batch_size,
                                         rate_unbalance,
                                         dataloader_workers)
        elif dataset_name == "cifar10":
            return get_data_noniid_cifar10(n_devices,
                                           n_classes,
                                           nsamples,
                                           log_dirpath,
                                           batch_size,
                                           rate_unbalance,
                                           dataloader_workers)
    elif mode == 'iid':
        if dataset_name == 'cifar10':
            data_dir = './data'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = tv.datasets.CIFAR10(data_dir, train=True, download=True,
                                                transform=apply_transform)

            test_dataset = tv.datasets.CIFAR10(data_dir, train=False, download=True,
                                               transform=apply_transform)
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers)
        elif dataset_name == 'mnist':
            data_dir = './data'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = tv.datasets.MNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

            test_dataset = tv.datasets.MNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers)


def iid_split(num_clients,
              train_data,
              batch_size, test_data, dataloader_workers):

    all_train_idx = np.arange(train_data.data.shape[0])

    sample_train_idx = np.array_split(all_train_idx, num_clients)

    all_test_idx = np.arange(test_data.data.shape[0])

    sample_test_idx = np.array_split(all_test_idx, num_clients)

    user_train_loaders = []
    user_test_loaders = []

    for idx in sample_train_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data,
                                                              sampler=torch.utils.data.SubsetRandomSampler(
                                                                  idx),
                                                              batch_size=batch_size, num_workers=dataloader_workers))
    for idx in sample_test_idx:
        user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                             sampler=torch.utils.data.SubsetRandomSampler(
                                                                 idx),
                                                             batch_size=batch_size, num_workers=dataloader_workers))
    return user_train_loaders, user_test_loaders


def get_data_noniid_cifar10(n_devices, n_classes, nsamples, log_dirpath, batch_size=32, rate_unbalance=1.0, dataloader_workers=1):

    train_data, test_data, user_train, user_test, user_labels = get_dataset_cifar10_extr_noniid(
        n_devices, n_classes, nsamples, rate_unbalance, log_dirpath)

    train_loaders = []
    test_loaders = []

    for i in range(n_devices):
        user_train_temp = []
        user_test_temp = []
        for j in range(user_train[i].size):
            user_train_temp.append(int(user_train[i][j]))
        for j in range(user_test[i].size):
            user_test_temp.append(int(user_test[i][j]))
        sampler_train = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(user_train_temp), batch_size, drop_last=False)
        loader_train = torch.utils.data.DataLoader(
            train_data, batch_sampler=sampler_train, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        sampler_test = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(user_test_temp), batch_size, drop_last=False)
        loader_test = torch.utils.data.DataLoader(
            test_data, batch_sampler=sampler_test, num_workers=dataloader_workers)
        test_loaders.append(loader_test)
    
    # create global_test_loader (test ticket model before reapplying mask)
    global_test = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(list(range(10000))), batch_size, drop_last=False)
    global_test_loader = torch.utils.data.DataLoader(
        test_data, batch_sampler=global_test, num_workers=dataloader_workers)
    
    return train_loaders, test_loaders, user_labels, global_test_loader


def get_data_noniid_mnist(n_devices, n_classes, nsamples, log_dirpath, batch_size=32, rate_unbalance=1.0, dataloader_workers=1):

    train_data, test_data, user_train, user_test, user_labels = get_dataset_mnist_extr_noniid(
        n_devices, n_classes, nsamples, rate_unbalance, log_dirpath)

    train_loaders = []
    test_loaders = []

    for i in range(n_devices):
        user_train_temp = []
        user_test_temp = []
        for j in range(user_train[i].size):
            user_train_temp.append(int(user_train[i][j]))
        for j in range(user_test[i].size):
            user_test_temp.append(int(user_test[i][j]))
        sampler_train = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(user_train_temp), batch_size, drop_last=False)
        loader_train = torch.utils.data.DataLoader(
            train_data, batch_sampler=sampler_train, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        sampler_test = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(user_test_temp), batch_size, drop_last=False)
        loader_test = torch.utils.data.DataLoader(
            test_data, batch_sampler=sampler_test)
        test_loaders.append(loader_test)

    # create global_test_loader (test ticket model before reapplying mask)
    global_test = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(list(range(10000))), batch_size, drop_last=False)
    global_test_loader = torch.utils.data.DataLoader(
        test_data, batch_sampler=global_test, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_labels, global_test_loader
