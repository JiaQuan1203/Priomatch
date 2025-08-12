import argparse
import os
import torch
import numpy as np
import random
import torchvision.datasets as dsets
import torch.utils.data as data
from semilearn.datasets.cv_datasets.eurosat import EuroSat

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to precompute ground truth labels", required=True)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datasets(dataset, transform, root_dir='./data'):

    if dataset == 'cifar10':
        data_path = os.path.join(root_dir, "cifar10")
        train_dataset = dsets.CIFAR10(root=data_path, train=True, transform=transform, download=False)
        val_dataset = dsets.CIFAR10(root=data_path, train=False, transform=transform, download=False)

    elif dataset == 'cifar100':
        data_path = os.path.join(root_dir, "cifar100")
        train_dataset = dsets.CIFAR100(root=data_path, train=True, transform=transform, download=False)
        val_dataset = dsets.CIFAR100(root=data_path, train=False, transform=transform, download=False)

    elif dataset == 'stl10':
        data_path = os.path.join(root_dir, "stl10")
        train_dataset = dsets.STL10(data_path, split='train', transform=transform, download=False)
        val_dataset = dsets.STL10(data_path, split='test', transform=transform, download=False)

    elif dataset == 'svhn':
        data_path = os.path.join(root_dir, "svhn")
        train_split = dsets.SVHN(root=data_path, split='train', transform=transform, download=False)
        extra_split = dsets.SVHN(root=data_path, split='extra', transform=transform, download=False)
        train_dataset = data.ConcatDataset([train_split, extra_split])
    
    elif dataset == 'eurosat':
        data_path = os.path.join(root_dir, "eurosat", 'data')
        train_dataset = EuroSat('produce', data_path, split="produce", transform=transform)
        val_dataset = EuroSat('test', data_path, split="test", transform=transform)

    return train_dataset, val_dataset

def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    elif hasattr(dataset, "labels"):
        return dataset.labels
    elif hasattr(dataset, "_labels"): # food101 or aircraft
        return dataset._labels
    elif hasattr(dataset, "_samples"): # cars
        return [elem[1] for elem in dataset._samples]
    else:
        return [dataset[i][1] for i in range(len(dataset))]


def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)

    train_dataset, val_dataset = get_datasets(args.dataset, None, args.root_dir)
    labels_train = get_labels(train_dataset)
    labels_val = get_labels(val_dataset)
    
    print(f"Num train: {len(labels_train)}")
    print(f"Num val: {len(labels_val)}")
    print(f"Num classes: {len(np.unique(labels_train))}")

    labels_dir = os.path.join(args.root_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    np.save(f"{labels_dir}/{args.dataset}_train.npy", labels_train)
    np.save(f"{labels_dir}/{args.dataset}_val.npy", labels_val)


if __name__ == '__main__':
    run()