# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import sample_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment


mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
img_size = 96

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(img_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_stl10(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=False):
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, split='train', download=True)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels.astype(np.int64)

    # Note this data can have imbalanced labeled set, and with unknown unlabeled set

    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes,
                                              lb_num_labels=num_labels,
                                              ulb_num_labels=args.ulb_num_labels,
                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                              ulb_imbalance_ratio=args.ulb_imb_ratio,
                                              load_exist=True)

    lb_data, lb_targets = data[lb_idx], targets[lb_idx]
    ulb_data, ulb_targets = data[ulb_idx], targets[ulb_idx]

    if include_lb_to_ulb:
        ulb_data = np.concatenate([lb_data, ulb_data], axis=0)
        ulb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
        ulb_targets = ulb_targets.astype(np.int64)

    # output the distribution of labeled data for remixmatch
    count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        count[c] += 1
    dist = np.array(count, dtype=float)
    dist = dist / dist.sum()
    dist = dist.tolist()
    out = {"distribution": dist}
    output_file = r"./data_statistics/"
    output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    with open(output_path, 'w') as w:
        json.dump(out, w)

    if alg == 'priomatch':
        phi = args.cluster_feature_model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
        lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
        ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_{args.ulb_imb_ratio}_seed{args.seed}_idx.npy')
        cluster_result_dir = os.path.join(base_dir, 'data', 'cluster_result', phi, f'{args.dataset}_{args.num_labels}')
        pred_label_path = os.path.join(cluster_result_dir,'pred_label.npy')
        silhouette_path = os.path.join(cluster_result_dir,'silhouette.npy')

        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)

        if include_lb_to_ulb:
            ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)

        pred_label = np.load(pred_label_path)
        silhouette = np.load(silhouette_path)

        lb_pred_label = pred_label[lb_idx]
        ulb_pred_label = pred_label[ulb_idx]

        lb_silhouette = silhouette[lb_idx]
        ulb_silhouette = silhouette[ulb_idx]

    if alg == 'priomatch':
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_medium, transform_strong, False)
        ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False, ulb_pred_label, ulb_silhouette)

    else:
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_medium, transform_strong, False)
        ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='test', download=True)
    data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset