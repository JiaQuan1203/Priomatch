import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
import clip
import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.utils.data as data
from semilearn.datasets.cv_datasets.eurosat import EuroSat

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to precompute embeddings")
    parser.add_argument('--phis', type=str, default="clipvitL14", help="Representation spaces to precompute", 
                            choices=['clipRN50', 'clipRN101', 'clipRN50x4', 'clipRN50x16', 'clipRN50x64', 'clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2L14', 'dinov2B14'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--root_dir', type=str, default="data", help='Root dir to store everything')
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(args)

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _convert_image_to_rgb(image):
    if torch.is_tensor(image):
        return image
    else:
        return image.convert("RGB")

def _safe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return transforms.ToTensor()(x)

def get_features(dataset, dataloader, model, device):
    all_features = []
    with torch.no_grad():
        if dataset == 'eurosat':
            for x in tqdm(dataloader):
                features = model(x['x_lb'].to(device))
                all_features.append(features.detach().cpu())
        else:
            for x, y in tqdm(dataloader):
                features = model(x.to(device))
                all_features.append(features.detach().cpu())

    return torch.cat(all_features).numpy()

def get_dataloaders(dataset, transform, batch_size, root_dir='data'):
    if transform is None:
        # just dummy resize -> both CLIP and DINO support 224 size of the image
        transform = get_default_transforms()
    train_dataset, val_dataset = get_datasets(dataset, transform, root_dir)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainloader, valloader


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
        train_dataset = dsets.STL10(data_path, split='train+unlabeled', transform=transform, download=False)
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

def get_default_transforms():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        _safe_to_tensor,
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])


phi_to_name = {'clipRN50': 'RN50', 'clipRN101': 'RN101', 'clipRN50x4': 'RN50x4', 'clipRN50x16': 'RN50x16', 'clipRN50x64': 'RN50x64',
                   'clipvitB32': 'ViT-B/32', 'clipvitB16': 'ViT-B/16', 'clipvitL14': 'ViT-L/14'}

def run(args=None):
    args = _parse_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.phis == 'dinov2L14':
        torch.hub.set_dir(os.path.join(args.root_dir, "checkpoints/dinov2"))
        model = torch.hub.load('./data/checkpoints/dinov2/', 'dinov2_vitl14', source='local').to(device)
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
        model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        preprocess = None

    if args.phis == 'dinov2B14':
        torch.hub.set_dir(os.path.join(args.root_dir, "checkpoints/dinov2"))
        model = torch.hub.load('./data/checkpoints/dinov2/', 'dinov2_vitb14', source='local').to(device)
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        preprocess = None

    else:
        ckpt_dir = os.path.join(args.root_dir, "checkpoints/clip")
        model, preprocess = clip.load(phi_to_name[args.phis], device=device, download_root=ckpt_dir)
        model.eval()
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        model = model.encode_image
        preprocess.transforms[2] = _convert_image_to_rgb
        preprocess.transforms[3] = _safe_to_tensor
    
    trainloader, valloader = get_dataloaders(args.dataset, preprocess, args.batch_size, args.root_dir)
    feats_train = get_features(args.dataset, trainloader, model, device)
    feats_val = get_features(args.dataset, valloader, model, device)

    representations_dir = f"{args.root_dir}/representations/{args.phis}"
    if not os.path.exists(representations_dir):
        os.makedirs(representations_dir)

    np.save(f'{representations_dir}/{args.dataset}_train.npy', feats_train)
    np.save(f'{representations_dir}/{args.dataset}_val.npy', feats_val)


if __name__ == '__main__':
    run()
