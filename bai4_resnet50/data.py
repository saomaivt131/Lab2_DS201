from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(img_size):
    # Train có augment, val/test không
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_train_val_loaders(data_root, img_size, batch_size,
                            val_ratio=0.1, num_workers=2, seed=42):
    root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size)

    if (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
        classes = train_ds.classes
    else:
        # Không có thư mục val thì tách một phần từ train
        # Hai ImageFolder trỏ cùng thư mục để val không bị augment
        full_train = datasets.ImageFolder(root / "train", transform=train_tf)
        full_eval = datasets.ImageFolder(root / "train", transform=eval_tf)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(full_train), generator=g).tolist()
        n_val = int(len(idx) * val_ratio)
        train_ds = Subset(full_train, idx[n_val:])
        val_ds = Subset(full_eval, idx[:n_val])
        classes = full_train.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    return train_dl, val_dl, classes


def build_test_loader(data_root, img_size, batch_size, num_workers=2):
    _, eval_tf = build_transforms(img_size)
    test_ds = datasets.ImageFolder(Path(data_root) / "test", transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    return test_dl, test_ds.classes
