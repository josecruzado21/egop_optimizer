from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

def ImageNet_uncached_dataloader(
    root: Path,
    num_classes: int = 1000,
    class_list: list = None,
    augment: bool = True,
    ten_crop: bool = False
) -> tuple:
    """
    Creates uncached ImageNet dataloaders for training and validation.

    Args:
        root (Path): Path to the ImageNet dataset root directory. Should contain 'train' and 'val' subfolders.
        num_classes (int, optional): Number of classes to use. Defaults to 1000.
        class_list (list, optional): List of class indices to include. If None, uses the first `num_classes` classes.
        augment (bool, optional): Whether to apply data augmentation to the training set. Defaults to True.
        ten_crop (bool, optional): Whether to use TenCrop for evaluation. Defaults to False.

    Returns:
        tuple: (train_dataset, val_dataset), which are either ImageFolder or Subset objects.
    """
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    if augment:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: F.resize(img, random.randint(256, 480))),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    if ten_crop:
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    else:
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    train_ImageNet = torchvision.datasets.ImageFolder(
        root=root / "train", transform=transform_train
    )
    val_ImageNet = torchvision.datasets.ImageFolder(
        root=root / "val", transform=transform_eval
    )

    do_filter = False
    if class_list is None and num_classes < 1000:
        class_list = torch.arange(num_classes).tolist()
        do_filter = True
    elif class_list is not None and len(class_list) < 1000:
        do_filter = True

    def get_filtered_indices(dataset, class_list):
        targets = torch.tensor(dataset.targets)
        class_set = torch.tensor(class_list)
        mask = torch.isin(targets, class_set)
        return mask.nonzero(as_tuple=True)[0].tolist()

    if do_filter:
        train_indices = get_filtered_indices(train_ImageNet, class_list)
        val_indices   = get_filtered_indices(val_ImageNet,   class_list)

        train_ImageNet = torch.utils.data.Subset(train_ImageNet, train_indices)
        val_ImageNet   = torch.utils.data.Subset(val_ImageNet,   val_indices)

    return train_ImageNet, val_ImageNet

def ImageNet_dataloader(
    batch_size: int = 256,
    num_classes: int = 1000,
    root: Path = None,
    class_list: list = None,
    augment: bool = True,
    seed: int = 42,
    num_workers: int = 1,
    prefetch_factor: int = 3,
    persistent_workers: bool = True,
    ten_crop: bool = False
) -> tuple:
    """
    Creates PyTorch DataLoaders for ImageNet training and validation sets.

    Args:
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 256.
        num_classes (int, optional): Number of classes to use. Defaults to 1000.
        root (Path, optional): Path to the ImageNet dataset root directory.
        class_list (list, optional): List of class indices to include. If None, uses the first `num_classes` classes.
        augment (bool, optional): Whether to apply data augmentation to the training set. Defaults to True.
        seed (int, optional): Random seed for shuffling. Defaults to 42.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 8.
        prefetch_factor (int, optional): Number of batches loaded in advance by each worker. Defaults to 3.
        persistent_workers (bool, optional): Whether to keep data loader workers alive. Defaults to True.
        ten_crop (bool, optional): Whether to use TenCrop for evaluation. Defaults to False.

    Returns:
        tuple: (train_loader, dev_loader) - DataLoader objects for training and validation.
    """
    if root is None:
        raise ValueError("The 'root' argument must be specified as a Path to the ImageNet dataset.")

    trainset, devset = ImageNet_uncached_dataloader(
        num_classes=num_classes,
        root=root,
        class_list=class_list,
        augment=augment,
        ten_crop=ten_crop
    )

    g = torch.Generator().manual_seed(seed)

    def make_loader(subset, bsize, shuffle):
        return torch.utils.data.DataLoader(
            subset,
            batch_size=bsize if bsize is not None else len(subset),
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=g,
        )

    train_loader = make_loader(trainset, batch_size, shuffle=True)
    dev_loader   = make_loader(devset,   batch_size, shuffle=False)
    return train_loader, dev_loader