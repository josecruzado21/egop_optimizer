import random
from collections import defaultdict
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import shutil
from typing import Union
from egop_optimizer.dataloaders.utils import stratified_split
import logging


def CIFAR10_cache_dataset(
    data_dir: str,
    save_dir: str = None,
    verbose: bool = True,
    delete_raw: bool = False,
) -> None:
    """
    Caches the CIFAR-10 dataset as tensors after applying standard normalization and transformations.

    Args:
        data_dir (str): Directory containing (or to download) raw CIFAR-10 data.
        save_dir (str): Directory to save cached data. If None, uses current working directory.
        verbose (bool): If True, logs progress information.
        delete_raw (bool): If True, deletes the original raw CIFAR-10 data after caching.

    Returns:
        None
    """
    logger = logging.getLogger("egop_optimizer.CIFAR10_cache_dataset")
    data_dir = Path(data_dir)
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    if verbose:
        logger.info("-" * 50)
        logger.info(f"Creating cached CIFAR-10 data in {save_dir}")
        logger.info(f"Using raw data directory: {data_dir}")
        logger.info("-" * 50 + "\n")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Standard CIFAR-10 transformation and normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    for split in ["train", "test"]:
        split_dir = save_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        cifar10_split = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=(split == "train"),
            transform=transform,
            download=True,
        )
        images = torch.stack([cifar10_split[i][0] for i in range(len(cifar10_split))])
        labels = torch.tensor(cifar10_split.targets)
        torch.save(
            (images, labels),
            split_dir / f"{split}.pt",
        )

    # Optionally remove raw CIFAR-10 data to save disk space
    if delete_raw:
        raw_data_path = data_dir / "CIFAR10"
        shutil.rmtree(raw_data_path)
        if verbose:
            logger.info(f"Deleted raw CIFAR-10 data at {raw_data_path}")
    if verbose:
        logger.info("Done caching.")
    return


def CIFAR10_cached_dataloader(
    num_classes: int = 10,
    data_dir: str = None,
    dev_split: float = 0.5,
    class_list: list = None,
    use_stratified_split: bool = False,
    seed: int = 42,
):
    """
    Loads cached CIFAR-10 tensors, applies optional class filtering, and splits test set into dev/test.

    Args:
        num_classes (int): Number of classes to include (default: 10).
        data_dir (str): Directory containing cached CIFAR-10 data.
        dev_split (float): Proportion of test set to use for dev set.
        class_list (list): List of class indices to include. If None, uses num_classes.
        use_stratified_split (bool): If True, splits test set stratified by class.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_CIFAR10, dev_CIFAR10, test_CIFAR10) as TensorDataset objects.
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified for cached dataloader.")
    data_dir = Path(data_dir)

    # 1. Load cached tensors
    train_images, train_labels = torch.load(
        data_dir / "train" / "train.pt", weights_only=True
    )
    test_images, test_labels = torch.load(
        data_dir / "test" / "test.pt", weights_only=True
    )

    # 2. Filter by class
    if class_list is None and num_classes < 10:
        class_list = torch.arange(num_classes)
        filter_classes = True
    elif class_list is not None and len(class_list) < 10:
        filter_classes = True
    else:
        filter_classes = False

    def filter_by_class(images, labels, class_list):
        mask = torch.isin(labels, torch.tensor(class_list))
        return images[mask], labels[mask]

    if filter_classes:
        train_images, train_labels = filter_by_class(
            train_images, train_labels, class_list
        )
        test_images, test_labels = filter_by_class(test_images, test_labels, class_list)

    # 3. Stratified or random split of test
    if use_stratified_split:
        dev_idx, test_idx = stratified_split(test_labels, dev_split, seed)
    else:
        num_dev = int(dev_split * len(test_labels))
        all_idx = list(range(len(test_labels)))
        dev_idx = all_idx[:num_dev]
        test_idx = all_idx[num_dev:]

    dev_images, dev_labels = test_images[dev_idx], test_labels[dev_idx]
    test_images, test_labels = test_images[test_idx], test_labels[test_idx]

    # 4. Wrap in DataSets
    train_CIFAR10 = TensorDataset(train_images, train_labels)
    dev_CIFAR10 = TensorDataset(dev_images, dev_labels)
    test_CIFAR10 = TensorDataset(test_images, test_labels)

    return train_CIFAR10, dev_CIFAR10, test_CIFAR10


def CIFAR10_uncached_dataloader(
    num_classes: int = 10,
    data_dir: str = None,
    dev_split: float = 0.5,
    class_list: list = None,
    augment: bool = False,
    use_stratified_split: bool = False,
    seed: int = 42,
):
    """
    Loads the CIFAR-10 dataset directly from disk, applies optional augmentation and class filtering,
    and splits the test set into dev/test subsets.

    Args:
        num_classes (int): Number of classes to include (default: 10).
        data_dir (str): Directory containing raw CIFAR-10 data.
        dev_split (float): Proportion of test set to use for dev set.
        class_list (list): List of class indices to include. If None, uses num_classes.
        augment (bool): If True, applies data augmentation to training set.
        use_stratified_split (bool): If True, splits test set stratified by class.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_CIFAR10, dev_CIFAR10, test_CIFAR10) as Dataset or Subset objects.
    """
    if data_dir is None:
        raise ValueError("data_dir must be specified for uncached dataloader.")
    data_dir = Path(data_dir)

    # 1. Define transformations
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
    )
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    # 2. Load datasets
    train_CIFAR10 = torchvision.datasets.CIFAR10(
        root=data_dir / "raw_data" / "CIFAR10",
        train=True,
        transform=transform_train,
        download=True,
    )
    test_CIFAR10 = torchvision.datasets.CIFAR10(
        root=data_dir / "raw_data" / "CIFAR10",
        train=False,
        transform=transform_test,
        download=True,
    )

    # 3. Filter by class
    if class_list is None and num_classes < 10:
        class_list = torch.arange(num_classes)
        filter_classes = True
    elif class_list is not None and len(class_list) < 10:
        filter_classes = True
    else:
        filter_classes = False

    def get_filtered_indices(dataset, class_list):
        targets = torch.tensor(dataset.targets)
        class_set = torch.tensor(class_list)
        mask = torch.isin(targets, class_set)
        return mask.nonzero(as_tuple=True)[0].tolist()

    if filter_classes:
        train_indices = get_filtered_indices(train_CIFAR10, class_list)
        test_indices = get_filtered_indices(test_CIFAR10, class_list)
        train_CIFAR10 = torch.utils.data.Subset(train_CIFAR10, train_indices)
        test_CIFAR10 = torch.utils.data.Subset(test_CIFAR10, test_indices)

    # 4. Split test into dev/test
    if use_stratified_split:
        dev_indices, test_indices = stratified_split(
            test_CIFAR10, group1_perc=dev_split, seed=seed
        )
        dev_CIFAR10 = torch.utils.data.Subset(test_CIFAR10, dev_indices)
        test_CIFAR10 = torch.utils.data.Subset(test_CIFAR10, test_indices)
    else:
        num_dev = int(dev_split * len(test_CIFAR10))
        dev_CIFAR10 = torch.utils.data.Subset(test_CIFAR10, list(range(0, num_dev)))
        test_CIFAR10 = torch.utils.data.Subset(
            test_CIFAR10, list(range(num_dev, len(test_CIFAR10)))
        )

    return train_CIFAR10, dev_CIFAR10, test_CIFAR10


def CIFAR10_dataloader(
    batch_size: int = 128,
    num_classes: int = 10,
    data_dir: str = None,
    dev_split: float = 0.5,
    class_list: list = None,
    augment: bool = False,
    use_stratified_split: bool = False,
    seed: int = 42,
    num_workers: int = 2,
    use_cached: bool = True,
    prefetch_factor: int = None,
    persistent_workers: bool = True,
):
    """
    Returns PyTorch DataLoaders for CIFAR-10 train, dev, and test sets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_classes (int): Number of classes to include.
        data_dir (str): Root directory for raw/cached CIFAR-10 data.
        dev_split (float): Proportion of test set to use for dev set.
        class_list (list): List of class indices to include. If None, uses num_classes.
        augment (bool): If True, applies data augmentation to training set.
        use_stratified_split (bool): If True, splits test set stratified by class.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of worker processes for DataLoader.
        use_cached (bool): If True, uses cached tensors; else loads raw data.
        prefetch_factor (int): Prefetch factor for DataLoader.
        persistent_workers (bool): If True, keeps workers alive between epochs.

    Returns:
        tuple: (train_loader, dev_loader, test_loader) as PyTorch DataLoader objects.
    """
    logger = logging.getLogger("egop_optimizer.CIFAR10_dataloader")
    if data_dir is None:
        raise ValueError("data_dir must be specified for CIFAR10_dataloader.")
    data_dir = Path(data_dir)

    if use_cached:
        save_dir_cached = data_dir / "cached_data" / "cached_CIFAR10"
        # Only create cached data if it doesn't exist or is empty
        if not (
            (save_dir_cached / "train" / "train.pt").exists()
            and (save_dir_cached / "test" / "test.pt").exists()
        ):
            logger.info(
                f"use_cached=True but cached data not found at {save_dir_cached}. Generating cached data now."
            )
            CIFAR10_cache_dataset(data_dir=data_dir, save_dir=save_dir_cached)
        trainset, devset, testset = CIFAR10_cached_dataloader(
            num_classes=num_classes,
            data_dir=save_dir_cached,
            dev_split=dev_split,
            class_list=class_list,
            use_stratified_split=use_stratified_split,
            seed=seed,
        )
    else:
        trainset, devset, testset = CIFAR10_uncached_dataloader(
            num_classes=num_classes,
            data_dir=data_dir,
            dev_split=dev_split,
            class_list=class_list,
            augment=augment,
            use_stratified_split=use_stratified_split,
            seed=seed,
        )

    g = torch.Generator()
    g.manual_seed(seed)

    def make_loader(subset, bsize):
        loader_kwargs = {
            "batch_size": bsize if bsize is not None else len(subset),
            "shuffle": True,
            "pin_memory": True,
            "num_workers": num_workers,
            "generator": g,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = persistent_workers
        return torch.utils.data.DataLoader(subset, **loader_kwargs)

    train_loader = make_loader(trainset, batch_size)
    dev_loader = make_loader(devset, batch_size)
    test_loader = make_loader(testset, batch_size)
    return train_loader, dev_loader, test_loader
