import pytest
from egop_optimizer.dataloaders.CIFAR10_dataloader import (
    stratified_split,
    CIFAR10_cache_dataset,
    CIFAR10_cached_dataloader,
    CIFAR10_uncached_dataloader,
    CIFAR10_dataloader,
)
import tempfile
import torch
from pathlib import Path


def test_stratified_split_basic():
    dataset = [(None, label) for label in [0] * 10 + [1] * 10 + [2] * 10]
    group1_perc = 0.6
    group1_idx, group2_idx = stratified_split(
        dataset, group1_perc=group1_perc, seed=123
    )
    group1_labels = [dataset[i][1] for i in group1_idx]
    group2_labels = [dataset[i][1] for i in group2_idx]
    # Check class counts
    assert group1_labels.count(0) == 6
    assert group1_labels.count(1) == 6
    assert group1_labels.count(2) == 6
    assert group2_labels.count(0) == 4
    assert group2_labels.count(1) == 4
    assert group2_labels.count(2) == 4


def test_cifar10_cache_dataset_creates_files():
    # Use a temporary directory for both raw data and cached data
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "cached"
        raw_data_dir = Path(tmpdir) / "raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        CIFAR10_cache_dataset(
            data_dir=raw_data_dir, save_dir=save_dir, verbose=False, delete_raw=False
        )
        # Check that train.pt and test.pt exist
        train_file = save_dir / "train" / "train.pt"
        test_file = save_dir / "test" / "test.pt"
        assert train_file.exists()
        assert test_file.exists()
        # Check that the saved files contain tensors
        train_images, train_labels = torch.load(train_file, weights_only=True)
        test_images, test_labels = torch.load(test_file, weights_only=True)
        assert train_images.shape[1:] == (3, 32, 32)
        assert train_labels.ndim == 1
        assert test_images.shape[1:] == (3, 32, 32)
        assert test_labels.ndim == 1


def test_CIFAR10_cached_dataloader_basic():
    # Create temporary directories for raw and cached data
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_data_dir = Path(tmpdir) / "raw"
        cached_data_dir = Path(tmpdir) / "cached"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        cached_data_dir.mkdir(parents=True, exist_ok=True)

        # Generate cached data
        CIFAR10_cache_dataset(
            data_dir=raw_data_dir,
            save_dir=cached_data_dir,
            verbose=False,
            delete_raw=False,
        )

        # Load cached dataloader
        train, dev, test = CIFAR10_cached_dataloader(
            num_classes=10,
            data_dir=cached_data_dir,
            dev_split=0.5,
            class_list=None,
            use_stratified_split=False,
            seed=42,
        )

        # Check types and shapes
        assert isinstance(train, torch.utils.data.TensorDataset)
        assert isinstance(dev, torch.utils.data.TensorDataset)
        assert isinstance(test, torch.utils.data.TensorDataset)
        assert train.tensors[0].shape[1:] == (3, 32, 32)
        assert dev.tensors[0].shape[1:] == (3, 32, 32)
        assert test.tensors[0].shape[1:] == (3, 32, 32)


def test_CIFAR10_uncached_dataloader_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a temporary directory for raw data
        data_dir = Path(tmpdir)
        train, dev, test = CIFAR10_uncached_dataloader(
            num_classes=10,
            data_dir=data_dir,
            dev_split=0.5,
            class_list=None,
            augment=False,
            use_stratified_split=False,
            seed=42,
        )
        # Check types
        assert isinstance(train, torch.utils.data.Dataset)
        assert isinstance(dev, torch.utils.data.Dataset)
        assert isinstance(test, torch.utils.data.Dataset)
        # Check lengths
        assert len(train) > 0
        assert len(dev) > 0
        assert len(test) > 0


def test_CIFAR10_dataloader_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        train_loader, dev_loader, test_loader = CIFAR10_dataloader(
            batch_size=16,
            num_classes=10,
            data_dir=data_dir,
            dev_split=0.5,
            class_list=None,
            augment=False,
            use_stratified_split=False,
            seed=42,
            num_workers=0,  # For testing, use single worker
            use_cached=True,
        )
        # Check types
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(dev_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)
        # Check batch shapes
        train_batch = next(iter(train_loader))
        assert train_batch[0].shape[1:] == (3, 32, 32)
        assert train_batch[1].ndim == 1
