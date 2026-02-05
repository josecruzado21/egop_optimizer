import unittest
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


class TestCIFAR10Dataloaders(unittest.TestCase):
    def test_stratified_split_basic(self):
        dataset = [(None, label) for label in [0] * 10 + [1] * 10 + [2] * 10]
        group1_perc = 0.6
        group1_idx, group2_idx = stratified_split(
            dataset, group1_perc=group1_perc, seed=123
        )
        group1_labels = [dataset[i][1] for i in group1_idx]
        group2_labels = [dataset[i][1] for i in group2_idx]
        self.assertEqual(group1_labels.count(0), 6)
        self.assertEqual(group1_labels.count(1), 6)
        self.assertEqual(group1_labels.count(2), 6)
        self.assertEqual(group2_labels.count(0), 4)
        self.assertEqual(group2_labels.count(1), 4)
        self.assertEqual(group2_labels.count(2), 4)

    def test_cifar10_cache_dataset_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "cached"
            raw_data_dir = Path(tmpdir) / "raw"
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            CIFAR10_cache_dataset(
                data_dir=raw_data_dir, save_dir=save_dir, verbose=False, delete_raw=False
            )
            train_file = save_dir / "train" / "train.pt"
            test_file = save_dir / "test" / "test.pt"
            self.assertTrue(train_file.exists())
            self.assertTrue(test_file.exists())
            train_images, train_labels = torch.load(train_file, weights_only=True)
            test_images, test_labels = torch.load(test_file, weights_only=True)
            self.assertEqual(train_images.shape[1:], (3, 32, 32))
            self.assertEqual(train_labels.ndim, 1)
            self.assertEqual(test_images.shape[1:], (3, 32, 32))
            self.assertEqual(test_labels.ndim, 1)

    def test_CIFAR10_cached_dataloader_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_data_dir = Path(tmpdir) / "raw"
            cached_data_dir = Path(tmpdir) / "cached"
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            cached_data_dir.mkdir(parents=True, exist_ok=True)
            CIFAR10_cache_dataset(
                data_dir=raw_data_dir,
                save_dir=cached_data_dir,
                verbose=False,
                delete_raw=False,
            )
            train, dev, test = CIFAR10_cached_dataloader(
                num_classes=10,
                data_dir=cached_data_dir,
                dev_split=0.5,
                class_list=None,
                use_stratified_split=False,
                seed=42,
            )
            self.assertIsInstance(train, torch.utils.data.TensorDataset)
            self.assertIsInstance(dev, torch.utils.data.TensorDataset)
            self.assertIsInstance(test, torch.utils.data.TensorDataset)
            self.assertEqual(train.tensors[0].shape[1:], (3, 32, 32))
            self.assertEqual(dev.tensors[0].shape[1:], (3, 32, 32))
            self.assertEqual(test.tensors[0].shape[1:], (3, 32, 32))

    def test_CIFAR10_uncached_dataloader_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
            self.assertIsInstance(train, torch.utils.data.Dataset)
            self.assertIsInstance(dev, torch.utils.data.Dataset)
            self.assertIsInstance(test, torch.utils.data.Dataset)
            self.assertGreater(len(train), 0)
            self.assertGreater(len(dev), 0)
            self.assertGreater(len(test), 0)

    def test_CIFAR10_dataloader_basic(self):
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
                num_workers=0,
                use_cached=True,
            )
            self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
            self.assertIsInstance(dev_loader, torch.utils.data.DataLoader)
            self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
            train_batch = next(iter(train_loader))
            self.assertEqual(train_batch[0].shape[1:], (3, 32, 32))
            self.assertEqual(train_batch[1].ndim, 1)


if __name__ == "__main__":
    unittest.main()