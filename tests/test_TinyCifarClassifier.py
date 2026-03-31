import unittest
import torch

import tempfile
from pathlib import Path

import pdb

from egop_optimizer.models.CifarClassifier import TinyCifarClassifier
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader
from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def load_cifar_debugging_subset():
    # Locate raw_data folder, assumed to be in same folder as egop_optimizer
    base = Path(__file__).resolve()
    data_dir = base.parents[2] / "raw_data" / "Cifar10"
    return CIFAR10_dataloader(
        # use large batches for faster testing
        batch_size=100,
        # use a single class for faster testing
        num_classes=1,
        data_dir=data_dir,
        # use a tiny percent of data for faster testing
        dev_split=0.01,
        class_list=None,
        augment=False,
        use_stratified_split=False,
        seed=42,
        num_workers=0,
        use_cached=True,
    )
    return


class TestBasic(unittest.TestCase):
    # def test_training_loop(self):
    #     """
    #     Test training loop execution. This takes a long time even with a small subset.
    #     """
    #     train_loader, val_loader, _ = load_cifar_debugging_subset()
    #     model = TinyCifarClassifier()
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    #     loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    #     epochs = 1

    #     basic_train_loop(
    #         model=model,
    #         trainloader=train_loader,
    #         optimizer=optimizer,
    #         loss_method=loss_method,
    #         epochs=epochs,
    #         LR_scheduler=None,
    #         valloader=val_loader,
    #     )
    #     return

    def test_train_steps(self, device=DEVICE):
        """
        Test a forward map and backprop, without running a whole basic_train_loop
        """
        train_loader, _, _ = load_cifar_debugging_subset()
        model = TinyCifarClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ave_loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        # Iterate over only 10 batches for testing
        batch_idx = 0
        for batch_data, batch_labels in train_loader:
            if batch_idx > 10:
                break
            if device is not None and (
                batch_data.device.type != device.type
                or batch_labels.device.type != device.type
            ):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                    device
                )
            optimizer.zero_grad()
            output = model(batch_data)
            batch_loss = ave_loss_fn(output, batch_labels)
            batch_loss.backward()
            optimizer.step()

            batch_idx += 1

        return


if __name__ == "__main__":
    unittest.main()
