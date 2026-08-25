import torch
import tempfile
from pathlib import Path

from egop_optimizer.models.CifarClassifier import TinyCifarClassifier
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader
from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def load_cifar_tmp_files():
    # Locate raw_data folder, assumed to be in same folder as egop_optimizer
    # base = Path(__file__).resolve()
    # data_dir = base.parents[2] / "raw_data" / "Cifar10"
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        return CIFAR10_dataloader(
            # use large batches for faster testing
            batch_size=100,
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


if __name__ == "__main__":
    train_loader, val_loader, _ = load_cifar_tmp_files()
    model = TinyCifarClassifier().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    epochs = 1

    basic_train_loop(
        model=model,
        trainloader=train_loader,
        optimizer=optimizer,
        loss_method=loss_method,
        epochs=epochs,
        LR_scheduler=None,
        valloader=val_loader,
        device=DEVICE,
    )
