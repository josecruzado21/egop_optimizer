
import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.CIFAR10Classifier import CIFAR10_model_34_layer_residual
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader
from egop_optimizer.utils.device_utils import get_available_device
DEVICE = get_available_device()


if __name__ == "__main__":
    model = CIFAR10_model_34_layer_residual()
    model = model.to(DEVICE)
    data_dir = "/Users/jose_cruzado/Documents/Personal/Code/egop_optimizer/data"
    model.reinitialize_seeded(seed=0)
    trainloader, valloader, testloader = CIFAR10_dataloader(batch_size=256, data_dir=data_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    epochs = 20

    basic_train_loop(
        model=model,
        trainloader=trainloader,
        optimizer=optimizer,
        loss_method=loss_method,
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        experiment_name="CIFAR10_resnet34_OG",
        report_validation_metrics=True)
