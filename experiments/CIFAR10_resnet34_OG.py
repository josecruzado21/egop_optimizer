import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.CIFAR10Classifier import CIFAR10_model_34_layer_residual
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader


if __name__ == "__main__":
    data_dir = Path("/Users/jose_cruzado/Documents/Personal/Code/egop_optimizer/data/")
    model = CIFAR10_model_34_layer_residual()
    trainloader, valloader, _ = CIFAR10_dataloader(batch_size=256,data_dir=data_dir,augment=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    epochs = 10

    basic_train_loop(
        model=model,
        trainloader=trainloader,
        optimizer=optimizer,
        loss_method=loss_method,
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        experiment_name="CIFAR10_resnet34_OG",
        ten_crop=False,
        report_validation_metrics=True,
    )
