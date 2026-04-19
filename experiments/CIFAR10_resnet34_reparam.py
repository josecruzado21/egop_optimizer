import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.CIFAR10Classifier import (
    CIFAR10_model_34_layer_residual,
    CIFAR10_model_34_layer_residual_reparam,
)
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer, layerwise_reparam_init_equiv


if __name__ == "__main__":
    data_dir = Path("/Users/jose_cruzado/Documents/Personal/Code/egop_optimizer/data/")

    trainloader, valloader, _ = CIFAR10_dataloader(
        batch_size=128,
        data_dir=data_dir,
        dev_split=0.99,
        augment=True,
    )

    OG_model = CIFAR10_model_34_layer_residual()
    OG_model = OG_model.to("mps")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    V_dict = compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        reparam_linear_layers=False,
    )

    reparam_model = CIFAR10_model_34_layer_residual_reparam(V_by_layer_dict=V_dict)

    _, reparam_model = layerwise_reparam_init_equiv(
        EGOP_model=reparam_model, OG_model=OG_model, seed=42
    )

    optimizer = torch.optim.AdamW(reparam_model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    epochs = 4

    basic_train_loop(
        model=reparam_model,
        trainloader=trainloader,
        optimizer=optimizer,
        loss_method=loss_method,
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        experiment_name="CIFAR10_resnet34_reparam",
        ten_crop=False,
        report_validation_metrics=True,
    )
