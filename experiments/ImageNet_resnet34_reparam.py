import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.ImagenetClassifier import (
    ImageNet_model_34_layer_residual,
    ImageNet_model_34_layer_residual_reparam,
)
from egop_optimizer.dataloaders.ImageNet_dataloader import ImageNet_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer, layerwise_reparam_init_equiv
from egop_optimizer.utils.device_utils import get_available_device
DEVICE = get_available_device()


if __name__ == "__main__":
    ten_crop = True
    data_dir = Path("/share/data/vdata/imagenet1k")
    trainloader, valloader = ImageNet_dataloader(root=data_dir, batch_size=256, ten_crop=ten_crop)

    OG_model = ImageNet_model_34_layer_residual()
    OG_model = OG_model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    V_dict = compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        reparam_linear_layers=False,
    )
    reparam_model = ImageNet_model_34_layer_residual_reparam(V_by_layer_dict=V_dict)
    reparam_model = reparam_model.to(DEVICE)
    _, reparam_model = layerwise_reparam_init_equiv(
        EGOP_model=reparam_model, OG_model=OG_model, seed=42
    )
    optimizer = torch.optim.AdamW(reparam_model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)
    epochs = 10
    basic_train_loop(
        model=reparam_model,
        trainloader=trainloader,
        optimizer=optimizer,
        loss_method=loss_method,
        epochs=epochs,
        LR_scheduler=None,
        valloader=valloader,
        experiment_name="ImageNet_resnet34_reparam",
        ten_crop=ten_crop,
        report_validation_metrics=True,
    )
