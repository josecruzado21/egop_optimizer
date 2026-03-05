import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.ImagenetClassifier import ImageNet_model_34_layer_residual
from egop_optimizer.dataloaders.ImageNet_dataloader import ImageNet_dataloader


if __name__ == "__main__":
    model = ImageNet_model_34_layer_residual()

    # ImageNet dataset path (cluster)
    data_dir = Path("/share/data/vdata/imagenet1k")
    trainloader, valloader = ImageNet_dataloader(root=data_dir, batch_size=256)

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
        experiment_name="ImageNet_resnet34_OG",
    )
