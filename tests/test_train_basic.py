import torch

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader


if __name__ == "__main__":
    model = TinyMNISTClassifier()
    trainloader, valloader, _ = tinyMNIST_dataloader(batch_size=128)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
    )
