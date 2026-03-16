import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.FashionMNISTClassfier import FashionMNISTClassifier
from egop_optimizer.dataloaders.fashionMNIST_dataloader import fashionMNIST_dataloader


if __name__ == "__main__":

    hidden_size = 100
    pool_factor = 1
    batch_size = 300
    num_classes = 10
    epochs = 100
    lr = 0.005
    weight_decay = 0.1
    num_runs = 10

    for seed in range(num_runs) :
        print(f"\n{'='*50}")
        seed = seed+42
        print(f"Run {seed + 1}/{num_runs} (seed={seed})")
        print(f"{'='*50}\n")

        model = FashionMNISTClassifier(
            pool_factor=pool_factor,
            hidden_size=hidden_size,
            num_classes=num_classes,
            weight_dist='gaussian'
        )
        model.reinitialize_seeded(seed)
        trainloader, valloader, testloader = fashionMNIST_dataloader(
            batch_size=batch_size,
            num_classes=num_classes,
            dev_split=0.25,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

        basic_train_loop(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_method=loss_method,
            epochs=epochs,
            LR_scheduler=None,
            valloader=valloader,
            testloader=testloader,
            experiment_name=f"fashionMNIST_OG_seed{seed}",
            report_validation_metrics=True,
        )
