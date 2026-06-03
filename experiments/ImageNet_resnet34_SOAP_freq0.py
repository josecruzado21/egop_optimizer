import csv
import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.engine.soap import SOAP
from egop_optimizer.models.ImagenetClassifier import ImageNet_model_34_layer_residual
from egop_optimizer.dataloaders.ImageNet_dataloader import ImageNet_dataloader
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

BASE_EXPERIMENT = "ImageNet_resnet34_SOAP"
SEEDS = [1, 2, 3, 4, 5]


def consolidate_metrics(experiment_names: list, output_path: Path):
    rows = []
    fieldnames = None
    for name in experiment_names:
        csv_path = Path(f"logs/{name}/metrics.csv")
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Consolidated metrics saved to {output_path}")


if __name__ == "__main__":
    data_dir = Path("/share/data/vdata/imagenet1k")
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)
    ten_crop = True
    epochs = 100
    batch_size = 128

    model = ImageNet_model_34_layer_residual().to(DEVICE)

    experiment_names = []
    for seed in SEEDS:
        experiment_name = f"{BASE_EXPERIMENT}_seed{seed}"
        experiment_names.append(experiment_name)

        if Path(f"logs/{experiment_name}/metrics.csv").exists():
            print(f"Skipping {experiment_name} — metrics.csv already exists.")
            continue

        model.reinitialize_seeded(seed=seed)

        trainloader, valloader = ImageNet_dataloader(
            root=data_dir, batch_size=batch_size, ten_crop=ten_crop, seed=seed
        )
        optimizer = SOAP(
            model.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            betas=(0.95, 0.95),
            precondition_frequency=0,
            max_precond_dim=10000,
            merge_dims=True,
            precondition_1d=False,
            normalize_grads=False,
            data_format="channels_first",
        )
        basic_train_loop(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_method=loss_method,
            epochs=epochs,
            LR_scheduler=None,
            valloader=valloader,
            experiment_name=experiment_name,
            ten_crop=ten_crop,
            report_validation_metrics=True,
        )

    consolidate_metrics(
        experiment_names=experiment_names,
        output_path=Path(f"logs/{BASE_EXPERIMENT}_all_seeds.csv"),
    )
