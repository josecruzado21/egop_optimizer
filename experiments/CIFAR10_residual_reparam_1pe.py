import csv
import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.CIFAR10Classifier import (
    CIFAR10_model_residual,
    CIFAR10_model_residual_reparam,
)
from egop_optimizer.dataloaders.CIFAR10_dataloader import CIFAR10_dataloader
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer, layerwise_reparam_init_equiv
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

BASE_EXPERIMENT = "CIFAR10_resnet34_reparam"
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
    data_dir = Path("/Users/jose_cruzado/Documents/Personal/Code/egop_optimizer/data/")
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)
    epochs = 3

    OG_model = CIFAR10_model_residual().to(DEVICE)

    # V is computed once — it defines the reparameterization basis, not the weights.
    # A fixed dataloader seed is used so V is the same across all runs.
    trainloader_for_V, _, _ = CIFAR10_dataloader(
        batch_size=128, data_dir=data_dir, dev_split=0.99, augment=True, seed=0
    )
    V_dict = compute_V_by_layer(
        use_randomized_svd = True,
        model_OG=OG_model,
        k=100,
        data_loader=trainloader_for_V,
        criterion=torch.nn.CrossEntropyLoss(reduction="mean"),
        reparam_linear_layers=True,
    )
    reparam_model = CIFAR10_model_residual_reparam(V_by_layer_dict=V_dict).to(DEVICE)

    experiment_names = []
    for seed in SEEDS:
        experiment_name = f"{BASE_EXPERIMENT}_seed{seed}"
        experiment_names.append(experiment_name)

        if Path(f"logs/{experiment_name}/metrics.csv").exists():
            print(f"Skipping {experiment_name} — metrics.csv already exists.")
            continue

        # layerwise_reparam_init_equiv calls OG_model.reinitialize_seeded(seed)
        # internally, then transforms those weights into the reparam coordinate system —
        # so the reparam model starts from an initialization equivalent to OG seed{seed}.
        _, reparam_model = layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=seed
        )

        trainloader, valloader, _ = CIFAR10_dataloader(
            batch_size=128, data_dir=data_dir, dev_split=0.99, augment=True, seed=seed
        )
        optimizer = torch.optim.AdamW(reparam_model.parameters(), lr=1e-3, weight_decay=0.01)

        basic_train_loop(
            model=reparam_model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_method=loss_method,
            epochs=epochs,
            LR_scheduler=None,
            valloader=valloader,
            experiment_name=experiment_name,
            ten_crop=False,
            report_validation_metrics=True,
            V_recalc_freq=1,
            model_OG=OG_model,
            k=100,
            reparam_linear_layers=True,
        )

    consolidate_metrics(
        experiment_names=experiment_names,
        output_path=Path(f"logs/{BASE_EXPERIMENT}_all_seeds.csv"),
    )
