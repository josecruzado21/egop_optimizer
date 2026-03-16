import torch
import numpy as np

# from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.engine.train_old_repo import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader


# ── Experiment configuration ──────────────────────────────────────────
# Optimizers and their "OG" (original / best-known) learning rates
OG_LR = {
    "Adam": 0.02,
    "Adagrad": 0.04,
    "GD": 0.04,
    "GD_w_momentum": 0.004,
}

solvers_to_run = ["Adam", "Adagrad", "GD", "GD_w_momentum"]

# Training
epochs = 200
num_trials = 10
batch_size = 300
num_batches = None  # unused for now

# Data
train_val_test_split = [None, 0.33, 0.66]

# Model
input_size = 64
hidden_sizes = [32]
output_size = 10
use_bias = True
weight_init_dist = "xavier_normal"
bias_dist = "uniform"

# EGOP (recorded for reference; not used in OG baseline)
EGOP_oversampling_factor = 1.0
weight_sampling_dist = "xavier"
EGOP_batch_size = 300
use_randomized_EVD = False


def make_optimizer(solver_name, parameters, lr):
    if solver_name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif solver_name == "Adagrad":
        return torch.optim.Adagrad(parameters, lr=lr)
    elif solver_name == "GD":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.0)
    elif solver_name == "GD_w_momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


if __name__ == "__main__":
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    trainloader, valloader, testloader = tinyMNIST_dataloader(
        batch_size=batch_size,
        train_val_test_split=train_val_test_split,
    )

    for solver_name in solvers_to_run:
        lr = OG_LR[solver_name]
        for trial in range(num_trials):
            seed = trial
            experiment_name = (
                f"tinyMNIST_OG/{solver_name}/LR_{lr}/trial_{trial}"
            )

            model = TinyMNISTClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_classes=output_size,
                seed=seed,
            )
            model.reinitialize_seeded(seed)

            optimizer = make_optimizer(solver_name, model.parameters(), lr)

            basic_train_loop(
                model=model,
                trainloader=trainloader,
                optimizer=optimizer,
                epochs=epochs,
                loss_method=loss_method,
                LR_scheduler=None,
                valloader=valloader,
                experiment_name=experiment_name,
                ten_crop=False,
                report_validation_metrics=True,
            )
