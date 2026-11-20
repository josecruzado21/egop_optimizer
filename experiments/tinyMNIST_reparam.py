import torch
import numpy as np
import os
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import (
    TinyMNISTClassifier,
    ReparamTinyMNISTClassifier,
)
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
    check_model_structure,
)
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader

import pdb

# ── Experiment configuration ──────────────────────────────────────────
# Optimizers and their "OG" (original / best-known) learning rates
# For main body results, we run reparameterized models at the learning rates
# selected by tuning in original coordinates. This comparison favors
# the methods in original coordinates while also making it easier to isolate
# the impact of reparameterization.
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

# EGOP experiment parameters
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


def get_V_dict_for_TinyMNIST(OG_model, use_randomized_svd):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return


if __name__ == "__main__":
    # Change working directory to experiments/
    os.chdir(Path(__file__).resolve().parent)

    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    trainloader, valloader, testloader = tinyMNIST_dataloader(
        batch_size=batch_size,
        train_val_test_split=train_val_test_split,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    for solver_name in solvers_to_run:
        lr = OG_LR[solver_name]
        for trial in range(num_trials):
            seed = trial
            experiment_name = f"tinyMNIST_reparam/{solver_name}/LR_{lr}/trial_{trial}"

            # Init. model in original coordinates for gradient sampling
            OG_model = TinyMNISTClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_classes=output_size,
                seed=seed,
            )
            V_dict = compute_V_by_layer(
                model_OG=OG_model,
                k=int(EGOP_oversampling_factor * OG_model.count_parameters()),
                data_loader=trainloader,
                criterion=criterion,
                use_randomized_svd=use_randomized_EVD,
            )
            # OG_model = TinyMNISTClassifier()
            # V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
            reparam_model = ReparamTinyMNISTClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_classes=output_size,
                seed=seed,
                V_by_layer_dict=V_dict,
            )
            # Init. OG model according to experiment seed
            OG_model.reinitialize_seeded(seed)
            OG_model, reparam_model = layerwise_reparam_init_equiv(
                EGOP_model=reparam_model, OG_model=OG_model, seed=seed
            )

            ## Uncomment to test with methods from test_init_equiv.py
            # equiv_test = SharedModelEquivalenceTester()
            # equiv_test.assert_models_are_equivalent(
            #     OG_model, reparam_model, verbose=True
            # )

            optimizer = make_optimizer(solver_name, reparam_model.parameters(), lr)

            basic_train_loop(
                model=reparam_model,
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
