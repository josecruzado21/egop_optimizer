import os

import torch
import logging
from datetime import datetime
import time
from tqdm.auto import tqdm
import pdb

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def compute_validation_loss(model, sum_loss_fn, valloader, device=DEVICE, ten_crop=False):
    """
    Evaluates the model on a validation dataset and computes the average loss and accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        sum_loss_fn (callable): Loss function with reduction='sum' for aggregating batch losses.
        valloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device, optional): Device to run evaluation on. Defaults to DEVICE.
        ten_crop (bool, optional): If True, applies 10-crop evaluation for image data. Defaults to False.

    Returns:
        tuple: (average validation loss per sample, validation accuracy)
    """
    total_val_loss = 0.0
    total_val, correct_val = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in valloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            if ten_crop:
                bs, ncrops, c, h, w = batch_data.size()
                batch_data = batch_data.view(-1, c, h, w)
                output = model(batch_data)
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = model(batch_data)
            total_val_loss += sum_loss_fn(output, batch_labels).item()
            _, predicted = torch.max(output, 1)
            correct_val += (predicted == batch_labels).sum().item()
            total_val += batch_labels.size(0)
    val_acc = correct_val / total_val if total_val > 0 else 0
    return total_val_loss / len(valloader.dataset), val_acc


def basic_train_loop(
    model,
    trainloader,
    optimizer,
    epochs,
    loss_method,
    LR_scheduler=None,
    valloader=None,
    device=DEVICE,
    experiment_name="default",
    ten_crop = False,
    report_validation_metrics = True,
):
    """
    Runs a basic training loop for a PyTorch model, with optional validation and logging.

    Args:
        model (torch.nn.Module): The model to train.
        trainloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        epochs (int): Number of training epochs.
        loss_method (callable): Loss function constructor, must accept a 'reduction' argument.
        LR_scheduler (optional): Learning rate scheduler (currently not supported).
        valloader (DataLoader, optional): DataLoader for the validation dataset.
        device (torch.device, optional): Device to use for training. Defaults to DEVICE.
        experiment_name (str, optional): Name for the experiment/log directory. Defaults to "default".
        ten_crop (bool, optional): If True, uses 10-crop evaluation for validation. Defaults to False.
        report_validation_metrics (bool, optional): If True, computes and logs validation metrics. Defaults to True.

    Logs:
        - Training and validation loss and accuracy per epoch.
        - Training and validation time per epoch.
        - Experiment setup information.

    Returns:
        None
    """
    log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    info_log_path = os.path.join(log_dir, "info.log")
    training_log_path = os.path.join(log_dir, "training.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Info logger (experiment setup)
    info_logger = logging.getLogger("info_logger")
    info_logger.setLevel(logging.INFO)
    info_handler = logging.FileHandler(info_log_path)
    info_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    info_logger.handlers = [info_handler]

    # Training logger (training progress)
    training_logger = logging.getLogger("training_logger")
    training_logger.setLevel(logging.INFO)
    training_handler = logging.FileHandler(training_log_path)
    training_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    training_logger.handlers = [training_handler, logging.StreamHandler()]

    # Log experiment info
    info_logger.info(f"Experiment: {experiment_name}")
    info_logger.info(f"Model: {model.__class__.__name__}")
    info_logger.info(f"Model architecture:\n{model}")
    info_logger.info(f"Optimizer: {optimizer}")
    info_logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    info_logger.info(f"Epochs: {epochs}")
    info_logger.info(f"Loss method: {loss_method}")
    info_logger.info(f"Scheduler: {LR_scheduler}")
    info_logger.info(f"Device: {device}")

    if device is not None:
        model = model.to(device)
    ave_loss_fn = loss_method(reduction="mean")
    sum_loss_fn = loss_method(reduction="sum")
    if LR_scheduler is not None:
        training_logger.error("Scheduler not yet supported.")
        raise Exception("Scheduler not yet supported.")

    for t in range(epochs):
        epoch_loss = 0
        total_train, correct_train = 0, 0
        training_logger.info(f"Starting epoch {t}")
        train_start = time.time()
        model.train()
        for batch_data, batch_labels in tqdm(trainloader, leave=False):
            if device is not None and (
                batch_data.device.type != device.type
                or batch_labels.device.type != device.type
            ):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            batch_loss = ave_loss_fn(output, batch_labels)
            batch_loss.backward()
            optimizer.step()
            sumloss = batch_loss.item() * batch_data.size(0)
            epoch_loss += sumloss
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == batch_labels).sum().item()
            total_train += batch_labels.size(0)
        train_end = time.time()
        train_duration = round((train_end - train_start)/60, 2)
        epoch_loss /= len(trainloader.dataset)
        train_acc = correct_train / total_train

        # Eval model
        if report_validation_metrics and valloader is not None:
            val_start = time.time()
            epoch_val_loss, val_acc = compute_validation_loss(model, sum_loss_fn, valloader, device, ten_crop)
            val_end = time.time()
            val_duration = round((val_end - val_start)/60, 2)
            training_logger.info(
                f"Epoch {t}: Training Loss = {epoch_loss:.2f}, Validation Loss = {epoch_val_loss:.2f}, "
                f"Training Acc. = {train_acc:.4f}, Validation Acc. = {val_acc:.4f}, "
                f"Training Time = {train_duration:.2f}m, Validation Time = {val_duration:.2f}m"
            )
        else:
            training_logger.info(
                f"Epoch {t}: Training Loss = {epoch_loss:.2f}, Training Acc. = {train_acc:.4f}, "
                f"Training Time = {train_duration:.2f}m"
            )
