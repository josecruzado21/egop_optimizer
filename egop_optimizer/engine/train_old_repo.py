"""
Training loop matching old repo's run_fashionMNIST_og_only.py logic:
- Training loss: running_loss / len(trainloader) (mean of batch means), computed during training
- Training acc: correct / total, computed during training
- Validation loss: same approach (mean of batch means) in eval() mode
- Validation acc: correct / total in eval() mode
- All using criterion = CrossEntropyLoss(reduction='mean')
"""

import os

import torch
import logging
import time
from tqdm.auto import tqdm

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def _evaluate(model, dataloader, criterion, device, ten_crop=False):
    """
    Evaluate model on a dataset in eval() mode.
    Returns (mean of batch mean losses, accuracy).
    Matches old repo's validation phase in run_fashionMNIST_og_only.py.
    """
    running_loss = 0.0
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            if ten_crop:
                bs, ncrops, c, h, w = batch_data.size()
                batch_data = batch_data.view(-1, c, h, w)
                output = model(batch_data)
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = model(batch_data)
            running_loss += criterion(output, batch_labels).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    model.train()
    acc = correct / total if total > 0 else 0
    return running_loss / len(dataloader), acc


def basic_train_loop(
    model,
    trainloader,
    optimizer,
    epochs,
    loss_method,
    LR_scheduler=None,
    valloader=None,
    testloader=None,
    device=DEVICE,
    experiment_name="default",
    ten_crop=False,
    report_validation_metrics=True,
):
    """
    Training loop matching old repo's run_fashionMNIST_og_only.py.

    Training loss and accuracy are computed during the training pass itself
    (not re-evaluated after). Uses reduction='mean' throughout.
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
    criterion = loss_method(reduction="mean")
    if LR_scheduler is not None:
        training_logger.error("Scheduler not yet supported.")
        raise Exception("Scheduler not yet supported.")

    for t in range(epochs):
        training_logger.info(f"Starting epoch {t}")
        train_start = time.time()

        # --- Training phase (matches old repo's train_og_model) ---
        model.train()
        running_loss = 0.0
        total_train, correct_train = 0, 0
        for batch_data, batch_labels in tqdm(trainloader, leave=False):
            if device is not None and (
                batch_data.device.type != device.type
                or batch_labels.device.type != device.type
            ):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = correct_train / total_train
        train_end = time.time()
        train_duration = round((train_end - train_start) / 60, 2)

        # --- Validation phase ---
        if report_validation_metrics and valloader is not None:
            val_start = time.time()
            val_loss, val_acc = _evaluate(model, valloader, criterion, device, ten_crop)
            val_end = time.time()
            val_duration = round((val_end - val_start) / 60, 2)
            training_logger.info(
                f"Epoch {t}: Training Loss = {train_loss:.10f}, Validation Loss = {val_loss:.10f}, "
                f"Training Acc. = {train_acc:.10f}, Validation Acc. = {val_acc:.10f}, "
                f"Training Time = {train_duration:.2f}m, Validation Time = {val_duration:.2f}m"
            )
        else:
            training_logger.info(
                f"Epoch {t}: Training Loss = {train_loss:.10f}, Training Acc. = {train_acc:.10f}, "
                f"Training Time = {train_duration:.2f}m"
            )

    # Final test evaluation
    if testloader is not None:
        test_loss, test_acc = _evaluate(model, testloader, criterion, device, ten_crop)
        training_logger.info(
            f"Final Test Loss = {test_loss:.10f}, Test Acc. = {test_acc:.10f}"
        )
