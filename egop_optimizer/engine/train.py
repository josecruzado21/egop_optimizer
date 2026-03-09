import os

import torch
import logging
from datetime import datetime
import time
from tqdm.auto import tqdm
import pdb

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def compute_validation_loss(model, sum_loss_fn, valloader, device=DEVICE):
    total_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in valloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            output = model(batch_data)
            total_val_loss += sum_loss_fn(output, batch_labels).item()
    return total_val_loss / len(valloader.dataset)


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
):
    """
    loss_method should accept an argument: reduction
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
        training_logger.info(f"Starting epoch {t}")
        counter = 0
        train_start = time.time()
        for batch_data, batch_labels in tqdm(trainloader, leave=False):
            counter += 1
            if counter >100:
                break
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
            epoch_loss += batch_loss
        train_end = time.time()
        train_duration = train_end - train_start
        epoch_loss /= len(trainloader.dataset)

        # Eval model
        val_start = time.time()
        epoch_val_loss = compute_validation_loss(model, sum_loss_fn, valloader, device)
        val_end = time.time()
        val_duration = val_end - val_start
        training_logger.info(
            f"Epoch {t}: total loss = {epoch_loss:.2f}, val loss = {epoch_val_loss}, "
            f"train time = {train_duration:.2f}s, val time = {val_duration:.2f}s"
        )
        model.train()

    return
