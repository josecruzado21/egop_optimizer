import os
import torch
import logging
from datetime import datetime
import time
from tqdm.auto import tqdm
import pdb

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    scheduler=None,
    metrics=None
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint.update(metrics)

    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, scheduler=None, device=DEVICE):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    train_losses = checkpoint.get("train_losses", [])
    train_accuracies = checkpoint.get("train_accuracies", [])
    train_times = checkpoint.get("train_times", [])
    val_losses = checkpoint.get("val_losses", [])
    val_accuracies = checkpoint.get("val_accuracies", [])
    val_times = checkpoint.get("val_times", [])

    return start_epoch, train_losses, train_accuracies, train_times, val_losses, val_accuracies, val_times

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
    checkpoint = True,
    initial_metrics = True,
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
    suffix = 0
    while os.path.exists(log_dir):
        suffix += 1
        log_dir = f"logs/{experiment_name}_cont{suffix}"
    os.makedirs(log_dir)
    if checkpoint:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
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

    start_epoch = 0
    train_losses = []
    train_accuracies = []
    train_times = []
    val_losses = []
    val_accuracies = []
    val_times = []

    if checkpoint and os.path.exists(latest_ckpt):
        start_epoch, train_losses, train_times, train_accuracies, val_losses, val_accuracies, val_times = load_checkpoint(
            model,
            optimizer,
            latest_ckpt,
            scheduler=LR_scheduler,
            device=device
        )

        training_logger.info(f"Resuming training from epoch {start_epoch}")

    if device is not None:
        model = model.to(device)
    ave_loss_fn = loss_method(reduction="mean")
    sum_loss_fn = loss_method(reduction="sum")
    if LR_scheduler is not None:
        training_logger.error("Scheduler not yet supported.")
        raise Exception("Scheduler not yet supported.")
    
    # --- Initial metrics reporting ---
    if initial_metrics:
        model.train()
        with torch.no_grad():
            total_train_loss = 0.0
            total_train, correct_train = 0, 0
            for batch_data, batch_labels in trainloader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = model(batch_data)
                total_train_loss += sum_loss_fn(output, batch_labels).item()
                _, predicted = torch.max(output, 1)
                correct_train += (predicted == batch_labels).sum().item()
                total_train += batch_labels.size(0)
        initial_train_loss = total_train_loss / len(trainloader.dataset)
        initial_train_acc = correct_train / total_train if total_train > 0 else 0

        training_logger.info(
            f"Initial Training Loss = {initial_train_loss:.2f}, Initial Training Acc. = {initial_train_acc:.4f}"
        )

        # Compute initial validation metrics if valloader is provided
        if valloader is not None:
            initial_val_loss, initial_val_acc = compute_validation_loss(model, sum_loss_fn, valloader, device, ten_crop)
            training_logger.info(
                f"Initial Validation Loss = {initial_val_loss:.2f}, Initial Validation Acc. = {initial_val_acc:.4f}"
            )

    for t in range(start_epoch, epochs + 1):
        if checkpoint:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{t}.pt")
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
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        train_times.append(train_duration)

        # Eval model
        if report_validation_metrics and valloader is not None:
            val_start = time.time()
            epoch_val_loss, val_acc = compute_validation_loss(model, sum_loss_fn, valloader, device, ten_crop)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(val_acc)
            val_end = time.time()
            val_duration = round((val_end - val_start)/60, 2)
            val_times.append(val_duration)
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
        

        if checkpoint:
            save_checkpoint(
                model,
                optimizer,
                t,
                latest_ckpt,
                scheduler=LR_scheduler,
                metrics={
                    "train_losses": train_losses,
                    "train_accuracies": train_accuracies,
                    "train_times": train_times,
                    "val_losses": val_losses,
                    "val_accuracies": val_accuracies,
                    "val_times": val_times,
                }
            )
