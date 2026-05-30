import csv
import os
import torch
import logging
from datetime import datetime
import time
from tqdm.auto import tqdm
import pdb

from egop_optimizer.utils.device_utils import get_available_device
from egop_optimizer.utils.EGOP_utils import compute_V_by_layer

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

    if hasattr(model, "V_by_layer_dict"):
        checkpoint["V_by_layer_dict"] = model.V_by_layer_dict

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
    V_by_layer_dict = checkpoint.get("V_by_layer_dict", None)

    return start_epoch, train_losses, train_accuracies, train_times, val_losses, val_accuracies, val_times, V_by_layer_dict

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
    testloader=None,
    device=DEVICE,
    experiment_name="default",
    ten_crop = False,
    report_validation_metrics = True,
    checkpoint = True,
    initial_metrics = True,
    V_recalc_freq=None,
    model_OG=None,
    k=1000,
    reparam_linear_layers=False,
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
    # suffix = 0
    # while os.path.exists(log_dir):
    #     suffix += 1
    #     log_dir = f"logs/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
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

    start_epoch = 1
    train_losses = []
    train_accuracies = []
    train_times = []
    val_losses = []
    val_accuracies = []
    val_times = []

    if checkpoint and os.path.exists(latest_ckpt):
        start_epoch, train_losses, train_accuracies, train_times, val_losses, val_accuracies, val_times, ckpt_V_dict = load_checkpoint(
            model,
            optimizer,
            latest_ckpt,
            scheduler=LR_scheduler,
            device=device
        )
        if ckpt_V_dict is not None and hasattr(model, "restore_V_from_dict"):
            model.restore_V_from_dict(ckpt_V_dict)
        training_logger.info(f"Resuming training from epoch {start_epoch}")

    if device is not None:
        model = model.to(device)
    ave_loss_fn = loss_method(reduction="mean")
    sum_loss_fn = loss_method(reduction="sum")
    if LR_scheduler is not None:
        training_logger.error("Scheduler not yet supported.")
        raise Exception("Scheduler not yet supported.")
    # --- Initial metrics reporting ---
    initial_train_loss = initial_train_acc = initial_val_loss = initial_val_acc = None
    if initial_metrics:
        # The reason why we have model.train() for the initial training metrics is to ensure that any 
        # layers that behave differently during training (like dropout or batch normalization) are in 
        # the correct mode when we compute the initial training loss and accuracy. 
        # This way, we get a more accurate representation of the model's performance on the training 
        # data before any updates are made.
        if start_epoch == 1:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            total_train_loss = 0.0
            total_train, correct_train = 0, 0
            for batch_data, batch_labels in tqdm(trainloader, leave=False):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                output = model(batch_data)
                total_train_loss += sum_loss_fn(output, batch_labels).item()
                _, predicted = torch.max(output, 1)
                correct_train += (predicted == batch_labels).sum().item()
                total_train += batch_labels.size(0)
        initial_train_loss = total_train_loss / len(trainloader.dataset)
        initial_train_acc = correct_train / total_train if total_train > 0 else 0

        training_logger.info(
            f"\nInitial Training Loss = {initial_train_loss:.2f}, Initial Training Acc. = {initial_train_acc:.4f}"
        )

        # Compute initial validation metrics if valloader is provided
        if valloader is not None:
            initial_val_loss, initial_val_acc = compute_validation_loss(model, sum_loss_fn, valloader, device, ten_crop)
            training_logger.info(
                f"Initial Validation Loss = {initial_val_loss:.2f}, Initial Validation Acc. = {initial_val_acc:.4f}\n"
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
                f"Epoch {t}: Training Loss = {epoch_loss:.10f}, Validation Loss = {epoch_val_loss:.10f}, "
                f"Training Acc. = {train_acc:.10f}, Validation Acc. = {val_acc:.10f}, "
                f"Training Time = {train_duration:.2f}m, Validation Time = {val_duration:.2f}m"
            )
        else:
            training_logger.info(
                f"Epoch {t}: Training Loss = {epoch_loss:.10f}, Training Acc. = {train_acc:.10f}, "
                f"Training Time = {train_duration:.2f}m"
            )
        

        if (
            V_recalc_freq is not None
            and model_OG is not None
            and hasattr(model, "update_reparam_weights")
            and t % V_recalc_freq == 0
        ):
            training_logger.info(f"Epoch {t}: recomputing EGOP basis (V_recalc_freq={V_recalc_freq})...")
            reparam_start = time.time()
            V_prev_dict = {name: module.V.clone() for name, module in model.named_modules() if hasattr(module, "V")}
            new_V_dict = compute_V_by_layer(
                model_OG=model_OG,
                k=k,
                data_loader=trainloader,
                criterion=loss_method(reduction="mean"),
                device=device,
                reparam_linear_layers=reparam_linear_layers,
                recalculate_V=True,
                current_model=model,
            )
            new_V_dict = {name: V.to(device) for name, V in new_V_dict.items()}
            model.update_reparam_weights(V_prev_dict, new_V_dict)
            model.zero_grad()
            reparam_dur = round((time.time() - reparam_start) / 60, 2)
            training_logger.info(f"Epoch {t}: basis updated in {reparam_dur:.2f}m")

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

    csv_path = os.path.join(log_dir, "metrics.csv")
    has_val = len(val_losses) > 0
    fieldnames = ["experiment_id", "epoch", "train_loss", "train_accuracy", "train_time_min"]
    if has_val:
        fieldnames += ["val_loss", "val_accuracy", "val_time_min"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        if start_epoch == 1 and initial_train_loss is not None:
            row0 = {"experiment_id": experiment_name, "epoch": 0, "train_loss": initial_train_loss, "train_accuracy": initial_train_acc, "train_time_min": ""}
            if has_val and initial_val_loss is not None:
                row0["val_loss"] = initial_val_loss
                row0["val_accuracy"] = initial_val_acc
                row0["val_time_min"] = ""
            writer.writerow(row0)
        for i in range(len(train_losses)):
            row = {
                "experiment_id": experiment_name,
                "epoch": i + 1,
                "train_loss": train_losses[i],
                "train_accuracy": train_accuracies[i],
                "train_time_min": train_times[i],
            }
            if has_val:
                row["val_loss"] = val_losses[i]
                row["val_accuracy"] = val_accuracies[i]
                row["val_time_min"] = val_times[i]
            writer.writerow(row)
    training_logger.info(f"Metrics saved to {csv_path}")
