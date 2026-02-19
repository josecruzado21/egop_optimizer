import torch

from tqdm.auto import tqdm
import pdb

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


def compute_validation_loss(model, sum_loss_fn, valloader):
    total_val_loss = 0
    for batch_data, batch_labels in valloader:
        output = model(batch_data)
        total_val_loss += sum_loss_fn(output, batch_labels)
    # Average over dataset size
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
):
    """
    loss_method should accept an argument: reduction
    """
    ave_loss_fn = loss_method(reduction="mean")
    sum_loss_fn = loss_method(reduction="sum")

    if LR_scheduler is not None:
        raise Exception("Scheduler not yet supported.")

    for t in range(epochs):
        epoch_loss = 0
        for batch_data, batch_labels in tqdm(trainloader):
            if (
                batch_data.device.type != device.type
                or batch_labels.device.type != device.type
            ):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                    device
                )
            # Training pass
            optimizer.zero_grad()
            output = model(batch_data)
            batch_loss = ave_loss_fn(output, batch_labels)
            # This is where the model learns by backpropagating
            batch_loss.backward()
            # And optimizes its weights here
            optimizer.step()

            epoch_loss += batch_loss
        else:
            model.eval()
            epoch_val_loss = compute_validation_loss(model, sum_loss_fn, valloader)
            print(
                f"Epoch {t}: total loss = {epoch_loss:.2f}, val loss = {epoch_val_loss}"
            )
            model.train()

    return
