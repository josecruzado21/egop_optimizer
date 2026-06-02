import torch
from pathlib import Path

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.ImagenetClassifier import ImageNet_model_34_layer_residual
from egop_optimizer.dataloaders.ImageNet_dataloader import ImageNet_dataloader
from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()

LR_VALUES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
WD_VALUES = [0.01]
TOTAL_EPOCHS = 10


def fmt(value):
    """Format a float as clean scientific notation, e.g. 0.001 -> '1e-3'."""
    s = f"{value:.0e}"          # "1e-03"
    mantissa, exp = s.split("e")
    return f"{int(float(mantissa))}e{int(exp)}"


def is_done(experiment_name):
    ckpt = Path(f"logs/{experiment_name}/checkpoints/latest.pt")
    if not ckpt.exists():
        return False
    try:
        cp = torch.load(ckpt, map_location="cpu")
        return cp.get("epoch", 0) >= TOTAL_EPOCHS
    except Exception:
        return False


if __name__ == "__main__":
    ten_crop = True
    data_dir = Path("/share/data/vdata/imagenet1k")
    trainloader, valloader = ImageNet_dataloader(root=data_dir, batch_size=256, ten_crop=ten_crop)
    loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

    for lr in LR_VALUES:
        for wd in WD_VALUES:
            exp_name = f"ImageNet_resnet34_OG_lr{fmt(lr)}_wd{fmt(wd)}"

            if is_done(exp_name):
                print(f"[sweep] SKIP {exp_name} (already completed {TOTAL_EPOCHS} epochs)")
                continue

            print(f"[sweep] START {exp_name}")

            model = ImageNet_model_34_layer_residual()
            model.reinitialize_seeded(seed=0)
            model = model.to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            basic_train_loop(
                model=model,
                trainloader=trainloader,
                optimizer=optimizer,
                loss_method=loss_method,
                epochs=TOTAL_EPOCHS,
                LR_scheduler=None,
                valloader=valloader,
                experiment_name=exp_name,
                ten_crop=ten_crop,
                report_validation_metrics=True,
            )

            print(f"[sweep] DONE {exp_name}")
