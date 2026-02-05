import torch
import numpy as np

# Device management
from egop_optimizer.utils.device_utils import get_available_device


DEVICE = get_available_device()


# Build a random ill-conditioned matrix A of shape nxd
def build_A(n, d, alpha):
    U, _ = np.linalg.qr(np.random.normal(size=(n, d)))
    Vh, _ = np.linalg.qr(np.random.normal(size=(d, d)))

    # Old code for shelf-structure singular values:
    # sing_vals = jnp.array([jnp.power(float(ii),-alpha) for ii in range(int(d/2))]+[jnp.power(float(ii),-3*alpha) for ii in range(int(d/2),d)])
    rank = min(d, n)
    sing_vals = np.power(np.arange(1.0, rank + 1.0), -alpha)

    A = U[:, :rank] @ np.diag(sing_vals) @ Vh[:rank, :]
    return A


def linear_networks_generate_dataloader(
    batch_size,
    input_size: int = 10,
    output_size: int = 10,
    alpha: float = 2.0,
    sample_list: list = [10000, 2000, 2000],
    noise_scale=None,
    device=DEVICE,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """
    Generates X a measurement matrix of size output_size x input_size with spectral decay governed by alpha.
    Then generates Gaussian i.i.d. data M_star and labels Y=X@M_star.
    Sample_list has int entries [num_train_samples, num_val_samples, num_test_samples].

    Loads data to-device upon creation. This might be incompatible with the way fashionMNIST's dataloaders are
    set up, which I think assumes data gets moved to device during training.
    """

    num_train_samples = sample_list[0]
    num_val_samples = sample_list[1]
    num_test_samples = sample_list[2]

    # Generate data
    # Ground-truth M_star Gaussian i.i.d.
    M_star = torch.normal(mean=0.0, std=1.0, size=(input_size, output_size)).to(device)
    # Measurement matrices with spectral decay
    trainX = (
        torch.from_numpy(build_A(n=num_train_samples, d=input_size, alpha=alpha))
        .float()
        .to(device)
    )
    valX = (
        torch.from_numpy(build_A(n=num_val_samples, d=input_size, alpha=alpha))
        .float()
        .to(device)
    )
    testX = (
        torch.from_numpy(build_A(n=num_test_samples, d=input_size, alpha=alpha))
        .float()
        .to(device)
    )
    # Labels
    trainY = trainX @ M_star
    valY = valX @ M_star
    testY = testX @ M_star

    # Adding noise
    # 2*std < (1/2)*scale of signal -> std = 0.25*smallest order magnitude abs val trainY  coordinate in R^d
    if noise_scale is None:
        pass
    elif noise_scale == "auto":
        Y_magnitude = torch.min(
            torch.floor(torch.log10(torch.mean(abs(trainY), axis=0)))
        ).to(int)
        noise_scale = 0.25 * torch.pow(10.0, Y_magnitude)
        train_noise = torch.normal(mean=0.0, std=noise_scale, size=trainY.shape).to(
            device
        )
        val_noise = torch.normal(mean=0.0, std=noise_scale, size=valY.shape).to(device)

        trainY = trainY + train_noise
        valY = valY + val_noise
    else:
        train_noise = torch.normal(mean=0.0, std=noise_scale, size=trainY.shape).to(
            device
        )
        val_noise = torch.normal(mean=0.0, std=noise_scale, size=valY.shape).to(device)

        trainY = trainY + train_noise
        valY = valY + val_noise

    ## Create dataloders from the subsets
    ## No need to pin memory b.c. tensors moved to device ahead of time.
    def make_loader(Xdata, Ydata):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xdata, Ydata),
            batch_size=batch_size,
            shuffle=True,
            # pin_memory=True,
            num_workers=0,
            # prefetch_factor=prefetch_factor,
            # persistent_workers=persistent_workers,
        )

    # Define pytorch dataloaders
    trainloader = make_loader(trainX, trainY)
    valloader = make_loader(valX, valY)
    testloader = make_loader(testX, testY)

    ## No need to pin memory b.c. tensors moved to device ahead of time.
    # trainloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(trainX, trainY),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    # valloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(valX, valY), batch_size=batch_size, shuffle=True
    # )
    # testloader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(testX, testY),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    return trainloader, valloader, testloader
