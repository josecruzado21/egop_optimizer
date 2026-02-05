import torch

"""
Script set up to allow eventual scaling of code with use of DataParallel or DistributedDataParallel.
"""


def get_available_device() -> torch.device:
    """
    Returns the appropriate device for training.
    Uses 'cuda:0' if available, 'mps' if available on macOS, otherwise falls back to 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
