import torch

"""
Script set up to allow eventual scaling of code with use of DataParallel or DistributedDataParallel.
"""


def get_available_device() -> torch.device:
    """
    Returns the appropriate device for training, preferring GPU acceleration when available.

    Args:
        None

    Returns:
        torch.device: 'cuda:0' if CUDA is available, 'mps' if Apple Silicon MPS is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
