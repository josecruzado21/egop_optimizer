import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm.auto import tqdm
import random
from ucimlrepo import fetch_ucirepo

import pdb

# Device management
from egop_optimizer.utils.device_utils import get_available_device


DEVICE = get_available_device()


def download_and_save_MNIST(data_dir):
    """
    Downloads the UCI Optical Recognition of Handwritten Digits dataset, recreates the
    original train/test split, and saves the data to disk in the expected raw file format.

    Args:
        data_dir (str): Directory where the processed .tra and .tes files will be saved.

    Returns:
        None: Writes optdigits.tra and optdigits.tes to disk.
    """

    # fetch dataset from UCI repo
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)

    # data (as pandas dataframes)
    X = optical_recognition_of_handwritten_digits.data.features
    y = optical_recognition_of_handwritten_digits.data.targets

    # Hardcode the train/test_and_val split used by UCI
    n_train = 3823
    n_test = 1797

    # Split into train and test_and_val
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test_and_val = X.iloc[-n_test:]
    y_test_and_val = y.iloc[-n_test:]

    # Make labels the last column of the array
    combined_train_data = pd.concat([X_train, y_train], axis=1)
    combined_test_data = pd.concat([X_test_and_val, y_test_and_val], axis=1)

    # Save with expected names
    data_dir = get_default_data_dir()
    test_path = os.path.join(data_dir, "optdigits.tes")
    train_path = os.path.join(data_dir, "optdigits.tra")

    combined_train_data.to_csv(train_path, index=False, header=False)
    combined_test_data.to_csv(test_path, index=False, header=False)

    return


# Checks that the train/test_and_val split which we manually perform on downloaded data is consistent
# the default train test_and_val split for manually downloading separate csvs.
# Requires one to download csv's manually from https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
# and place in the location expected by get_data_dir().
def check_donwload_has_same_train_test_split():
    # fetch dataset from UCI repo
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)

    # data (as pandas dataframes)
    X = optical_recognition_of_handwritten_digits.data.features
    y = optical_recognition_of_handwritten_digits.data.targets

    # Compare with downloaded csvs during development
    data_dir = get_default_data_dir()
    test_path = os.path.join(data_dir, "optdigits.tes")
    train_path = os.path.join(data_dir, "optdigits.tra")
    train_array = pd.read_csv(train_path, header=None).to_numpy()
    test_array = pd.read_csv(test_path, header=None).to_numpy()
    # Labels are in last row of dataframe/array
    trainX = train_array[:, :-1]
    trainY = train_array[:, -1]
    # Take validation as subset of training data
    test_and_val_X = test_array[:, :-1]
    test_and_val_Y = test_array[:, -1]

    n_train = len(trainX)
    n_test = len(test_and_val_X)

    # Split into train and test_and_val
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test_and_val = X.iloc[-n_test:]
    y_test_and_val = y.iloc[-n_test:]

    # See whether data are consistent
    assert np.array_equal(X_train.to_numpy(), trainX)
    assert np.array_equal(y_train.to_numpy().flatten(), trainY)
    assert np.array_equal(X_test_and_val.to_numpy(), test_and_val_X)
    assert np.array_equal(y_test_and_val.to_numpy().flatten(), test_and_val_Y)
    print("All data consistent!")
    return


def get_default_data_dir():
    """
    Returns the filesystem path to the raw optical recognition dataset directory.

    Args:
        None

    Returns:
        Path: Path to raw_data/optical_recognition_of_handwritten_digits relative to the project root.
    """
    # Locate raw_data folder, assumed to be in same folder as egop_optimizer
    base = Path(__file__).resolve()
    data_dir = (
        base.parents[2] / "raw_data" / "optical_recognition_of_handwritten_digits"
    )
    return data_dir


def get_MNIST_train_val_test_data(
    train_val_test_split=[None, 0.33, 0.66],
    device=DEVICE,
    verbose=False,
    data_dir=None,
    seed=342,  # for controling random val/test splits
):
    """
    Loads the Optical Recognition of Handwritten Digits dataset, recreates the fixed UCI
    training split, and divides the remaining data into validation and test sets.

    Args:
        train_val_test_split (list): Fractions for [train, val, test]. Train must be None since
            the training set size is fixed (default: [None, 0.33, 0.66]).
        device (torch.device): Device to place returned tensors on.
        verbose (bool): If True, prints relative split sizes (default: False).
        data_dir (str): Directory containing raw .tra and .tes files. If None, uses default location.
        seed (int): Random seed for validation/test split reproducibility (default: 342).

    Returns:
        tuple: (trainX, trainY, valX, valY, testX, testY) as torch.Tensor objects on the specified device.
    """

    if train_val_test_split[0] is not None:
        raise Exception(
            "tinyMHIST: Train split argument passed, but train size is fixed."
        )

    if data_dir is None:
        data_dir = get_default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    test_path = os.path.join(data_dir, "optdigits.tes")
    train_path = os.path.join(data_dir, "optdigits.tra")

    # Check if data exists at path
    if not os.path.isfile(test_path) or not os.path.isfile(train_path):
        print(
            f"Data not located at path {train_path}. Donwloading from UCI repository."
        )
        download_and_save_MNIST(data_dir)

    # Load directly
    try:
        train_array = pd.read_csv(train_path, header=None).to_numpy()
        test_array = pd.read_csv(test_path, header=None).to_numpy()

    except:
        raise Exception(
            f"Unable to load data. Verify the existence of data in folder: {data_dir}"
        )

    if train_val_test_split[0] is not None:
        raise Exception(
            "tinyMHIST: Train split argument passed, but train size is fixed."
        )

    # Labels are in last column of dataframe/array
    trainX = train_array[:, :-1]
    trainY = train_array[:, -1]
    # Take validation as subset of training data
    test_and_val_X = test_array[:, :-1]
    test_and_val_Y = test_array[:, -1]

    # Split val and test
    valX, testX, valY, testY = sk_train_test_split(
        test_and_val_X,
        test_and_val_Y,
        test_size=train_val_test_split[2]
        / (train_val_test_split[1] + train_val_test_split[2]),
        shuffle=True,
        random_state=seed,
        stratify=test_and_val_Y,
    )

    test_frac = testX.shape[0] / (valX.shape[0] + trainX.shape[0] + testX.shape[0])
    train_frac = trainX.shape[0] / (valX.shape[0] + trainX.shape[0] + testX.shape[0])
    val_frac = valX.shape[0] / (valX.shape[0] + trainX.shape[0] + testX.shape[0])

    if verbose:
        print(
            f"Train, validation, test relative size: {train_frac, val_frac, test_frac}"
        )

    # Format as tensors and move to device
    trainX = torch.from_numpy(trainX).to(torch.float32).to(device)
    trainY = (
        torch.from_numpy(trainY).to(torch.float32).type(torch.LongTensor).to(device)
    )
    valX = torch.from_numpy(valX).to(torch.float32).to(device)
    valY = torch.from_numpy(valY).to(torch.float32).type(torch.LongTensor).to(device)
    testX = torch.from_numpy(testX).to(torch.float32).to(device)
    testY = torch.from_numpy(testY).to(torch.float32).type(torch.LongTensor).to(device)

    return trainX, trainY, valX, valY, testX, testY


def tinyMNIST_dataloader(batch_size, train_val_test_split=[None, 0.33, 0.66]):
    """
    Creates PyTorch DataLoaders for the Optical Recognition digits dataset using the
    predefined training split and configurable validation/test splits.

    Args:
        batch_size (int): Number of samples per batch. If None, uses full-batch training.
        train_val_test_split (list): Fractions for [train, val, test]. Train must be None
            since the training set size is fixed (default: [None, 0.33, 0.66]).

    Returns:
        tuple: (trainloader, valloader, testloader) as torch.utils.data.DataLoader objects.
    """
    trainX, trainY, valX, valY, testX, testY = get_MNIST_train_val_test_data(
        train_val_test_split=train_val_test_split,
    )

    # If no batch size provided, use full batch gradients
    if batch_size is None:
        batch_size = trainX.shape[0]
    trainloader = DataLoader(
        TensorDataset(trainX, trainY), batch_size=batch_size, shuffle=False
    )
    valloader = DataLoader(
        TensorDataset(valX, valY), batch_size=batch_size, shuffle=False
    )
    testloader = DataLoader(
        TensorDataset(testX, testY), batch_size=batch_size, shuffle=False
    )
    return trainloader, valloader, testloader
