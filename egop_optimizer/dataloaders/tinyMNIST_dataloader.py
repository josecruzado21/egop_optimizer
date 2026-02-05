import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import os
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
from tqdm.auto import tqdm
import random

import pdb

# Device management
# from egop_optimizer.utils.device_utils import get_available_device
from utils.device_utils import get_available_device

DEVICE = get_available_device()


# TODO: modify to automatically download from UCI site
def get_MNIST_train_val_test_data(
    data_dir="../raw_data/optical_recognition_of_handwritten_digits/",
    train_val_test_split=[None, 0.33, 0.66],
    device=DEVICE,
    verbose=False,
):
    rand_seed = 342

    if train_val_test_split[0] is not None:
        raise Exception(
            "tinyMHIST: Train split argument passed, but train size is fixed."
        )

    test_path = os.path.join(data_dir, "optdigits.tes")
    train_path = os.path.join(data_dir, "optdigits.tra")
    # Load directly
    train_array = pd.read_csv(train_path, header=None).to_numpy()
    test_array = pd.read_csv(test_path, header=None).to_numpy()

    # Labels are in last row of dataframe/array
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
        random_state=rand_seed,
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
