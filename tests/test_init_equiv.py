import torch
import unittest

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import (
    TinyMNISTClassifier,
    ReparamTinyMNISTClassifier,
)
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader
from egop_optimizer.utils.EGOP_utils import (
    compute_V_by_layer,
    layerwise_reparam_init_equiv,
)
from egop_optimizer.utils.device_utils import get_available_device

import pdb

DEVICE = get_available_device()


def setUpModule():
    """
    Prints the test file name when the test module is executed.

    Args:
        None

    Returns:
        None: Writes the file path to stdout.
    """
    print(f"\nRunning tests in {__file__}")


def get_V_dict_for_TinyMNIST(OG_model, use_randomized_svd):
    """
    Instantiates dataloaders, criterion, and runs compute_V_by_layer
    """
    trainloader, _, _ = tinyMNIST_dataloader(batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    return compute_V_by_layer(
        model_OG=OG_model,
        k=100,
        data_loader=trainloader,
        criterion=criterion,
        use_randomized_svd=use_randomized_svd,
    )


class TestTinyMNISTReparam(unittest.TestCase):
    def test_execution(self):
        OG_model = TinyMNISTClassifier()
        V_dict = get_V_dict_for_TinyMNIST(OG_model=OG_model, use_randomized_svd=False)
        reparam_model = ReparamTinyMNISTClassifier(V_by_layer_dict=V_dict)

        layerwise_reparam_init_equiv(
            EGOP_model=reparam_model, OG_model=OG_model, seed=42
        )
        return


if __name__ == "__main__":
    unittest.main()
