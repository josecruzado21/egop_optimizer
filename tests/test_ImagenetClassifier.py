import unittest
import torch
from egop_optimizer.models.ImagenetClassifier import (
    ImageNet_model_34_layer_residual,
)


def setUpModule():
    print(f"\nRunning tests in {__file__}")


# Dummy ImageNet-sized input: batch=2, channels=3, height=224, width=224
BATCH_SIZE = 2
NUM_CLASSES = 1000
DUMMY_INPUT = torch.randn(BATCH_SIZE, 3, 224, 224)



class TestResidual34(unittest.TestCase):
    def test_initialization(self):
        model = ImageNet_model_34_layer_residual()

    def test_forward(self):
        model = ImageNet_model_34_layer_residual()
        model.eval()
        with torch.no_grad():
            output = model(DUMMY_INPUT)
        self.assertEqual(output.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_reinitialize(self):
        model = ImageNet_model_34_layer_residual()
        model.reinitialize()




if __name__ == "__main__":
    unittest.main()