import unittest
import torch
from egop_optimizer.models.ImagenetClassifier import (
    ImageNet_model_alexnet,
    ImageNet_model_34_layer_plain,
    ImageNet_model_34_layer_residual,
    ImageNet_model_VGG19,
)


def setUpModule():
    print(f"\nRunning tests in {__file__}")


# Dummy ImageNet-sized input: batch=2, channels=3, height=224, width=224
BATCH_SIZE = 2
NUM_CLASSES = 1000
DUMMY_INPUT = torch.randn(BATCH_SIZE, 3, 224, 224)


class TestAlexNet(unittest.TestCase):
    def test_initialization(self):
        model = ImageNet_model_alexnet()

    def test_forward(self):
        model = ImageNet_model_alexnet()
        model.eval()
        with torch.no_grad():
            output = model(DUMMY_INPUT)
        self.assertEqual(output.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_reinitialize(self):
        model = ImageNet_model_alexnet()
        model.reinitialize()



class TestPlain34(unittest.TestCase):
    def test_initialization(self):
        model = ImageNet_model_34_layer_plain()

    def test_forward(self):
        model = ImageNet_model_34_layer_plain()
        model.eval()
        with torch.no_grad():
            output = model(DUMMY_INPUT)
        self.assertEqual(output.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_reinitialize(self):
        model = ImageNet_model_34_layer_plain()
        model.reinitialize()


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


class TestVGG19(unittest.TestCase):
    def test_initialization(self):
        model = ImageNet_model_VGG19()

    def test_forward(self):
        model = ImageNet_model_VGG19()
        model.eval()
        with torch.no_grad():
            output = model(DUMMY_INPUT)
        self.assertEqual(output.shape, (BATCH_SIZE, NUM_CLASSES))

    def test_reinitialize(self):
        model = ImageNet_model_VGG19()
        model.reinitialize()




if __name__ == "__main__":
    unittest.main()