import unittest
import torch
import os
import shutil

from egop_optimizer.engine.train import basic_train_loop
from egop_optimizer.models.TinyMNISTClassifier import TinyMNISTClassifier
from egop_optimizer.dataloaders.tinyMNIST_dataloader import tinyMNIST_dataloader

class TestTrainLogging(unittest.TestCase):
    def test_logging_file_creation_and_content(self):
        experiment_name = "unittest_exp"
        log_dir = f"logs/{experiment_name}"
        training_log_path = os.path.join(log_dir, "training.log")
        info_log_path = os.path.join(log_dir, "info.log")

        # Clean up before test
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        model = TinyMNISTClassifier()
        trainloader, valloader, _ = tinyMNIST_dataloader(batch_size=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_method = lambda reduction: torch.nn.CrossEntropyLoss(reduction=reduction)

        basic_train_loop(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_method=loss_method,
            epochs=10,
            LR_scheduler=None,
            valloader=valloader,
            experiment_name=experiment_name,
        )

        # Check log files creation
        self.assertTrue(os.path.exists(training_log_path), f"{training_log_path} was not created.")
        self.assertTrue(os.path.exists(info_log_path), f"{info_log_path} was not created.")

        # Check training log file content
        with open(training_log_path, "r") as f:
            contents = f.read()
            self.assertIn("Epoch", contents, "Training log does not contain expected training logs.")

        # Check info log file content
        with open(info_log_path, "r") as f:
            contents = f.read()
            self.assertIn("Experiment", contents, "Info log does not contain experiment info.")
            self.assertIn("Model architecture", contents, "Info log does not contain model architecture.")

if __name__ == "__main__":
    unittest.main()