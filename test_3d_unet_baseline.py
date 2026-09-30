import ast
import json
import os
import random
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_3d_unet import train_3d_unet


def load_unet_class():
    with open("notebook33_part2.ipynb") as handle:
        notebook = json.load(handle)
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "Baselines")
    source = "\n".join(part.rstrip("\n") for part in cell["source"])
    start = source.index("class ThreeDUNetBaseline")
    end = source.index("def baseline_cnn_predict")
    class_source = source[start:end]
    ast.parse(class_source)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "MODALITIES": ("t1", "t1ce", "t2", "flair"),
        "NUM_CLASSES": 4,
    }
    exec(class_source, namespace)
    return namespace["ThreeDUNetBaseline"]


class ThreeDUNetBaselineTest(unittest.TestCase):
    def test_forward_backward_and_shapes(self):
        model = load_unet_class()(base_channels=2)
        inputs = torch.randn(1, 4, 15, 17, 19)
        targets = torch.randint(0, 4, (1, 15, 17, 19))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, (1, 4, 15, 17, 19))
        F.cross_entropy(outputs, targets).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_registry_integration_is_present(self):
        with open("notebook33_part2.ipynb") as handle:
            notebook = json.load(handle)
        source = "\n".join(
            part.rstrip("\n")
            for cell in notebook["cells"]
            if cell.get("id") == "Baselines"
            for part in cell["source"]
        )
        self.assertIn("RUN_3D_UNET_BASELINE", source)
        self.assertIn('registry["3D U-Net"] = baseline_metric_summary(unet_rows)', source)
        self.assertIn('registry["model_protocols"]["3D U-Net"] = unet_protocol', source)

    def test_training_helper_saves_checkpoint(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Conv3d(4, 4, 1)

            def forward(self, inputs):
                return self.layer(inputs)

        metadata = {
            "inputs": torch.randn(1, 4, 4, 4, 4),
            "targets": torch.zeros(1, 4, 4, 4, dtype=torch.long),
        }
        checkpoint = "baseline_3d_unet_test_checkpoint.pt"
        try:
            _, protocol = train_3d_unet(
                TinyModel(),
                [(None, "case1"), (None, "case2")],
                [(None, "case")],
                load_meta=lambda _: metadata,
                make_batch=lambda meta: (meta["inputs"], meta["targets"]),
                evaluate=lambda model, meta: {
                    "Dice_WT": 0.1,
                    "Dice_TC": 0.2,
                    "Dice_ET": 0.3,
                },
                loss_fn=lambda outputs, targets: F.cross_entropy(outputs, targets),
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                patience=1,
                checkpoint_path=checkpoint,
                seed=42,
                batch_size=2,
                gradient_accumulation_steps=2,
            )
            self.assertTrue(os.path.exists(checkpoint))
            self.assertEqual(protocol["best_epoch"], 1)
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_device_seed_and_precision_protocol(self):
        model = nn.Conv3d(4, 4, 1)
        checkpoint = "baseline_3d_unet_device_test.pt"
        try:
            expected_seed = 123
            random.seed(expected_seed)
            expected_python = random.random()
            np.random.seed(expected_seed)
            expected_numpy = np.random.random()
            torch.manual_seed(expected_seed)
            expected_torch = torch.rand(1).item()
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            _, protocol = train_3d_unet(
                model,
                [],
                [],
                load_meta=lambda _: {},
                make_batch=lambda meta: (None, None),
                evaluate=lambda current_model, meta: {},
                loss_fn=lambda outputs, targets: outputs.sum(),
                device=torch.device("cpu"),
                epochs=0,
                learning_rate=1e-3,
                weight_decay=0.0,
                patience=1,
                checkpoint_path=checkpoint,
                seed=expected_seed,
                precision="auto",
            )
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertAlmostEqual(random.random(), expected_python)
            self.assertAlmostEqual(np.random.random(), expected_numpy)
            self.assertAlmostEqual(torch.rand(1).item(), expected_torch)
            self.assertFalse(protocol["amp_enabled"])
            self.assertFalse(protocol["reproducibility"]["bitwise_determinism_claimed"])
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_synthetic_smoke(self):
        model = load_unet_class()(base_channels=2)
        volume_shape = (16, 16, 16)
        metadata = {
            "inputs": torch.randn(1, 4, *volume_shape),
            "targets": torch.zeros(1, *volume_shape, dtype=torch.long),
        }
        checkpoint = "baseline_3d_unet_cuda_test.pt"
        try:
            _, protocol = train_3d_unet(
                model,
                [(None, "case")],
                [(None, "case")],
                load_meta=lambda _: metadata,
                make_batch=lambda meta: (meta["inputs"], meta["targets"]),
                evaluate=lambda current_model, meta: {
                    "Dice_WT": 0.1,
                    "Dice_TC": 0.2,
                    "Dice_ET": 0.3,
                },
                loss_fn=lambda outputs, targets: F.cross_entropy(outputs, targets),
                device=torch.device("cuda"),
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                patience=1,
                checkpoint_path=checkpoint,
                seed=42,
                precision="auto",
                channels_last_3d=True,
            )
            print(
                "CUDA smoke:",
                {
                    "device": protocol["device"],
                    "volume_shape": volume_shape,
                    "batch_size": protocol["batch_size"],
                    "precision": protocol["amp_dtype"] or "fp32",
                    "epoch_metrics": protocol["epoch_metrics"],
                },
            )
            self.assertEqual(next(model.parameters()).device.type, "cuda")
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)


if __name__ == "__main__":
    unittest.main()
