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
            self.assertEqual(protocol["optimizer_steps"], 1)
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_gradient_accumulation_divisible_and_partial_groups(self):
        class ScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.1]))

            def forward(self, inputs):
                return inputs * self.weight

        targets_by_case = {
            "case1": torch.tensor([0.2]),
            "case2": torch.tensor([0.3]),
            "case3": torch.tensor([0.4]),
        }
        inputs = torch.ones(1)

        def run(case_ids, accumulation):
            checkpoint = f"baseline_3d_unet_accum_{len(case_ids)}_{accumulation}.pt"
            model = ScalarModel()
            try:
                _, protocol = train_3d_unet(
                    model,
                    [(None, case_id) for case_id in case_ids],
                    [(None, case_ids[0])],
                    load_meta=lambda path: {"target": targets_by_case[path]},
                    make_batch=lambda meta: (inputs, meta["target"]),
                    evaluate=lambda current_model, meta: {
                        "Dice_WT": 0.1,
                        "Dice_TC": 0.2,
                        "Dice_ET": 0.3,
                    },
                    loss_fn=lambda outputs, targets: (outputs - targets).square().mean(),
                    device=torch.device("cpu"),
                    epochs=1,
                    learning_rate=0.01,
                    weight_decay=0.0,
                    patience=1,
                    checkpoint_path=checkpoint,
                    seed=42,
                    batch_size=2 if accumulation == 1 else 1,
                    gradient_accumulation_steps=accumulation,
                )
                return float(model.weight.detach()), protocol["optimizer_steps"]
            finally:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)

        divisible_weight, divisible_steps = run(["case1", "case2"], 2)
        reference_weight, reference_steps = run(["case1", "case2"], 1)
        self.assertAlmostEqual(divisible_weight, reference_weight, places=6)
        self.assertEqual(divisible_steps, 1)
        self.assertEqual(reference_steps, 1)

        partial_weight, partial_steps = run(["case1", "case2", "case3"], 2)
        self.assertEqual(partial_steps, 2)
        expected_weight = 0.1
        first_gradient = sum(
            2.0 * (expected_weight - float(targets_by_case[case_id]))
            for case_id in ("case1", "case2")
        ) / 2.0
        first_moment = 0.1 * first_gradient
        first_variance = 0.001 * first_gradient**2
        expected_weight -= 0.01 * (
            first_moment / (1 - 0.9)
        ) / ((first_variance / (1 - 0.999)) ** 0.5 + 1e-8)
        second_gradient = 2.0 * (expected_weight - float(targets_by_case["case3"]))
        second_moment = 0.9 * first_moment + 0.1 * second_gradient
        second_variance = 0.999 * first_variance + 0.001 * second_gradient**2
        expected_weight -= 0.01 * (
            second_moment / (1 - 0.9**2)
        ) / ((second_variance / (1 - 0.999**2)) ** 0.5 + 1e-8)
        self.assertAlmostEqual(partial_weight, expected_weight, places=6)

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
                [(None, "case")],
                [(None, "case")],
                load_meta=lambda _: {
                    "inputs": torch.zeros(1, 4, 4, 4, 4),
                    "targets": torch.zeros(1, 4, 4, 4, dtype=torch.long),
                },
                make_batch=lambda meta: (meta["inputs"], meta["targets"]),
                evaluate=lambda current_model, meta: {
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
                seed=expected_seed,
                precision="auto",
            )
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertAlmostEqual(random.random(), expected_python)
            self.assertAlmostEqual(np.random.random(), expected_numpy)
            self.assertAlmostEqual(torch.rand(1).item(), expected_torch)
            self.assertFalse(protocol["amp_enabled"])
            self.assertFalse(protocol["reproducibility"]["cuda_random"])
            self.assertFalse(protocol["reproducibility"]["bitwise_determinism_claimed"])
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_invalid_training_configuration_is_rejected(self):
        model = nn.Conv3d(1, 1, 1)
        common = {
            "load_meta": lambda _: {
                "inputs": torch.zeros(1, 1, 2, 2, 2),
                "targets": torch.zeros(1, 2, 2, 2, dtype=torch.long),
            },
            "make_batch": lambda meta: (meta["inputs"], meta["targets"]),
            "evaluate": lambda current_model, meta: {
                "Dice_WT": 0.1,
                "Dice_TC": 0.2,
                "Dice_ET": 0.3,
            },
            "loss_fn": lambda outputs, targets: outputs.square().mean(),
            "device": torch.device("cpu"),
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "patience": 1,
            "checkpoint_path": "baseline_3d_unet_invalid.pt",
            "seed": 1,
        }
        for overrides in (
            {"epochs": 0, "train_set": [(None, "case")], "val_set": [(None, "case")]},
            {"epochs": 1, "train_set": [], "val_set": [(None, "case")]},
            {"epochs": 1, "train_set": [(None, "case")], "val_set": []},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    train_3d_unet(model, **common, **overrides)

        with self.assertRaises(ValueError):
            train_3d_unet(
                model,
                [(None, "case")],
                [(None, "case")],
                **common,
                epochs=1,
                precision="bf16",
            )

        for invalid_metrics in (
            {"Dice_WT": 0.1, "Dice_TC": 0.2},
            {"Dice_WT": float("nan"), "Dice_TC": 0.2, "Dice_ET": 0.3},
        ):
            with self.subTest(invalid_metrics=invalid_metrics):
                with self.assertRaises(ValueError):
                    train_3d_unet(
                        model,
                        [(None, "case")],
                        [(None, "case")],
                        **{
                            key: value
                            for key, value in common.items()
                            if key != "evaluate"
                        },
                        epochs=1,
                        evaluate=lambda current_model, meta: invalid_metrics,
                    )

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
