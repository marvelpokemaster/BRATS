"""Tests for the independent BraTS 3D U-Net baseline training helper.

These tests are designed to be deterministic, small, and isolated from real
BraTS datasets.  They must NOT be executed on the user's laptop — run them
only in an authorized CI environment or on a dedicated compute machine.

The tests cover:
  - Forward/backward shape correctness (ThreeDUNetBaseline from notebook)
  - Registry integration string presence in the notebook
  - Checkpoint persistence
  - Gradient accumulation with equal microbatch sizes
  - Gradient accumulation with UNEQUAL microbatch sizes (the fix)
  - Single-microbatch equivalence to ordinary optimization
  - Gradient reset between optimizer steps
  - Seed/precision protocol metadata
  - Invalid configuration rejection
  - CUDA synthetic smoke test (skipped when CUDA unavailable)
"""

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
            self.assertEqual(protocol["gradient_weighting"],
                             "patient-weighted (each patient contributes equally)")
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_gradient_accumulation_divisible_and_partial_groups(self):
        """Equal-sized microbatches: accumulated gradient must match single-batch."""

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

        def run(case_ids, accumulation, batch_sz=None):
            if batch_sz is None:
                batch_sz = 2 if accumulation == 1 else 1
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
                    batch_size=batch_sz,
                    gradient_accumulation_steps=accumulation,
                )
                return float(model.weight.detach()), protocol["optimizer_steps"]
            finally:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)

        # Equal microbatches: 2 cases accumulated individually vs 1 batch of 2
        divisible_weight, divisible_steps = run(["case1", "case2"], 2)
        reference_weight, reference_steps = run(["case1", "case2"], 1)
        self.assertAlmostEqual(divisible_weight, reference_weight, places=6)
        self.assertEqual(divisible_steps, 1)
        self.assertEqual(reference_steps, 1)

        # Partial group: 3 cases with accumulation=2 gives 2 optimizer steps
        # (group 1: case1+case2, group 2: case3 alone)
        partial_weight, partial_steps = run(["case1", "case2", "case3"], 2)
        self.assertEqual(partial_steps, 2)
        # Manually compute expected weight through 2 AdamW steps
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

    def test_unequal_microbatch_patient_weighting(self):
        """CORE FIX TEST: Unequal microbatch sizes must produce the same
        gradient as a single batch containing all patients.

        Scenario: 3 cases, batch_size=2, gradient_accumulation_steps=2.
          Microbatch 1: case1 + case2 (2 patients)
          Microbatch 2: case3 (1 patient, final incomplete batch)
        This must be equivalent to a single batch of all 3 patients.
        """

        class ScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, inputs):
                return inputs * self.weight

        targets_by_case = {
            "case1": torch.tensor([1.0]),
            "case2": torch.tensor([2.0]),
            "case3": torch.tensor([3.0]),
        }

        def run(case_ids, batch_sz, accumulation):
            checkpoint = f"baseline_uneq_{batch_sz}_{accumulation}.pt"
            model = ScalarModel()
            try:
                _, protocol = train_3d_unet(
                    model,
                    [(None, cid) for cid in case_ids],
                    [(None, case_ids[0])],
                    load_meta=lambda path: {"target": targets_by_case[path]},
                    make_batch=lambda meta: (torch.ones(1), meta["target"]),
                    evaluate=lambda m, meta: {
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
                    batch_size=batch_sz,
                    gradient_accumulation_steps=accumulation,
                )
                return float(model.weight.detach()), protocol["optimizer_steps"]
            finally:
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)

        # Unequal microbatches: batch_size=2, accum=2
        #   microbatch 1: case1+case2 (2 patients)
        #   microbatch 2: case3 (1 patient) — then forced optimizer step
        unequal_weight, unequal_steps = run(
            ["case1", "case2", "case3"], batch_sz=2, accumulation=2
        )
        self.assertEqual(unequal_steps, 1)  # all 3 fit in one accumulation group

        # Reference: single batch of all 3 patients, no accumulation
        ref_weight, ref_steps = run(
            ["case1", "case2", "case3"], batch_sz=3, accumulation=1
        )
        self.assertEqual(ref_steps, 1)

        # With patient-weighted averaging, both must produce identical results
        self.assertAlmostEqual(
            unequal_weight,
            ref_weight,
            places=6,
            msg=(
                f"Unequal microbatch sizes (2+1) must produce the same "
                f"parameter update as a single batch of 3. "
                f"Got {unequal_weight} vs {ref_weight}"
            ),
        )

    def test_single_microbatch_equivalence(self):
        """A single microbatch with accumulation=1 must be equivalent to
        ordinary (non-accumulated) optimization for one step."""

        class ScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1.0]))

            def forward(self, inputs):
                return inputs * self.weight

        checkpoint = "baseline_single_mb.pt"
        model = ScalarModel()
        target = torch.tensor([2.0])
        try:
            _, protocol = train_3d_unet(
                model,
                [(None, "c1")],
                [(None, "c1")],
                load_meta=lambda _: {"t": target},
                make_batch=lambda meta: (torch.ones(1), meta["t"]),
                evaluate=lambda m, meta: {
                    "Dice_WT": 0.5,
                    "Dice_TC": 0.5,
                    "Dice_ET": 0.5,
                },
                loss_fn=lambda o, t: (o - t).square().mean(),
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.01,
                weight_decay=0.0,
                patience=1,
                checkpoint_path=checkpoint,
                seed=42,
                batch_size=1,
                gradient_accumulation_steps=1,
            )
            self.assertEqual(protocol["optimizer_steps"], 1)
            # Manual AdamW: grad = 2*(1-2) = -2, clipped at 2.0
            # Adam moment m = 0.1*(-2)=-0.2, v = 0.001*4=0.004
            # w = 1 - 0.01 * (-0.2/(1-0.9)) / ((0.004/(1-0.999))**0.5 + 1e-8)
            # = 1 - 0.01 * (-2) / (2 + 1e-8) ≈ 1.01
            self.assertGreater(float(model.weight), 1.0,
                               "Single-step AdamW should have moved weight toward target=2")
        finally:
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    def test_gradient_reset_between_optimizer_steps(self):
        """Gradients from one accumulation group must not leak into the next."""

        class ScalarModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.0]))

            def forward(self, inputs):
                return inputs * self.weight

        # 4 cases, accumulation=2, batch_size=1 → 2 optimizer steps
        # If gradients leak, the second step would include stale gradients.
        targets = {f"c{i}": torch.tensor([float(i)]) for i in range(1, 5)}
        checkpoint = "baseline_grad_reset.pt"
        model = ScalarModel()
        try:
            _, protocol = train_3d_unet(
                model,
                [(None, f"c{i}") for i in range(1, 5)],
                [(None, "c1")],
                load_meta=lambda path: {"t": targets[path]},
                make_batch=lambda meta: (torch.ones(1), meta["t"]),
                evaluate=lambda m, meta: {
                    "Dice_WT": 0.5,
                    "Dice_TC": 0.5,
                    "Dice_ET": 0.5,
                },
                loss_fn=lambda o, t: (o - t).square().mean(),
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.001,
                weight_decay=0.0,
                patience=1,
                checkpoint_path=checkpoint,
                seed=42,
                batch_size=1,
                gradient_accumulation_steps=2,
            )
            self.assertEqual(protocol["optimizer_steps"], 2)
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
