"""Training helper for the independent BraTS 3D U-Net baseline.

Gradient accumulation uses **patient-weighted** averaging: each patient
contributes equally to the effective batch gradient regardless of how many
patients happen to fall into each microbatch.  The loss function is assumed
to use mean reduction across the batch dimension (the default for
``F.cross_entropy`` and for the project's ``voxel_loss``).

Implementation:
  For each microbatch with ``n_i`` patients and mean-reduced loss ``L_i``,
  the backward pass receives ``n_i * L_i`` so that the accumulated gradient
  is ``sum_i  n_i * grad(L_i)``.  At the optimizer step the gradient is
  divided by ``N_total = sum_i n_i`` (the total number of patients in the
  accumulation group), producing ``(1/N) * sum_i n_i * grad(L_i)`` which
  is mathematically equivalent to the gradient of a single mean-reduced
  loss computed over all ``N_total`` patients simultaneously.

When all microbatches have the same size this is identical to dividing by
the number of microbatches; the correction matters only for incomplete
final microbatches.
"""

import os
import random
import time

import numpy as np
import torch


def train_3d_unet(
    model,
    train_set,
    val_set,
    *,
    load_meta,
    make_batch,
    evaluate,
    loss_fn,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    checkpoint_path,
    seed,
    precision="auto",
    channels_last_3d=False,
    batch_size=1,
    gradient_accumulation_steps=1,
    loss_name=None,
):
    """Train a supplied 3D U-Net and persist its best validation checkpoint."""
    train_set = list(train_set)
    val_set = list(val_set)
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")
    if not train_set:
        raise ValueError("train_set must not be empty")
    if not val_set:
        raise ValueError("val_set must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seed_applied = device.type == "cuda"
    if cuda_seed_applied:
        torch.cuda.manual_seed_all(seed)

    use_amp = False
    amp_dtype = None
    use_grad_scaler = False
    precision_fallback_reason = None
    if device.type == "cuda" and precision != "off":
        if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
        elif precision == "bf16":
            raise RuntimeError("BF16 precision requested but unsupported by CUDA device")
        elif precision == "fp16":
            use_amp = True
            amp_dtype = torch.float16
            use_grad_scaler = True
        elif precision == "auto":
            precision_fallback_reason = "BF16 unsupported; using FP32"
        elif precision not in ("auto", "bf16", "fp16"):
            raise ValueError(f"unsupported precision: {precision}")
    elif device.type != "cuda" and precision not in ("auto", "off"):
        raise ValueError(f"{precision} precision requires a CUDA device")
    elif precision not in ("auto", "off", "bf16", "fp16"):
        raise ValueError(f"unsupported precision: {precision}")

    model = model.to(device)
    if channels_last_3d:
        model = model.to(memory_format=torch.channels_last_3d)
    model.channels_last_3d = bool(channels_last_3d)
    model.amp_enabled = bool(use_amp)
    model.amp_dtype = amp_dtype
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    best_state = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    epoch_metrics = []
    optimizer_step_count = 0

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_inputs = []
        pending_targets = []
        pending_count = 0
        accumulated_microbatches = 0
        # Patient-weighted gradient accumulation: track total patients, not
        # just microbatch count, so that each patient contributes equally.
        accumulated_patients = 0

        def optimizer_step():
            nonlocal accumulated_microbatches, accumulated_patients, optimizer_step_count
            if accumulated_microbatches == 0:
                return
            if use_grad_scaler:
                scaler.unscale_(optimizer)
            # Normalize by total number of patients across all microbatches
            # in this accumulation group.  Each microbatch's backward pass
            # scaled its loss by its own patient count, so accumulated
            # gradients equal  sum_i  n_i * grad(L_i).  Dividing by N_total
            # yields the patient-weighted mean gradient.
            gradient_scale = 1.0 / accumulated_patients
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_microbatches = 0
            accumulated_patients = 0
            optimizer_step_count += 1

        def train_pending():
            nonlocal pending_inputs, pending_targets, pending_count
            if not pending_inputs:
                return
            n_patients = pending_count  # actual patient count in this microbatch
            inputs = torch.cat(pending_inputs, dim=0)
            targets = torch.cat(pending_targets, dim=0)
            if channels_last_3d:
                inputs = inputs.contiguous(memory_format=torch.channels_last_3d)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(inputs)
                loss = loss_fn(logits, targets)
            # Scale by the number of patients in this microbatch so that
            # each patient's contribution to the accumulated gradient is
            # independent of how many patients happen to share a microbatch.
            # The optimizer_step will normalize by accumulated_patients.
            scaled_loss = loss * n_patients
            if use_grad_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_inputs = []
            pending_targets = []
            pending_count = 0
            nonlocal accumulated_microbatches, accumulated_patients
            accumulated_microbatches += 1
            accumulated_patients += n_patients
            if accumulated_microbatches >= gradient_accumulation_steps:
                optimizer_step()

        for _, meta_path in train_set:
            meta = load_meta(meta_path)
            inputs, targets = make_batch(meta)
            if inputs.shape[0] != targets.shape[0]:
                raise ValueError("input and target batch dimensions must match")
            if pending_inputs and (
                pending_count + inputs.shape[0] > batch_size
                or inputs.shape[2:] != pending_inputs[0].shape[2:]
            ):
                train_pending()
            pending_inputs.append(inputs)
            pending_targets.append(targets)
            pending_count += inputs.shape[0]
            if pending_count >= batch_size:
                train_pending()

        if pending_inputs:
            train_pending()
        optimizer_step()

        validation_rows = []
        for _, meta_path in val_set:
            meta = load_meta(meta_path)
            validation_rows.append(evaluate(model, meta))
        required_metrics = ("Dice_WT", "Dice_TC", "Dice_ET")
        if not validation_rows:
            raise ValueError("validation produced no metric rows")
        for row in validation_rows:
            if any(metric not in row for metric in required_metrics):
                raise ValueError("validation metrics must include WT, TC, and ET Dice")
            if not all(np.isfinite(float(row[metric])) for metric in required_metrics):
                raise ValueError("validation Dice metrics must be finite")
        score = float(
            np.mean(
                [
                    np.mean([float(row[metric]) for metric in required_metrics])
                    for row in validation_rows
                ]
            )
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        epoch_metrics.append(
            {
                "epoch": int(epoch),
                "elapsed_seconds": float(time.perf_counter() - epoch_started),
                "peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    protocol = {
        "architecture": "Independent 3D U-Net (2-level encoder-decoder)",
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "epoch_cap": int(epochs),
        "early_stopping_patience": int(patience),
        "best_epoch": int(best_epoch),
        "loss": loss_name or getattr(loss_fn, "__name__", "supplied_loss_fn"),
        "seed": int(seed),
        "checkpoint": os.path.abspath(checkpoint_path),
        "device": str(device),
        "precision_requested": precision,
        "amp_enabled": use_amp,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "precision_fallback_reason": precision_fallback_reason,
        "channels_last_3d": bool(channels_last_3d),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_weighting": "patient-weighted (each patient contributes equally)",
        "optimizer_steps": int(optimizer_step_count),
        "epoch_metrics": epoch_metrics,
        "reproducibility": {
            "seed_applied": True,
            "python_random": True,
            "numpy_random": True,
            "torch_random": True,
            "cuda_random": cuda_seed_applied,
            "deterministic_algorithms": False,
            "bitwise_determinism_claimed": False,
        },
    }
    torch.save(
        {"state_dict": model.state_dict(), "protocol": protocol},
        checkpoint_path,
    )
    return model, protocol
