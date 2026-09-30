"""Training helper for the independent BraTS 3D U-Net baseline."""

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
):
    """Train a supplied 3D U-Net and persist its best validation checkpoint."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    device = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_amp = False
    amp_dtype = None
    use_grad_scaler = False
    if device.type == "cuda" and precision != "off":
        if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
        elif precision in ("auto", "fp16"):
            use_amp = True
            amp_dtype = torch.float16
            use_grad_scaler = True
        elif precision not in ("auto", "bf16", "fp16"):
            raise ValueError(f"unsupported precision: {precision}")
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

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_inputs = []
        pending_targets = []
        pending_count = 0
        optimizer_steps = 0

        def train_pending():
            nonlocal pending_inputs, pending_targets, pending_count, optimizer_steps
            if not pending_inputs:
                return
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
                loss = loss_fn(logits, targets) / gradient_accumulation_steps
            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            pending_inputs = []
            pending_targets = []
            pending_count = 0
            optimizer_steps += 1
            if optimizer_steps % gradient_accumulation_steps == 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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
        if optimizer_steps % gradient_accumulation_steps:
            if use_grad_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        validation_rows = []
        for _, meta_path in val_set:
            meta = load_meta(meta_path)
            validation_rows.append(evaluate(model, meta))
        score = (
            float(
                np.mean(
                    [
                        (row["Dice_WT"] + row["Dice_TC"] + row["Dice_ET"]) / 3
                        for row in validation_rows
                    ]
                )
            )
            if validation_rows
            else 0.0
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
        "loss": "Dice + CE",
        "seed": int(seed),
        "checkpoint": os.path.abspath(checkpoint_path),
        "device": str(device),
        "precision_requested": precision,
        "amp_enabled": use_amp,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "channels_last_3d": bool(channels_last_3d),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "epoch_metrics": epoch_metrics,
        "reproducibility": {
            "seed_applied": True,
            "python_random": True,
            "numpy_random": True,
            "torch_random": True,
            "cuda_random": bool(torch.cuda.is_available()),
            "deterministic_algorithms": False,
            "bitwise_determinism_claimed": False,
        },
    }
    torch.save(
        {"state_dict": model.state_dict(), "protocol": protocol},
        checkpoint_path,
    )
    return model, protocol
