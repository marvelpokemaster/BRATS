"""Training helper for the independent BraTS 3D U-Net baseline."""

import os

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
):
    """Train a supplied 3D U-Net and persist its best validation checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for _, meta_path in train_set:
            meta = load_meta(meta_path)
            inputs, targets = make_batch(meta)
            logits = model(inputs.to(device))
            loss = loss_fn(logits, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

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
    }
    torch.save(
        {"state_dict": model.state_dict(), "protocol": protocol},
        checkpoint_path,
    )
    return model, protocol
