"""Measured CUDA batching with bounded CPU prefetch and recoverable OOMs.

No compilation, custom CUDA kernels, architecture changes or unbounded cache.
Training uses fixed patient groups; microbatch reductions do not change the
number of optimizer updates. Precision/batch selection is logged at runtime.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import time
import numpy as np


def bounded_prefetch(iterable, prepare=lambda x: x, workers=1, depth=2):
    """Preserve order, bound outstanding tasks, propagate preparation errors."""
    if workers <= 0:
        for item in iterable:
            yield prepare(item)
        return
    iterator = iter(iterable)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(max(1, depth)):
            try:
                pending.append(pool.submit(prepare, next(iterator)))
            except StopIteration:
                break
        try:
            while pending:
                value = pending.popleft().result()
                try:
                    pending.append(pool.submit(prepare, next(iterator)))
                except StopIteration:
                    pass
                yield value
        finally:
            for future in pending:
                future.cancel()


def patient_groups(iterable, size=4):
    if size < 1:
        raise ValueError("Effective batch size must be positive")
    group = []
    for value in iterable:
        group.append(value)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


@contextmanager
def preserved_probe_state(head):
    import torch
    dev = next(head.parameters()).device
    mode = head.training
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        head.load_state_dict(state)
        head.zero_grad(set_to_none=True)
        head.train(mode)


def amp_context(runtime, device):
    import torch
    names = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    precision = runtime.get("precision", "fp32")
    return torch.autocast(device_type=device.type, dtype=names[precision],
                          enabled=device.type == "cuda" and precision != "fp32")


def _cleanup_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _probe(head, x, y, loss_fn, size, precision, training):
    import torch
    dev = next(head.parameters()).device
    tx = torch.as_tensor(x).unsqueeze(0).repeat(size, 1, 1, 1, 1).to(dev)
    ty = torch.as_tensor(y).unsqueeze(0).repeat(size, 1, 1, 1).long().to(dev) if training else None
    head.train(training)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    # One warmup plus two timed passes; include softmax/transfer for inference.
    elapsed = []
    for repeat in range(3):
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.set_grad_enabled(training), amp_context({"precision": precision}, dev):
            output = head(tx)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if training:
                loss = loss_fn(output, ty)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite calibration loss")
            else:
                probabilities = logits.float().softmax(1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("Non-finite inference probabilities")
                probabilities.cpu()
        if training:
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError("Non-finite calibration gradients")
        torch.cuda.synchronize(dev)
        if repeat:
            elapsed.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(dev)
    return {"batch_size": size, "seconds": float(np.mean(elapsed)),
            "patients_per_second": size / float(np.mean(elapsed)), "peak_bytes": peak,
            "probe_total_seconds": time.perf_counter() - started}


def configure_gpu(head, x, y=None, loss_fn=None, *, roi=128, precision="auto",
                  max_microbatch=4, max_inference_batch=8, memory_fraction=.70,
                  autotune=True, resume_precision=None):
    """Benchmark real inputs at worst-case ROI without changing learned weights.

    Candidate probes reserve 30% of currently available memory by default.
    Batch one is the fallback; only memory/finite-value failures are handled.
    Driver/library/kernel compatibility errors surface instead of being hidden.
    """
    import torch
    dev = next(head.parameters()).device
    if dev.type != "cuda":
        result = {"precision": "fp32", "microbatch": 1, "inference_batch": 1,
                  "device": str(dev), "calibration": [], "autotuned": False}
        head.brats_gpu_runtime = result
        return result
    if not .1 <= memory_fraction <= .9:
        raise ValueError("Memory fraction must be between .1 and .9")
    preferred = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    selected_precision = resume_precision or (preferred if precision == "auto" else precision)
    if selected_precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("Precision must be auto, bf16, fp16 or fp32")
    if selected_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Saved BF16 run requires a BF16-capable GPU; choose the matching molab GPU")
    result = {"precision": selected_precision, "microbatch": 1, "inference_batch": 1,
              "device": torch.cuda.get_device_name(dev), "total_vram_bytes": torch.cuda.get_device_properties(dev).total_memory,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "memory_fraction": memory_fraction,
              "calibration": [], "autotuned": bool(autotune)}
    if not autotune:
        head.brats_gpu_runtime = result
        return result
    # Probe the largest possible patch. Padding is used ONLY in calibration.
    pads = tuple((0, max(0, roi - n)) for n in x.shape[1:])
    probe_x = np.pad(np.asarray(x, dtype=np.float32), ((0, 0),) + pads)
    probe_y = np.pad(np.asarray(y, dtype=np.int64), pads) if y is not None else None
    free, _ = torch.cuda.mem_get_info(dev)
    baseline = torch.cuda.memory_allocated(dev)
    limit = baseline + int(free * memory_fraction)
    modes = [("microbatch", max_microbatch, True)] if loss_fn is not None else []
    modes.append(("inference_batch", max_inference_batch, False))
    while True:
        try:
            with preserved_probe_state(head):
                for key, maximum, training in modes:
                    records = []
                    sizes = [size for size in (1, 2, 4, 8) if size <= maximum]
                    for size in sizes:
                        try:
                            record = _probe(head, probe_x, probe_y, loss_fn, size, result["precision"], training)
                        except torch.cuda.OutOfMemoryError:
                            _cleanup_cuda()
                            result["calibration"].append({"mode": key, "batch_size": size, "status": "oom"})
                            if size == 1:
                                raise
                            break
                        record.update(mode=key, precision=result["precision"], status="ok" if record["peak_bytes"] <= limit else "memory_margin")
                        result["calibration"].append(record)
                        if record["peak_bytes"] > limit:
                            _cleanup_cuda()
                            break
                        records.append(record)
                    if records:
                        # Prefer a smaller batch unless throughput improves by >5%.
                        best = records[0]
                        for record in records[1:]:
                            if record["patients_per_second"] > best["patients_per_second"] * 1.05:
                                best = record
                        result[key] = best["batch_size"]
            break
        except FloatingPointError:
            _cleanup_cuda()
            if resume_precision or precision != "auto" or result["precision"] == "fp32":
                raise
            # Discard ALL mixed-precision timings before benchmarking FP32.
            # Probe state/RNG were restored on exit from the context above.
            result["precision"] = "fp32"
            result["microbatch"] = result["inference_batch"] = 1
            result["calibration"] = []
    _cleanup_cuda()
    head.brats_gpu_runtime = result
    print("[GPU]", result["device"], "precision=", result["precision"],
          "train microbatch=", result["microbatch"], "inference windows=", result["inference_batch"])
    return result


def predict_batch(head, batch, runtime):
    import torch
    dev = next(head.parameters()).device
    tensor = _batch_to_device(batch, dev)
    with torch.inference_mode(), amp_context(runtime, dev):
        output = head(tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probs = logits.float().softmax(1)
        if not torch.isfinite(probs).all():
            raise FloatingPointError("Non-finite inference probabilities")
    return probs.cpu().numpy()


def _batch_to_device(values, device):
    import torch
    if len(values) == 1:
        return values[0].unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    # stack() normally loses pinned storage; explicitly provide pinned output.
    storage = torch.empty((len(values),) + tuple(values[0].shape), dtype=values[0].dtype,
                          pin_memory=device.type == "cuda")
    return torch.stack(values, out=storage).to(device, non_blocking=device.type == "cuda")


def patient_dice_tensor(logits, targets):
    import torch
    pred = logits.detach().argmax(1)
    values = []
    for p, t in ((pred > 0, targets > 0), ((pred == 1) | (pred == 3), (targets == 1) | (targets == 3)),
                 (pred == 3, targets == 3)):
        dims = tuple(range(1, pred.ndim))
        denominator = p.sum(dims) + t.sum(dims)
        values.append(torch.where(denominator > 0, 2. * (p & t).sum(dims) / denominator.clamp_min(1), 1.))
    return torch.stack(values).mean(0)


def _backward_group(head, group, loss_fn, scaler, runtime):
    import torch
    dev = next(head.parameters()).device
    losses, dices = [], []
    start = 0
    while start < len(group):
        size = min(runtime["microbatch"], len(group) - start)
        # Unequal shapes are processed separately; no artificial BG padding.
        while size > 1 and any(group[start + k][0].shape != group[start][0].shape for k in range(1, size)):
            size -= 1
        examples = group[start:start + size]
        x = _batch_to_device([v[0] for v in examples], dev)
        y = _batch_to_device([v[1] for v in examples], dev).long()
        with amp_context(runtime, dev):
            output = head(x)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            loss = loss_fn(output, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; last completed checkpoint retained")
        scaler.scale(loss * (size / len(group))).backward()
        losses.append(loss.detach().float() * size)
        dices.append(patient_dice_tensor(logits, y))
        start += size
    return float(torch.stack(losses).sum().cpu()) / len(group), torch.cat(dices).cpu().tolist()


def train_patient_group(head, group, optimizer, scaler, loss_fn, runtime):
    """One update for a fixed group. CUDA OOM retries the SAME examples."""
    import torch
    dev = next(head.parameters()).device
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
    while True:
        optimizer.zero_grad(set_to_none=True)
        head.train()
        try:
            loss, dices = _backward_group(head, group, loss_fn, scaler, runtime)
            break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            _cleanup_cuda()
            if runtime["microbatch"] <= 1:
                raise
            runtime["microbatch"] = max(1, runtime["microbatch"] // 2)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, dev)
            print("[GPU] OOM: retrying identical patient group with microbatch", runtime["microbatch"])
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(head.parameters(), 2., error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return loss, dices


def train_voxel_experiment(head, train_items, val_items, prepare_case, validate_case, loss_fn, *,
                           epochs, seed, learning_rate, weight_decay, effective_batch=4,
                           max_microbatch=4, max_inference_batch=8, roi=128, precision="auto",
                           memory_fraction=.70, autotune=True, workers=2, patience=15):
    """Use the same batching/loss/AMP schedule for optional CNN controls.

    prepare_case must be CPU-only; any graph priors must be cached beforehand.
    This helper returns best weights/history; it is not a rolling resume driver.
    """
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = next(head.parameters()).device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x, y = prepare_case((train_items[0], seed))
    runtime = configure_gpu(head, x.numpy(), y.numpy(), loss_fn, roi=roi, precision=precision,
                            max_microbatch=max_microbatch, max_inference_batch=max_inference_batch,
                            memory_fraction=memory_fraction, autotune=autotune)
    del x, y
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * .02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and runtime["precision"] == "fp16")
    best_score, best_epoch, stale, best_state = -float("inf"), 0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        items = sorted(train_items, key=lambda item: str(item[1]).replace("\\", "/").split("/")[-1])
        random.shuffle(items)
        seeds = np.random.randint(0, 2**31 - 1, size=len(items))
        prepared = bounded_prefetch(zip(items, seeds), prepare_case,
                                     workers=workers if device.type == "cuda" else 0, depth=max(2, effective_batch))
        losses, dices = [], []
        try:
            for group in patient_groups(prepared, effective_batch):
                loss, group_dice = train_patient_group(head, group, optimizer, scaler, loss_fn, runtime)
                losses.extend([loss] * len(group)); dices.extend(group_dice)
        finally:
            prepared.close()
        scheduler.step()
        head.eval()
        scores = [float(validate_case(head, item)) for item in val_items]
        if not scores:
            raise RuntimeError("Optional experiment has an empty validation split")
        score = float(np.mean(scores))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),
                        "train_patch_dice":float(np.mean(dices)),"val_dice":score,
                        "epoch_seconds":time.perf_counter()-start})
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        else:
            stale += 1
        print(f"[CNN control] epoch {epoch}: patient validation Dice={score:.4f}")
        if stale >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"best_epoch":best_epoch,"best_val_dice":best_score,"history":history,
                   "effective_batch_size":effective_batch,"scheduler":"cosine",
                   "gpu_runtime":runtime,"seed":seed,"patience":patience}
