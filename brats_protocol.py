"""Shared, label-independent inference and audited evaluation for both stages.

Ground truth is allowed ONLY in metrics and make_training_patch. HD95 uses mm;
its empty-region policy is custom and is not the official BraTS evaluator.
"""
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np

PROTOCOL_VERSION = "review-v3-full-volume"
MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")


def region_masks(seg):
    return {"WT": seg > 0, "TC": (seg == 1) | (seg == 3), "ET": seg == 3}


def mean_region_dice(pred, target):
    values = []
    for region, p in region_masks(pred).items():
        t = region_masks(target)[region]
        denom = int(p.sum()) + int(t.sum())
        values.append(2 * np.count_nonzero(p & t) / denom if denom else 1.0)
    return float(np.mean(values))


def segmentation_metrics(pred, target, spacing=(1., 1., 1.)):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("Metrics require matching 3D volumes")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Invalid voxel spacing")
    out = {}
    for region, p in region_masks(pred).items():
        t = region_masks(target)[region]
        ps, ts, tp = int(p.sum()), int(t.sum()), int((p & t).sum())
        union = ps + ts - tp
        if not ps and not ts:
            hd = 0.0
        elif not ps or not ts:
            # Finite penalty for a complete miss/false positive; never drop it.
            hd = float(np.linalg.norm((np.asarray(pred.shape) - 1) * spacing))
        else:
            p_surface = p & ~binary_erosion(p, structure=np.ones((3, 3, 3)), border_value=0)
            t_surface = t & ~binary_erosion(t, structure=np.ones((3, 3, 3)), border_value=0)
            distances = np.concatenate((distance_transform_edt(~t_surface, sampling=spacing)[p_surface],
                                        distance_transform_edt(~p_surface, sampling=spacing)[t_surface]))
            hd = float(np.percentile(distances, 95))
        values = {"Dice": 2 * tp / (ps + ts) if ps + ts else 1.,
                  "IoU": tp / union if union else 1.,
                  "Sensitivity": tp / ts if ts else 1.,
                  "Precision": tp / ps if ps else float(ts == 0), "HD95": hd}
        out.update({f"{k}_{region}": float(v) for k, v in values.items()})
    return out


def summarize_metric_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation set")
    keys = list(rows[0])
    arrays = {k: np.asarray([r[k] for r in rows], dtype=float) for k in keys}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Non-finite metric; do not silently omit failed cases")
    result = {k: float(v.mean()) for k, v in arrays.items()}
    result["std"] = {k: float(v.std(ddof=0)) for k, v in arrays.items()}
    result["n_valid"] = {k: int(len(v)) for k, v in arrays.items()}
    result["n_cases"] = len(rows)
    return result


def window_bounds(shape, roi=128, overlap=.25):
    shape = tuple(int(v) for v in shape)
    roi = (roi,) * 3 if isinstance(roi, int) else tuple(roi)
    if len(shape) != 3 or min(shape) <= 0 or len(roi) != 3 or min(roi) < 4:
        raise ValueError("Use a positive 3D shape and ROI >=4")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    starts = []
    sizes = [min(n, int(r)) for n, r in zip(shape, roi)]
    for n, size in zip(shape, sizes):
        step = max(1, int(size * (1 - overlap)))
        values = list(range(0, n - size + 1, step))
        if values[-1] != n - size:
            values.append(n - size)
        starts.append(values)
    return [(tuple(lo), tuple(l + s for l, s in zip(lo, sizes)))
            for lo in itertools.product(*starts)]


def probability_window(node_probs, meta, lo, hi):
    slices = tuple(slice(l, h) for l, h in zip(lo, hi))
    shape = tuple(h - l for l, h in zip(lo, hi))
    acc = np.zeros(shape + (4,), dtype=np.float32)
    count = np.zeros(shape, dtype=np.float32)
    for nt, probs in node_probs.items():
        node_map = np.asarray(meta["node_map"][nt])[slices]
        valid = node_map >= 0
        acc[valid] += np.asarray(probs, dtype=np.float32)[node_map[valid]]
        count[valid] += 1
    valid = count > 0
    acc[valid] /= count[valid, None]
    acc[~valid, 0] = 1.
    return acc.transpose(3, 0, 1, 2)


def restore_full_prediction(pred_crop, meta):
    pred = np.zeros(tuple(meta["original_shape"]), dtype=np.uint8)
    slices = tuple(slice(int(l), int(h)) for l, h in zip(meta["lo"], meta["hi"]))
    pred[slices] = pred_crop
    return pred


def sliding_window_predict(head, meta, node_probs=None, roi=128, overlap=.25):
    """Predict every MRI-brain voxel. Does not access meta['seg'] or labels."""
    import torch
    import torch.nn.functional as F
    from brats_gpu import configure_gpu, bounded_prefetch, predict_batch, _cleanup_cuda
    shape = tuple(meta["crop_shape"])
    sums = np.zeros((4,) + shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.float32)
    dev = next(head.parameters()).device
    head.eval()
    bounds = window_bounds(shape, roi, overlap)
    def prepare_window(bound):
        lo, hi = bound
        slices = tuple(slice(l, h) for l, h in zip(lo, hi))
        x = np.stack([np.asarray(meta["vis"][m], dtype=np.float32)[slices] for m in MODALITY_ORDER])
        if node_probs is not None:
            x = np.concatenate((x, probability_window(node_probs, meta, lo, hi)), axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        original = tensor.shape[1:]
        # Two downsamplings require at least four voxels on each axis.
        pad = [v for n in reversed(original) for v in (0, max(0, 4 - n))]
        tensor = F.pad(tensor, pad)
        if dev.type == "cuda":
            tensor = tensor.pin_memory()  # Preparation runs in the CPU worker.
        return tensor, original, slices
    runtime = getattr(head, "brats_gpu_runtime", None)
    if runtime is None:
        first, _, _ = prepare_window(bounds[0])
        options = getattr(head, "brats_gpu_options", {})
        runtime = configure_gpu(head, first.numpy(), roi=roi, **options)
    pending = []
    def consume(windows):
        cursor = 0
        while cursor < len(windows):
            size = min(runtime["inference_batch"], len(windows) - cursor)
            try:
                probabilities = predict_batch(head, [w[0] for w in windows[cursor:cursor + size]], runtime)
            except torch.cuda.OutOfMemoryError:
                _cleanup_cuda()
                if size == 1:
                    raise
                runtime["inference_batch"] = max(1, size // 2)
                print("[GPU] Inference OOM: reducing window batch to", runtime["inference_batch"])
                continue
            for probability, (_, original, slices) in zip(probabilities, windows[cursor:cursor + size]):
                sums[(slice(None),) + slices] += probability[:, :original[0], :original[1], :original[2]]
                count[slices] += 1
            cursor += size
    for window in bounded_prefetch(bounds, prepare_window, workers=1 if dev.type == "cuda" else 0,
                                    depth=max(2, runtime["inference_batch"])):
        pending.append(window)
        if len(pending) == runtime["inference_batch"]:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    if not np.all(count > 0):
        raise RuntimeError("Sliding-window coverage failure")
    return restore_full_prediction((sums / count[None]).argmax(0).astype(np.uint8), meta)


def make_training_patch(meta, roi=128, node_probs=None, rng=None, tumour_probability=.5):
    """50% tumour-centred / 50% uniform patches, exclusively for TRAINING.

    CNN-only and graph+CNN use the same coordinates when given the same RNG.
    Validation/test must use sliding_window_predict instead.
    """
    rng = np.random if rng is None else rng
    shape = np.asarray(meta["crop_shape"], dtype=int)
    size = np.minimum(shape, roi)
    seg = np.asarray(meta["seg"])[tuple(slice(int(l), int(h)) for l, h in zip(meta["lo"], meta["hi"]))]
    coords = np.argwhere(seg > 0) if tumour_probability > 0 else []
    if len(coords) and rng.random() < tumour_probability:
        centre = coords[int(rng.randint(len(coords)))]
        lo = np.minimum(np.maximum(centre - size // 2, 0), shape - size)
    else:
        lo = np.array([rng.randint(int(n - s) + 1) for n, s in zip(shape, size)])
    hi = lo + size
    slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    x = np.stack([np.asarray(meta["vis"][m], dtype=np.float32)[slices] for m in MODALITY_ORDER])
    if node_probs is not None:
        x = np.concatenate((x, probability_window(node_probs, meta, lo, hi)), axis=0)
    y = seg[slices].astype(np.int64)
    pads = tuple((0, max(0, 4 - int(n))) for n in y.shape)
    return np.pad(x, ((0, 0),) + pads), np.pad(y, pads)


def audit_dataset(all_paths, modality_kind, output_dir, expected_count=1251, exclusions=(), official_ids=None):
    """Audit canonical IDs, duplicate modality files, geometry and content.

    Return the canonical complete cases and a report; never pick an arbitrary
    case to discard. Hash uncompressed, float32 voxel content plus geometry.
    Exclusions require an explicit ID -> reason mapping supplied by the user.
    """
    import nibabel as nib
    exclusions = dict(exclusions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "source_hash_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    groups, errors = {}, []
    for path in sorted(set(all_paths)):
        kind = modality_kind(path)
        if not kind:
            continue
        match = re.search(r"BraTS2021_\d+", Path(path).name, re.IGNORECASE)
        if not match:
            errors.append(f"Unrecognized case ID: {Path(path).name}")
            continue
        case_id = "BraTS2021_" + match.group(0).split("_")[-1]
        groups.setdefault(case_id, {}).setdefault(kind, []).append(path)
    complete, records, content_ids = [], {}, {}
    required = (*MODALITY_ORDER, "seg")
    for case_id, files in sorted(groups.items()):
        if case_id in exclusions:
            if not str(exclusions[case_id]).strip():
                errors.append(f"Exclusion has no reason: {case_id}")
            continue
        if any(len(files.get(m, [])) != 1 for m in required):
            errors.append(f"Missing or duplicate modalities: {case_id}")
            continue
        selected = {m: files[m][0] for m in required}
        digests, geometries = {}, []
        for m, path in selected.items():
            stat = Path(path).stat()
            key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
            if key not in cache:
                image = nib.load(path)
                volume = image.get_fdata(dtype=np.float32)
                affine = np.round(image.affine, 6).astype("<f8")
                digest = hashlib.sha256(np.ascontiguousarray(volume, dtype="<f4").tobytes() + affine.tobytes()
                                        + str(volume.shape).encode()).hexdigest()
                cache[key] = {"sha256": digest, "shape": list(volume.shape),
                              "affine": affine.tolist(), "spacing": list(map(float, image.header.get_zooms()[:3]))}
            value = cache[key]
            digests[m] = value["sha256"]
            geometries.append((value["shape"], value["affine"]))
            # Legacy metric calls use the BraTS 1 mm default; verify that premise.
            if not np.allclose(value["spacing"], (1., 1., 1.)):
                errors.append(f"Non-1mm geometry requires spacing-aware calls: {case_id}/{m}")
        if any(g != geometries[0] for g in geometries[1:]):
            errors.append(f"Unaligned modalities/segmentation: {case_id}")
        mri_digest = hashlib.sha256(json.dumps({m: digests[m] for m in MODALITY_ORDER}, sort_keys=True).encode()).hexdigest()
        if mri_digest in content_ids:
            errors.append(f"Duplicate MRI content: {content_ids[mri_digest]} and {case_id}")
        content_ids[mri_digest] = case_id
        records[case_id] = digests
        complete.append((str(Path(selected["t1"]).parent), selected))
    if len(complete) != expected_count:
        errors.append(f"Expected {expected_count} patients, found {len(complete)}; investigate the manifest")
    if official_ids is not None and set(records) != set(official_ids):
        errors.append("Canonical IDs differ from the supplied official manifest")
    missing_exclusions = set(exclusions) - set(groups)
    if missing_exclusions:
        errors.append(f"Excluded IDs not found: {sorted(missing_exclusions)}")
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    report = {"protocol": PROTOCOL_VERSION, "expected_count": expected_count, "complete_count": len(complete),
              "source_fingerprint": fingerprint, "cases": records, "exclusions": exclusions,
              "official_manifest_verified": official_ids is not None, "errors": errors}
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    (output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return complete, report
