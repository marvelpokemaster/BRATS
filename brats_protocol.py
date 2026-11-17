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
CASE_ID_PATTERN = re.compile(r"BraTS2021_\d+", re.IGNORECASE)


def adopt_recorded_plan(stored, current, adopted_key, identity, allowed_source_changes):
    """Verify that a held-out evaluation may reuse a plan recorded at training time.

    The research plan key hashes the helper-source files. A later bug fix therefore
    moves the evaluation onto a new, empty store and the trained rows look missing.
    Adoption is accepted only if (a) the stored protocol hashes to the requested key,
    (b) every manifest field except ``source_hashes`` is identical, and (c) the only
    source files that differ are explicitly allowed. Returns (manifest, amendment).
    """
    if not isinstance(stored, dict):
        raise RuntimeError("Cannot adopt plan %s: its protocol.json is missing from the store" % adopted_key)
    if identity(stored)[:20] != adopted_key:
        raise RuntimeError("Stored protocol does not hash to plan %s; refusing to adopt" % adopted_key)
    fields = (set(stored) | set(current)) - {"source_hashes"}
    different = sorted(k for k in fields if identity({"v": stored.get(k)}) != identity({"v": current.get(k)}))
    if different:
        raise RuntimeError("Running configuration differs from plan %s in: %s" % (adopted_key, ", ".join(different)))
    old, new = stored.get("source_hashes", {}), current.get("source_hashes", {})
    changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    unexpected = [n for n in changed if n not in allowed_source_changes]
    if unexpected:
        raise RuntimeError("Unexpected source change(s) since training: " + ", ".join(unexpected))
    amendment = dict(adopted_key=adopted_key, recomputed_key=identity(current)[:20],
                     changed_sources={n: dict(recorded=old.get(n), running=new.get(n)) for n in changed},
                     scope="held-out evaluation only; no retraining")
    return stored, amendment


def select_main_plan_key(candidates, rows, seeds):
    """Pick the research store that holds the trained main rows.

    ``candidates`` maps a plan key to the set of file names in that store's catalog.
    A store qualifies only if every ``runs/<row>/seed_<seed>/trained.json`` exists.
    Several qualifying stores are ranked by the number of saved held-out test
    evaluations, and a tie is refused rather than guessed. Returns (key, report).
    """
    needed = {"runs/%s/seed_%s/trained.json" % (row, seed) for row in rows for seed in seeds}
    report = {}
    for key, names in candidates.items():
        names = set(names)
        report[key] = dict(trained=len(needed & names), needed=len(needed),
                           test_evaluations=sum(1 for n in names if n.startswith("evaluation/test/")),
                           missing=sorted(needed - names))
    complete = sorted(k for k, r in report.items() if r["trained"] == r["needed"])
    if not complete:
        best = sorted(report.items(), key=lambda kv: -kv[1]["trained"])[:3]
        raise RuntimeError("No research store holds all %d trained main rows. Closest: %s. Re-run Part 3 (training) "
                           "before Part 5." % (len(needed), "; ".join("%s has %d (missing %s)" % (k, r["trained"], ", ".join(r["missing"][:3])) for k, r in best) or "none found"))
    top = max(report[k]["test_evaluations"] for k in complete)
    leaders = [k for k in complete if report[k]["test_evaluations"] == top]
    if len(leaders) > 1:
        raise RuntimeError("Several research stores hold all trained main rows (%s) and none is ahead on saved test evaluations; "
                           "set MAIN_PLAN_KEY_OVERRIDE to the intended one." % ", ".join(leaders))
    return leaders[0], report


def case_id_from_path(path):
    """Canonical BraTS2021 ID parsed from the NIfTI filename, never the folder."""
    match = CASE_ID_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"Unrecognized case ID: {Path(path).name}")
    return "BraTS2021_" + match.group(0).split("_")[-1]


def select_canonical_nifti(paths):
    """Keep NIfTI files stored in a folder named after their own case ID.

    The Kaggle mirror adds BraTS2021_00495.tar / BraTS2021_00621.tar whose files
    sit at the archive root; they are byte-identical to the foldered copies in
    BraTS2021_Training_Data.tar. Extracted into a shared directory, those loose
    copies would share one parent folder, so only foldered copies are canonical.
    """
    return [p for p in paths
            if CASE_ID_PATTERN.search(Path(p).name) and Path(p).parent.name == case_id_from_path(p)]


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


def project_probabilities_torch(maps, probs, num_classes=4):
    """GPU/torch twin of probability_window over whole tensors (same float32 arithmetic).

    maps: per-node-type integer tensors (-1 = no node); probs: per-type float32 (nodes, classes),
    both in the iteration order of the node_probs dict. Returns (classes, *shape) float32.
    """
    import torch
    shape = tuple(maps[0].shape)
    dev = maps[0].device
    acc = torch.zeros(shape + (num_classes,), dtype=torch.float32, device=dev)
    count = torch.zeros(shape, dtype=torch.float32, device=dev)
    for node_map, values in zip(maps, probs):
        valid = node_map >= 0
        acc[valid] += values.to(device=dev, dtype=torch.float32)[node_map[valid].long()]
        count[valid] += 1
    valid = count > 0
    acc[valid] /= count[valid].unsqueeze(-1)
    acc[~valid, 0] = 1.
    return acc.permute(3, 0, 1, 2)


def _pad_min4(tensor):
    import torch.nn.functional as F
    pad = [v for n in reversed(tuple(tensor.shape[-3:])) for v in (0, max(0, 4 - int(n)))]
    return F.pad(tensor, pad) if any(pad) else tensor


def sliding_window_predict(head, meta, node_probs=None, roi=128, overlap=.25):
    """Predict every MRI-brain voxel. Does not access meta['seg'] or labels.

    GPU-resident version: the four MRI channels and graph priors are moved to the
    device once per case, the prior is projected on the device, and the overlapping
    window probabilities are accumulated and arg-maxed there. Windows, padding,
    softmax, accumulation order and tie-breaking match sliding_window_predict_reference.
    """
    import torch
    from brats_gpu import configure_gpu, amp_context, _cleanup_cuda
    shape = tuple(meta["crop_shape"])
    dev = next(head.parameters()).device
    head.eval()
    bounds = window_bounds(shape, roi, overlap)
    pin = dev.type == "cuda"
    def to_dev(array):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        return (tensor.pin_memory() if pin else tensor).to(dev, non_blocking=pin)
    volume = to_dev(np.stack([np.asarray(meta["vis"][m]) for m in MODALITY_ORDER])).float()
    if node_probs is not None:
        prior = project_probabilities_torch([to_dev(np.asarray(meta["node_map"][nt])) for nt in node_probs],
                                            [to_dev(np.asarray(p, dtype=np.float32)) for p in node_probs.values()])
        volume = torch.cat((volume, prior), 0)
        del prior
    def window(bound):
        slices = tuple(slice(l, h) for l, h in zip(*bound))
        return _pad_min4(volume[(slice(None),) + slices]), slices
    runtime = getattr(head, "brats_gpu_runtime", None)
    if runtime is None:
        first, _ = window(bounds[0])
        options = getattr(head, "brats_gpu_options", {})
        runtime = configure_gpu(head, first.cpu().numpy(), roi=roi, **options)
    sums = torch.zeros((4,) + shape, dtype=torch.float32, device=dev)
    count = torch.zeros(shape, dtype=torch.float32, device=dev)
    cursor = 0
    while cursor < len(bounds):
        size = min(runtime["inference_batch"], len(bounds) - cursor)
        chunk = [window(b) for b in bounds[cursor:cursor + size]]
        try:
            with torch.no_grad(), amp_context(runtime, dev):
                output = head(torch.stack([w for w, _ in chunk]))
                logits = output[0] if isinstance(output, (tuple, list)) else output
                probabilities = logits.float().softmax(1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite inference probabilities")
        except torch.cuda.OutOfMemoryError:
            del chunk
            _cleanup_cuda()
            if size == 1:
                raise
            runtime["inference_batch"] = max(1, size // 2)
            print("[GPU] Inference OOM: reducing window batch to", runtime["inference_batch"])
            continue
        for probability, (_, slices) in zip(probabilities, chunk):
            extent = tuple(s.stop - s.start for s in slices)
            sums[(slice(None),) + slices] += probability[:, :extent[0], :extent[1], :extent[2]]
            count[slices] += 1
        del chunk, probabilities, logits, output
        cursor += size
    if not bool((count > 0).all()):
        raise RuntimeError("Sliding-window coverage failure")
    pred = (sums / count[None]).argmax(0).to(torch.uint8).cpu().numpy()
    del sums, count, volume
    return restore_full_prediction(pred, meta)


def sliding_window_predict_reference(head, meta, node_probs=None, roi=128, overlap=.25):
    """Original CPU-accumulating implementation, kept as the equivalence reference for tests."""
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
    lo, hi, seg = training_patch_bounds(meta, roi, rng, tumour_probability)
    slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    x = np.stack([np.asarray(meta["vis"][m], dtype=np.float32)[slices] for m in MODALITY_ORDER])
    if node_probs is not None:
        x = np.concatenate((x, probability_window(node_probs, meta, lo, hi)), axis=0)
    y = seg[slices].astype(np.int64)
    pads = tuple((0, max(0, 4 - int(n))) for n in y.shape)
    return np.pad(x, ((0, 0),) + pads), np.pad(y, pads)


def training_patch_bounds(meta, roi=128, rng=None, tumour_probability=.5):
    """Patch corner selection shared by make_training_patch and patch_example (same RNG draws)."""
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
    return lo, lo + size, seg


def patch_example(meta, roi=128, node_probs=None, rng=None, tumour_probability=.5, pin=False):
    """Compact CPU form of make_training_patch; materialize_examples expands it on the device.

    Carries the stored (float16) MRI window, labels, node maps and node probabilities
    instead of the dense float32 tensor, so the expensive prior projection runs on the GPU.
    """
    import torch
    lo, hi, seg = training_patch_bounds(meta, roi, rng, tumour_probability)
    slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    def tensor(array):
        value = torch.from_numpy(np.ascontiguousarray(array))
        return value.pin_memory() if pin else value
    example = dict(vis=tensor(np.stack([np.asarray(meta["vis"][m])[slices] for m in MODALITY_ORDER])),
                   y=tensor(np.asarray(seg[slices], dtype=np.uint8)), maps=None, probs=None)
    if node_probs is not None:
        example["maps"] = [tensor(np.asarray(meta["node_map"][nt])[slices]) for nt in node_probs]
        example["probs"] = [tensor(np.asarray(p, dtype=np.float32)) for p in node_probs.values()]
    return example


def example_shape(example):
    return tuple(example["vis"].shape[1:]) if isinstance(example, dict) else tuple(example[0].shape[1:])


def materialize_examples(examples, device):
    """Stack compact patch examples into (x float32, y int64) on the device; equals make_training_patch."""
    import torch
    non_blocking = getattr(device, "type", str(device)) == "cuda"
    xs, ys = [], []
    for example in examples:
        x = example["vis"].to(device, non_blocking=non_blocking).float()
        if example["maps"] is not None:
            prior = project_probabilities_torch([m.to(device, non_blocking=non_blocking) for m in example["maps"]],
                                                [p.to(device, non_blocking=non_blocking) for p in example["probs"]])
            x = torch.cat((x, prior), 0)
        xs.append(_pad_min4(x))
        ys.append(_pad_min4(example["y"].to(device, non_blocking=non_blocking).long()))
    return torch.stack(xs), torch.stack(ys)


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
        try:
            case_id = case_id_from_path(path)
        except ValueError:
            errors.append(f"Unrecognized case ID: {Path(path).name}")
            continue
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
