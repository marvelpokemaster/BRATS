"""The v9 GPU data path must reproduce the original CPU path exactly.

* patch_example + materialize_examples == make_training_patch (same RNG draws, same values)
* project_probabilities_torch == probability_window
* sliding_window_predict (device-resident) == sliding_window_predict_reference

Runs with plain `python test_gpu_equivalence.py` (uses CUDA when available) or pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from brats_protocol import (make_training_patch, patch_example, materialize_examples, probability_window,  # noqa: E402
                            project_probabilities_torch, sliding_window_predict, sliding_window_predict_reference,
                            training_patch_bounds)
from brats_gpu import ordered_results  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def synthetic_meta(seed, crop=(20, 26, 18), types=('t1ce', 'flair'), nodes=(37, 29), with_probs=True):
    rng = np.random.RandomState(seed)
    original = tuple(c + 10 for c in crop)
    lo = np.array([3, 5, 4]); hi = lo + np.array(crop)
    seg = np.zeros(original, np.uint8)
    inner = rng.choice([0, 0, 0, 1, 2, 3], size=crop).astype(np.uint8)
    seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = inner
    meta = dict(crop_shape=crop, original_shape=original, lo=lo, hi=hi, seg=seg,
                vis={m: rng.randn(*crop).astype(np.float16) for m in ('t1', 't1ce', 't2', 'flair')},
                node_map={nt: rng.randint(-1, k, size=crop).astype(np.int32) for nt, k in zip(types, nodes)})
    probs = None
    if with_probs:
        probs = {}
        for nt, k in zip(types, nodes):
            p = rng.rand(k, 4).astype(np.float32)
            probs[nt] = (p / p.sum(1, keepdims=True)).astype(np.float32)
    return meta, probs


def test_training_patch_equivalence():
    checked = 0
    for crop, roi in (((20, 26, 18), 16), ((20, 26, 18), 64), ((3, 26, 18), 16)):
        for with_probs in (True, False):
            meta, probs = synthetic_meta(7, crop=crop, with_probs=with_probs)
            for tumour in (0., .5, 1.):
                for seed in range(12):
                    x_ref, y_ref = make_training_patch(meta, roi, probs, rng=np.random.RandomState(seed), tumour_probability=tumour)
                    example = patch_example(meta, roi, probs, rng=np.random.RandomState(seed), tumour_probability=tumour)
                    x, y = materialize_examples([example], DEVICE)
                    assert x.dtype == torch.float32 and y.dtype == torch.int64
                    assert torch.equal(x[0].cpu(), torch.from_numpy(x_ref)), (crop, roi, with_probs, tumour, seed)
                    assert torch.equal(y[0].cpu(), torch.from_numpy(y_ref)), (crop, roi, with_probs, tumour, seed)
                    checked += 1
    assert checked == 216


def test_bounds_consume_rng_like_original():
    meta, _ = synthetic_meta(3)
    a, b = np.random.RandomState(11), np.random.RandomState(11)
    make_training_patch(meta, 16, None, rng=a)
    training_patch_bounds(meta, 16, rng=b)
    assert a.randint(10**9) == b.randint(10**9)


def test_projection_equivalence():
    meta, probs = synthetic_meta(5)
    lo, hi = (2, 3, 1), (17, 24, 15)
    ref = probability_window(probs, meta, lo, hi)
    slices = tuple(slice(l, h) for l, h in zip(lo, hi))
    got = project_probabilities_torch([torch.from_numpy(meta['node_map'][nt][slices]).to(DEVICE) for nt in probs],
                                      [torch.from_numpy(p).to(DEVICE) for p in probs.values()])
    assert torch.equal(got.cpu(), torch.from_numpy(ref))


class TinyHead(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv3d(channels, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x), None


def test_sliding_window_equivalence():
    for with_probs in (True, False):
        meta, probs = synthetic_meta(9, crop=(21, 30, 19), with_probs=with_probs)
        head = TinyHead(8 if with_probs else 4).to(DEVICE).eval()
        for roi, overlap, batch in ((8, .25, 1), (12, .5, 3), (16, .25, 16)):
            head.brats_gpu_runtime = dict(precision='fp32', microbatch=1, inference_batch=batch)
            ref = sliding_window_predict_reference(head, meta, probs, roi=roi, overlap=overlap)
            head.brats_gpu_runtime = dict(precision='fp32', microbatch=1, inference_batch=batch)
            got = sliding_window_predict(head, meta, probs, roi=roi, overlap=overlap)
            assert got.dtype == np.uint8 and got.shape == tuple(meta['original_shape'])
            mismatch = float((got != ref).mean())
            assert mismatch == 0., 'prediction mismatch fraction %.2e (roi=%s, batch=%s)' % (mismatch, roi, batch)


def test_ordered_results_keeps_order_and_bounds_backlog():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        out = ordered_results(pool, ((lambda v: v * v, i) for i in range(50)), limit=4)
    assert out == [i * i for i in range(50)]


if __name__ == '__main__':
    print('device:', DEVICE)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('PASS', name)
            except Exception as exc:
                failures += 1; print('FAIL', name, type(exc).__name__, exc)
    print('ALL PASSED' if not failures else '%d FAILED' % failures)
    sys.exit(1 if failures else 0)
