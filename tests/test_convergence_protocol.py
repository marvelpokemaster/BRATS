"""Convergence protocol v9: stopping reasons, session pause/resume, pilot allowance.

Runs with plain `python tests/test_convergence_protocol.py` (no pytest needed) or pytest.
CPU only; needs torch. Each test uses its own temporary ArtifactStore.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from brats_experiments import ArtifactStore, BudgetPause, Deadline  # noqa: E402
from brats_practical import train_phase, PhaseClock  # noqa: E402


class FakeDeadline:
    """Session deadline that can trip on the n-th check or report little time left."""
    def __init__(self, trip_at=None, remaining=10**9, phase_seconds=None):
        self.calls, self.trip_at, self.left = 0, trip_at, remaining
        if phase_seconds is not None:
            self.phase_seconds = phase_seconds

    def check(self):
        self.calls += 1
        if self.trip_at is not None and self.calls >= self.trip_at:
            raise BudgetPause('fake session limit')

    def remaining(self):
        return self.left() if callable(self.left) else self.left


def make_job(scores, sleep=0.):
    """Linear model trained on random data; validation Dice follows `scores[epoch]`."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    seen = {'epoch': 0, 'train_calls': 0}

    def train_epoch(epoch, opt, scaler, runtime, deadline):
        seen['epoch'] = epoch; seen['train_calls'] += 1
        for _ in range(4):
            deadline.check()
            x = torch.randn(8, 3)
            loss = model(x).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if sleep:
            time.sleep(sleep)
        return dict(train_loss=float(loss))

    def validate(deadline):
        return dict(val_dice=float(scores[seen['epoch']]), val_loss=0.)

    def calibrate(precision, deadline):
        return dict(precision='fp32', microbatch=1, inference_batch=1)
    return model, train_epoch, validate, calibrate, seen


def run(store, scores, epochs, patience, deadline, sleep=0.):
    model, train_epoch, validate, calibrate, seen = make_job(scores, sleep)
    state = train_phase(model, store, 'job.pt', 'sig', epochs=epochs, patience=patience, lr=.01, decay=0.,
                        initial_precision='fp32', calibrate=calibrate, train_epoch=train_epoch,
                        validate=validate, deadline=deadline, push_every=1)
    return model, state, seen


def test_early_stopping_is_final_and_reported():
    scores = {e: (0.5 + 0.05 * e if e <= 5 else 0.6) for e in range(1, 51)}
    with tempfile.TemporaryDirectory() as tmp:
        _, state, _ = run(ArtifactStore(tmp, 'x'), scores, epochs=50, patience=3, deadline=FakeDeadline())
    assert state['complete'] and state['stopping_reason'] == 'early_stopping'
    assert state['best_epoch'] == 5 and state['epoch'] == 8
    assert state['convergence_demonstrated'] is True
    assert state['phase_seconds_limit'] is None and state['protocol'] == 'convergence-v9'
    assert [row['epoch'] for row in state['history']] == list(range(1, 9))


def test_epoch_cap_when_still_improving():
    scores = {e: 0.1 * e for e in range(1, 7)}
    with tempfile.TemporaryDirectory() as tmp:
        _, state, _ = run(ArtifactStore(tmp, 'x'), scores, epochs=6, patience=3, deadline=FakeDeadline())
    assert state['stopping_reason'] == 'epoch_cap' and state['epoch'] == 6 and state['best_epoch'] == 6
    assert state['convergence_demonstrated'] is False


def test_session_pause_never_finalizes_and_resume_matches_uninterrupted_run():
    scores = {e: 0.5 + 0.01 * e for e in range(1, 11)}
    with tempfile.TemporaryDirectory() as tmp:
        model_a, state_a, _ = run(ArtifactStore(tmp, 'x'), scores, epochs=10, patience=100, deadline=FakeDeadline())
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp, 'x')
        # 1 check at start + 1 per epoch loop + 4 per train_epoch: trips inside epoch 4.
        try:
            run(store, scores, epochs=10, patience=100, deadline=FakeDeadline(trip_at=1 + 3 * 5 + 3))
            raise AssertionError('expected BudgetPause')
        except BudgetPause:
            pass
        saved = store.read('job.pt')
        assert saved['complete'] is False and saved['epoch'] == 3 and saved['stopping_reason'] == 'in_progress'
        model_b, state_b, seen = run(store, scores, epochs=10, patience=100, deadline=FakeDeadline())
    assert state_b['complete'] and state_b['stopping_reason'] == 'epoch_cap' and state_b['sessions'] == 2
    assert seen['train_calls'] == 7  # epochs 4..10 only
    assert [r['epoch'] for r in state_b['history']] == [r['epoch'] for r in state_a['history']]
    assert [r['val_dice'] for r in state_b['history']] == [r['val_dice'] for r in state_a['history']]
    for key, value in state_a['best'].items():
        assert torch.equal(value, state_b['best'][key]), key
    assert state_b['best_epoch'] == state_a['best_epoch'] == 10


def test_preemptive_pause_when_next_epoch_does_not_fit():
    scores = {e: 0.5 + 0.01 * e for e in range(1, 11)}
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp, 'x')
        try:
            run(store, scores, epochs=10, patience=100, deadline=FakeDeadline(remaining=0.), sleep=.05)
            raise AssertionError('expected BudgetPause')
        except BudgetPause as exc:
            assert 'rerun this notebook to resume' in str(exc)
        saved = store.read('job.pt')
        assert saved['complete'] is False and saved['epoch'] == 1


def test_resume_finalizes_saved_patience_without_training():
    scores = {e: (0.9 if e == 1 else 0.1) for e in range(1, 21)}
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(tmp, 'x')
        try:
            run(store, scores, epochs=20, patience=3, deadline=FakeDeadline(trip_at=1 + 3 * 5 + 3))
        except BudgetPause:
            pass
        saved = store.read('job.pt')
        saved['stale'] = 3  # as if the patience-triggering epoch was saved just before the session died
        store.save('job.pt', saved)
        _, state, seen = run(store, scores, epochs=20, patience=3, deadline=FakeDeadline())
    assert state['stopping_reason'] == 'early_stopping' and seen['train_calls'] == 0


def test_pilot_allowance_still_applies():
    scores = {e: 0.1 * e for e in range(1, 21)}
    with tempfile.TemporaryDirectory() as tmp:
        _, state, _ = run(ArtifactStore(tmp, 'x'), scores, epochs=20, patience=50,
                          deadline=FakeDeadline(phase_seconds=.8), sleep=.3)
    assert state['stopping_reason'] == 'compute_budget' and state['phase_seconds_limit'] == .8
    assert state['epoch'] == 2 and state['protocol'] == 'practical-v8-compute-limited'


def test_deadline_remaining_and_clock():
    deadline = Deadline(1.0)
    assert 3500 < deadline.remaining() <= 3600
    clock = PhaseClock(deadline, None, spent=10.)
    assert clock.elapsed() >= 10. and clock.remaining() > 3500
    clock.check()


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('PASS', name)
            except Exception as exc:  # report every test
                failures += 1; print('FAIL', name, type(exc).__name__, exc)
    print('ALL PASSED' if not failures else '%d FAILED' % failures)
    sys.exit(1 if failures else 0)
