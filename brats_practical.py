"""Convergence-protocol training (v9).

Every main row trains to an epoch cap with early stopping on validation Dice and
keeps the best-validation checkpoint. There is no training-time cap. A session
that runs out of time saves the last completed epoch and raises BudgetPause; the
next session resumes from it. A pause never marks training complete.
The separate SLIC pilot keeps its own declared per-phase allowance.
"""
import copy
import time
import numpy as np
import torch

COMPUTE_PROTOCOL = 'convergence-v9'
PHASE_SECONDS = None          # no per-phase training-time cap for main rows
PILOT_SECONDS = 45 * 60       # validation-only SLIC pilot keeps its declared allowance
EPOCH_SAFETY = 1.15           # pause instead of starting an epoch that cannot finish
MAIN_ROWS = ('full', 'no_voxel', 'cnn_only', 'graphsage_backbone', 'fixed_shared_hops', 'segresnet')
FINAL_REASONS = ('early_stopping', 'epoch_cap')


class PhaseLimit(RuntimeError):
    pass


class PhaseClock:
    """Cumulative training seconds across sessions; optional pilot-only allowance."""
    def __init__(self, session, seconds=None, spent=0.):
        self.session = session
        self.seconds = None if seconds is None else float(seconds)
        self.spent = float(spent)
        self.start = time.monotonic()

    def elapsed(self):
        return self.spent + time.monotonic() - self.start

    def remaining(self):
        return self.session.remaining() if hasattr(self.session, 'remaining') else None

    def check(self):
        self.session.check()
        if self.seconds is not None and self.elapsed() >= self.seconds:
            raise PhaseLimit('Predeclared pilot training allowance reached')


def train_phase(model, store, name, signature, *, epochs, patience, lr, decay,
                initial_precision, calibrate, train_epoch, validate, deadline, push_every=5):
    from brats_experiments import BudgetPause, cpu_state, rng_state, restore_rng
    dev = next(model.parameters()).device
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*.02)
    state = store.read(name)
    if state and state['signature'] != signature:
        raise RuntimeError('Training checkpoint identity mismatch for ' + name)
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        if state['complete']:
            model.load_state_dict(state['best']); model.eval()
            model.brats_gpu_runtime = state['runtime']
            return state
    runtime = state['runtime'] if state else dict(precision=initial_precision, microbatch=1, inference_batch=1)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type=='cuda' and runtime['precision']=='fp16')
    if state and state.get('precision_validated'):
        scaler.load_state_dict(state['scaler'])
    epoch = state['epoch'] if state else 0
    history = copy.deepcopy(state['history']) if state else []
    best = state['best'] if state else None
    best_score = state['best_score'] if state else -1.
    best_epoch = state['best_epoch'] if state else 0
    stale = state['stale'] if state else 0
    sessions = (state.get('sessions', 0) if state else 0) + 1
    allowance = getattr(deadline, 'phase_seconds', PHASE_SECONDS)
    clock = PhaseClock(deadline, allowance, state.get('phase_seconds_used', 0.) if state else 0.)
    validated = bool(state and state.get('precision_validated'))
    protocol = COMPUTE_PROTOCOL if allowance is None else 'practical-v8-compute-limited'

    def snapshot(complete=False, reason='in_progress'):
        return dict(signature=signature, epoch=epoch, model=cpu_state(model), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), best=best,
            best_score=best_score, best_epoch=best_epoch, stale=stale, history=history, complete=complete,
            runtime=runtime, precision_validated=validated, stopping_reason=reason,
            phase_seconds_limit=allowance, phase_seconds_used=clock.elapsed(), training_seconds=clock.elapsed(),
            sessions=sessions, epoch_cap=epochs, patience=patience, protocol=protocol,
            convergence_demonstrated=False)

    if state is None:
        store.save(name, snapshot(), push=True)
    reason = 'epoch_cap'
    try:
        clock.check()
        if stale >= patience:
            # A previous session saved the patience-triggering epoch but stopped before finalizing.
            reason = 'early_stopping'
        else:
            # Includes calibration/cache preparation; repeated on every resumed session.
            runtime = calibrate(runtime['precision'] if validated else None, clock)
            validated = True
            scaler = torch.amp.GradScaler('cuda', enabled=dev.type=='cuda' and runtime['precision']=='fp16')
            if state and state.get('precision_validated'):
                scaler.load_state_dict(state['scaler'])
            store.save(name, snapshot(), push=True)
            for next_epoch in range(epoch+1, epochs+1):
                clock.check()
                if history:
                    need = EPOCH_SAFETY * max(row['seconds'] for row in history[-3:])
                    if allowance is not None and clock.elapsed() + need >= allowance:
                        reason = 'compute_budget'; break
                    left = clock.remaining()
                    if left is not None and need >= left:
                        raise BudgetPause('%s: the next epoch (~%.0f min) does not fit the %.0f min left in this session. '
                                          'Progress is saved at epoch %d; rerun this notebook to resume.' % (name, need/60, left/60, epoch))
                started = time.monotonic()
                lr_used = optimizer.param_groups[0]['lr']
                if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
                model.train()
                metrics = train_epoch(next_epoch, optimizer, scaler, runtime, clock)
                model.eval()
                validation = validate(clock)
                validation = validation if isinstance(validation, dict) else dict(val_dice=float(validation))
                score = float(validation['val_dice'])
                if not np.isfinite(score): raise FloatingPointError('Non-finite validation Dice')
                scheduler.step(); epoch = next_epoch
                history.append(dict(epoch=epoch, seconds=time.monotonic()-started,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else 0,
                    lr=lr_used, **metrics, **validation))
                if score > best_score:
                    best_score, best_epoch, stale, best = score, epoch, 0, cpu_state(model)
                else: stale += 1
                store.save(name, snapshot(), push=epoch % push_every==0)
                pace = float(np.mean([row['seconds'] for row in history[-3:]]))
                print(f'[{name}] epoch {epoch}/{epochs}: val Dice={score:.4f} (best {best_score:.4f} @ {best_epoch}, '
                      f'stale {stale}/{patience}); {history[-1]["seconds"]/60:.1f} min/epoch; '
                      f'at most {pace*(epochs-epoch)/3600:.1f} h more to the epoch cap', flush=True)
                if stale>=patience:
                    reason='early_stopping'; break
    except PhaseLimit:
        reason = 'compute_budget'
    except BudgetPause:
        # Keep the last completed epoch; never finalize on a session pause.
        if store.read(name) is not None:
            store.push(name)
        raise
    # Read the last completed epoch; never persist a partial optimizer group/epoch.
    saved = store.read(name)
    if saved['best'] is None:
        saved.update(stopping_reason='insufficient_budget_no_validated_epoch', phase_seconds_used=clock.elapsed())
        store.save(name,saved,push=True)
        raise BudgetPause('No validated epoch was completed. Preserve the ZIP and resume; this study is incomplete.')
    saved.update(complete=True, stopping_reason=reason, phase_seconds_limit=allowance,
                 phase_seconds_used=clock.elapsed(), training_seconds=clock.elapsed(), sessions=sessions,
                 epoch_cap=epochs, patience=patience, protocol=protocol,
                 convergence_demonstrated=reason == 'early_stopping')
    store.save(name,saved,push=True)
    model.load_state_dict(saved['best']); model.eval()
    model.brats_gpu_runtime = saved['runtime']
    return saved


class SegResNetBaseline(torch.nn.Module):
    """MONAI SegResNet with its documented default depth, 8 initial filters."""
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.net = SegResNet(spatial_dims=3, init_filters=8, in_channels=in_channels,
                            out_channels=num_classes, blocks_down=(1,2,2,4), blocks_up=(1,1,1))
        self.channels_last_3d=False
        self.amp_enabled=False
        self.amp_dtype=None

    def forward(self,x):
        # SegResNet downsamples 3x, so pad every spatial dim to a multiple of 8, then crop back.
        _size = x.shape[2:]
        _pad = [v for n in reversed(_size) for v in (0, (-n) % 8)]
        _out = self.net(torch.nn.functional.pad(x, _pad) if any(_pad) else x)
        return _out[(slice(None), slice(None)) + tuple(slice(0, n) for n in _size)], None


def run_slic_pilot(context, settings, main_store):
    """Separate validation-only pilot, including fresh 15k training on the same subset."""
    from brats_experiments import ResearchRunner, BudgetPause
    c = dict(context, stage1_checkpoint=None, MAIN_STAGE2_STATE=None, EPOCHS=20)
    s = dict(settings, pilot=True, seeds=(c['SEED'],), mode='train', jobs=(), active_seeds=(),
             plan_names=('full','slic_5000','slic_10000','slic_20000'), max_cases_per_split=0)
    runner = ResearchRunner(c,s)
    runner.deadline.phase_seconds=PILOT_SECONDS
    receipt=dict(plan=runner.key,scope='validation-only graph pilot; 64 train / 16 validation; no test subjects',
                 complete=False,conclusive_resolution_selection=False)
    try:
        status=runner.run()
        if status['all_training_complete']:
            runner.settings['mode']='validation'
            status=runner.run()
            receipt['complete']=bool(status.get('report_ready'))
        receipt['status']=status
    except BudgetPause as exc:
        receipt['message']=str(exc)
    finally:
        runner.store.flush()
        main_store.save('slic_pilot_receipt.json',receipt)
        main_store.flush()
    print('SLIC pilot:',receipt,flush=True)
    return receipt
