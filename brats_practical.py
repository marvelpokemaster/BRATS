"""Predeclared compute-limited training; a budget stop is not convergence."""
import copy
import time
import numpy as np
import torch

PHASE_SECONDS = 135 * 60
PILOT_SECONDS = 45 * 60
MAIN_ROWS = ('full', 'no_voxel', 'cnn_only', 'graphsage_backbone', 'fixed_shared_hops', 'segresnet')


class PhaseLimit(RuntimeError):
    pass


class PhaseClock:
    def __init__(self, session, seconds=PHASE_SECONDS, spent=0.):
        self.session, self.seconds, self.spent = session, float(seconds), float(spent)
        self.start = time.monotonic()

    def elapsed(self):
        return self.spent + time.monotonic() - self.start

    def check(self):
        self.session.check()
        if self.elapsed() >= self.seconds:
            raise PhaseLimit('Predeclared training compute limit reached')


def train_phase(model, store, name, signature, *, epochs, patience, lr, decay,
                initial_precision, calibrate, train_epoch, validate, deadline, push_every=5):
    from brats_experiments import BudgetPause, cpu_state, rng_state, restore_rng
    dev = next(model.parameters()).device
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*.02)
    state = store.read(name)
    if state and state['signature'] != signature:
        raise RuntimeError('Compute-limited checkpoint identity mismatch')
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
    allowance = getattr(deadline, 'phase_seconds', PHASE_SECONDS)
    clock = PhaseClock(deadline, allowance, state.get('phase_seconds_used', 0.) if state else 0.)
    validated = bool(state and state.get('precision_validated'))

    def snapshot(complete=False, reason='in_progress'):
        return dict(signature=signature, epoch=epoch, model=cpu_state(model), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), best=best,
            best_score=best_score, best_epoch=best_epoch, stale=stale, history=history, complete=complete,
            runtime=runtime, precision_validated=validated, stopping_reason=reason,
            phase_seconds_limit=allowance, phase_seconds_used=clock.elapsed(),
            protocol='practical-v8-compute-limited', convergence_demonstrated=False)

    if state is None:
        store.save(name, snapshot(), push=True)
    reason = 'epoch_cap'
    try:
        clock.check()
        # Includes calibration/cache preparation in the declared phase allowance.
        runtime = calibrate(runtime['precision'] if validated else None, clock)
        validated = True
        scaler = torch.amp.GradScaler('cuda', enabled=dev.type=='cuda' and runtime['precision']=='fp16')
        if state and state.get('precision_validated'):
            scaler.load_state_dict(state['scaler'])
        store.save(name, snapshot(), push=True)
        for next_epoch in range(epoch+1, epochs+1):
            clock.check()
            if history and clock.elapsed() + 1.15*history[-1]['seconds'] >= allowance:
                reason = 'compute_budget'; break
            started = time.monotonic()
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
                **metrics, **validation))
            if score > best_score:
                best_score, best_epoch, stale, best = score, epoch, 0, cpu_state(model)
            else: stale += 1
            store.save(name, snapshot(), push=epoch % push_every==0)
            print(f'[{name}] epoch {epoch}: val Dice={score:.4f}; phase {clock.elapsed()/60:.1f}/{allowance/60:.0f} min', flush=True)
            if stale>=patience:
                reason='early_stopping'; break
    except (PhaseLimit, BudgetPause) as exc:
        reason = 'compute_budget' if isinstance(exc,PhaseLimit) else 'session_budget'
    # Read the last completed epoch; never persist a partial optimizer group/epoch.
    saved = store.read(name)
    if saved['best'] is None:
        saved.update(stopping_reason='insufficient_budget_no_validated_epoch', phase_seconds_used=clock.elapsed())
        store.save(name,saved,push=True)
        raise BudgetPause('No validated epoch fits the declared allowance. Preserve the ZIP; this study is incomplete, not converged.')
    saved.update(complete=True, stopping_reason=reason, phase_seconds_limit=allowance,
                 phase_seconds_used=clock.elapsed(), convergence_demonstrated=False)
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
        return self.net(x), None


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
