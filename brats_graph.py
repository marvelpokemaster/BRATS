"""Adaptive graph microbatches with fixed patient-mean optimizer groups.

No GPU work at import. Calibration snapshots weights/optimizer/scaler/RNG.
Only CUDA OOM is retried; no patient is skipped and no partial update is committed.
"""
import copy
import gc
import random
import time
import numpy as np
import torch
from torch_geometric.data import Batch

GRAPH_TRAINING_PROTOCOL = 'practical-v8-patient-mean'


def patient_descriptors(data, nt, classes, cross_index, flags, function):
    """Normalize each descriptor on its own disconnected patient graph."""
    store = data[nt]
    ptr = getattr(store, 'ptr', None)
    if ptr is None:
        ptr = torch.tensor([0, len(classes)], device=classes.device)
    edge = data[nt, 'spatial', nt].edge_index
    values = []
    for lo, hi in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        keep = (edge[0] >= lo) & (edge[0] < hi)
        local = edge[:, keep] - lo
        cross_keep = (cross_index[0] >= lo) & (cross_index[0] < hi)
        cross = cross_index[:, cross_keep].clone()
        cross[0] -= lo  # descriptors use source fan-out only
        values.append(function(local, hi-lo, classes[lo:hi], cross, flags, classes.device))
    return torch.cat(values, 0)


def patient_hop_penalty(base, data):
    values = getattr(base, 'hop_regularization_by_node', {})
    if not values:
        return base.hop_regularization
    terms = []
    for nt, value in values.items():
        batch = getattr(data[nt], 'batch', None)
        if batch is None:
            terms.append(value.mean())
        else:
            terms.append(torch.stack([value[batch == i].mean() for i in range(data.num_graphs)]).mean())
    return torch.stack(terms).mean() / max(base.k_max, 1)


def patient_hierarchy_loss(logits, target, num_graphs, weights, dice_weight,
                           ce_function, dice_function, focal_function, focal_weight, gamma):
    batch = getattr(target, 'batch', None)
    if batch is None:
        batch = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    terms = []
    for i in range(num_graphs):
        keep = batch == i
        if not keep.any():
            raise ValueError('Empty node type in a patient graph')
        term = ce_function(logits[keep], target.y_frac[keep], weights)
        if focal_weight:
            term = term + focal_weight*focal_function(logits[keep], target.y_frac[keep], gamma=gamma)
        terms.append(term)
    dice = dice_function(logits.float().softmax(-1), target.y_frac, target.vol, batch, num_graphs)
    return torch.stack(terms).mean() + dice_weight*dice


def _rng(generator=None):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                generator=generator.get_state() if generator is not None else None)


def _restore_rng(state, generator=None):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    if generator is not None and state['generator'] is not None:
        generator.set_state(state['generator'])


def _clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GraphBatchEngine:
    def __init__(self, model, reconstructor, forward, segmentation, mask, reconstruction,
                 *, separate_clean=True, diagnostic=None):
        self.model, self.rec = model, reconstructor
        self.forward, self.segmentation = forward, segmentation
        self.mask, self.reconstruction = mask, reconstruction
        self.separate_clean, self.diagnostic = separate_clean, diagnostic
        self.device = next(model.parameters()).device
        self.modules = torch.nn.ModuleDict({'graph': model, **({'rec': reconstructor} if reconstructor is not None else {})})

    def release_forward_state(self):
        base = getattr(self.model, 'base_model', self.model)
        if hasattr(base, 'hop_regularization'):
            base.hop_regularization = base.hop_regularization.detach()
        base.hop_regularization_by_node = {}
        base.last_edge_gates = {}
        base.last_hop_info = {}

    def seeds(self, count, generator=None):
        dev = self.device if generator is not None else torch.device('cpu')
        return torch.randint(0, 2**31-1, (count,), generator=generator, device=dev).cpu().tolist()

    def _corrupt(self, graphs, seeds):
        originals, corruptions, infos, generators = [], [], [], []
        for graph, seed in zip(graphs, seeds):
            original = graph.clone().to(self.device)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            corruption, info = self.mask(original, self.rec, gen=generator)
            originals.append(original); corruptions.append(corruption)
            infos.append(info); generators.append(generator)
        return Batch.from_data_list(corruptions), originals, infos, generators

    def _rec_loss(self, hidden, batch, originals, infos, generators):
        terms, parts = [], {}
        for i, (original, info, generator) in enumerate(zip(originals, infos, generators)):
            local = {nt: h[batch[nt].batch == i] for nt, h in hidden.items()}
            loss, row = self.reconstruction(self.rec, local, info, original, gen=generator, device=self.device)
            terms.append(loss)
            for key, value in row.items():
                parts[key] = parts.get(key, 0.) + float(value)/len(originals)
        return torch.stack(terms).mean(), parts

    def _micro(self, graphs, seeds, runtime, teacher, lam, scaler=None, weight=1.):
        from brats_gpu import amp_context
        train = scaler is not None
        active = self.rec is not None and lam > 0
        if active and not self.separate_clean:
            batch, originals, infos, generators = self._corrupt(graphs, seeds)
        else:
            batch = Batch.from_data_list(graphs).to(self.device)
        with amp_context(runtime, self.device):
            logits, aux, hidden = self.forward(self.model, batch, teacher)
            seg = self.segmentation(logits, batch, aux)
            if not torch.isfinite(seg):
                raise FloatingPointError('Non-finite graph segmentation loss')
            diagnostic = self.diagnostic(logits, batch).detach().cpu().tolist() if self.diagnostic else []
            seg_value = float(seg.detach())
            rec_value, parts = 0., {}
            if active and not self.separate_clean:
                rec_loss, parts = self._rec_loss(hidden, batch, originals, infos, generators)
                if not torch.isfinite(rec_loss):
                    raise FloatingPointError('Non-finite graph reconstruction loss')
                rec_value = float(rec_loss.detach())
                seg = seg + lam*rec_loss
        if train:
            scaler.scale(seg*weight).backward()
        # Drop the clean graph before constructing the reconstruction forward.
        del logits, aux, hidden, seg, batch
        self.release_forward_state()
        if active and self.separate_clean:
            batch, originals, infos, generators = self._corrupt(graphs, seeds)
            with amp_context(runtime, self.device):
                corrupt_logits, corrupt_aux, hidden = self.forward(self.model, batch, 0.)
                rec_loss, parts = self._rec_loss(hidden, batch, originals, infos, generators)
                if not torch.isfinite(rec_loss):
                    raise FloatingPointError('Non-finite graph reconstruction loss')
                rec_value = float(rec_loss.detach())
            if train and rec_loss.requires_grad:
                scaler.scale(lam*rec_loss*weight).backward()
            del batch, originals, infos, generators, hidden, rec_loss, corrupt_logits, corrupt_aux
            self.release_forward_state()
        return dict(total=seg_value+lam*rec_value, seg=seg_value, rec=rec_value,
                    parts=parts, dices=diagnostic)

    def _attempt(self, group, seeds, runtime, teacher, lam, scaler=None):
        summary = dict(total=0., seg=0., rec=0., parts={}, dices=[])
        for start in range(0, len(group), runtime['microbatch']):
            graphs = group[start:start+runtime['microbatch']]
            weight = len(graphs)/len(group)
            row = self._micro(graphs, seeds[start:start+len(graphs)], runtime, teacher, lam, scaler, weight)
            for key in ('total', 'seg', 'rec'):
                summary[key] += weight*row[key]
            for key, value in row['parts'].items():
                summary['parts'][key] = summary['parts'].get(key, 0.) + weight*value
            summary['dices'].extend(row['dices'])
        return summary

    def train_group(self, group, optimizer, scaler, runtime, *, teacher=0., lam=0., generator=None, seeds=None):
        if not group:
            raise ValueError('Empty effective graph group')
        seeds = self.seeds(len(group), generator) if seeds is None else seeds
        state = _rng()  # seeds are consumed once, including on an OOM retry
        while True:
            optimizer.zero_grad(set_to_none=True)
            self.modules.train()
            failed = False
            try:
                result = self._attempt(group, seeds, runtime, teacher, lam, scaler)
            except torch.OutOfMemoryError:
                failed = True
            if not failed:
                break
            # Outside the except block: failed-step traceback/tensors can be freed.
            optimizer.zero_grad(set_to_none=True)
            self.release_forward_state(); _clean(); _restore_rng(state)
            if runtime['microbatch'] == 1:
                raise torch.OutOfMemoryError('One patient graph does not fit. Last completed checkpoint is retained; no patient was skipped.')
            runtime['microbatch'] = max(1, runtime['microbatch']//2)
            runtime['oom_retries'] = runtime.get('oom_retries', 0)+1
            print('[Graph GPU] Retrying same optimizer group at microbatch', runtime['microbatch'], flush=True)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 2., error_if_nonfinite=not scaler.is_enabled())
        # A CUDA OOM during optimizer.step is not retried: it can partially update parameters.
        scaler.step(optimizer); scaler.update()
        return result

    @torch.no_grad()
    def evaluate_group(self, group, runtime, *, lam=0., generator=None):
        self.modules.eval()
        seeds = self.seeds(len(group), generator)
        state = _rng()
        while True:
            failed = False
            try:
                result = self._attempt(group, seeds, runtime, 0., lam)
            except torch.OutOfMemoryError:
                failed = True
            if not failed:
                return result
            self.release_forward_state(); _clean(); _restore_rng(state)
            if runtime['microbatch'] == 1:
                raise torch.OutOfMemoryError('One validation graph does not fit')
            runtime['microbatch'] = max(1, runtime['microbatch']//2)

    def calibrate(self, graphs, *, effective_batch=16, max_microbatch=16, memory_fraction=.75,
                  lam=0., teacher=0., optimizer=None, scaler=None, precision=None, deadline=None):
        if effective_batch < 1 or max_microbatch < 1 or not 0.1 <= memory_fraction <= .9:
            raise ValueError('Invalid graph batching/memory settings')
        precision = precision or ('fp16' if self.device.type == 'cuda' else 'fp32')
        runtime = dict(protocol=GRAPH_TRAINING_PROTOCOL, precision=precision, effective_batch=effective_batch,
                       microbatch=1, memory_fraction=memory_fraction, calibration=[], oom_retries=0,
                       torch_version=torch.__version__, cuda_version=torch.version.cuda)
        if self.device.type != 'cuda':
            return runtime
        from brats_gpu import preserved_probe_state
        maximum = min(effective_batch, max_microbatch, len(graphs))
        # Dense edge activation size dominates this architecture; include node count as tie-breaker.
        largest = sorted(graphs, key=lambda g: (g.num_edges, g.num_nodes), reverse=True)[:maximum]
        optimizer = optimizer or torch.optim.AdamW(self.modules.parameters(), lr=.001)
        scaler = scaler or torch.amp.GradScaler('cuda', enabled=precision=='fp16')
        opt_state = copy.deepcopy(optimizer.state_dict())
        scale_state = copy.deepcopy(scaler.state_dict())
        rng = _rng()
        records = []
        self.release_forward_state(); _clean()
        free, total = torch.cuda.mem_get_info(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        limit = baseline + int(free*memory_fraction)
        runtime.update(device=torch.cuda.get_device_name(self.device), total_vram_bytes=total,
                       available_vram_bytes=free, allocation_limit_bytes=limit)
        sizes = sorted(set([1]+[n for n in (2,4,8,16,32) if n <= maximum]+[maximum]))
        try:
            for size in sizes:
                if deadline is not None:
                    deadline.check()
                failed = False
                started = time.perf_counter()
                try:
                    with preserved_probe_state(self.modules):
                        self.modules.train()
                        runtime['microbatch'] = size
                        torch.cuda.reset_peak_memory_stats(self.device)
                        torch.cuda.synchronize(self.device)
                        # Two complete steps include Adam state allocation and a warmup pass.
                        for repeat in range(2):
                            optimizer.zero_grad(set_to_none=True)
                            seeds = self.seeds(size)
                            self._attempt(largest[:size], seeds, runtime, teacher, lam, scaler)
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.modules.parameters(), 2.)
                            scaler.step(optimizer); scaler.update()
                            torch.cuda.synchronize(self.device)
                            if repeat == 0:
                                measured_start = time.perf_counter()
                        seconds = time.perf_counter()-measured_start
                        peak = torch.cuda.max_memory_allocated(self.device)
                        reserved = torch.cuda.max_memory_reserved(self.device)
                except torch.OutOfMemoryError:
                    failed = True
                finally:
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.load_state_dict(copy.deepcopy(opt_state))
                    scaler.load_state_dict(copy.deepcopy(scale_state))
                    self.release_forward_state(); _restore_rng(rng); _clean()
                if failed:
                    runtime['calibration'].append(dict(batch_size=size, status='oom'))
                    if size == 1:
                        raise torch.OutOfMemoryError('Calibration: even one large training graph does not fit the visible GPU.')
                    break
                row = dict(batch_size=size, status='ok' if peak <= limit else 'memory_margin',
                           peak_bytes=peak, peak_reserved_bytes=reserved, seconds=seconds,
                           patients_per_second=size/seconds, probe_seconds=time.perf_counter()-started)
                runtime['calibration'].append(row)
                print('[Graph GPU] probe', size, row['status'], f'{peak/1024**3:.2f} GiB', flush=True)
                if peak > limit:
                    if size == 1:
                        raise RuntimeError('One graph violates the configured memory headroom. Review available GPU memory before training.')
                    break
                records.append(row)
        finally:
            self.release_forward_state(); _restore_rng(rng); _clean()
        best = records[0]
        for row in records[1:]:
            if row['patients_per_second'] > best['patients_per_second']*1.05:
                best = row
        runtime['microbatch'] = best['batch_size']
        print('[Graph GPU] selected physical batch', runtime['microbatch'], 'effective optimizer batch', effective_batch, flush=True)
        return runtime


def publish_part1_bundle(root, repo_id, repo_type, api, download, checkpoint_path,
                         checkpoint_name, manifest_name, checksum, manifest):
    """One ZIP commit containing the completed checkpoint and its manifest."""
    from pathlib import Path
    from brats_bundles import ZipStore
    from brats_transfer import write_json_atomic
    root = Path(root)
    store = ZipStore(root/'practical_v8_final', 'stage1_practical_v8/final',
                     dict(protocol=GRAPH_TRAINING_PROTOCOL), api=api, download=download,
                     repo_id=repo_id, repo_type=repo_type, interval_seconds=7200)
    value = dict(manifest, sha256=checksum, size_bytes=Path(checkpoint_path).stat().st_size,
                 graph_training_protocol=GRAPH_TRAINING_PROTOCOL)
    pointer = root/manifest_name
    write_json_atomic(pointer, value)
    store.add_file(checkpoint_path, checkpoint_name)
    store.add_file(pointer, manifest_name)
    store.collect(root/'figures','figures')
    runtime = root/'graph_gpu_runtime.json'
    if runtime.exists():
        store.add_file(runtime,'graph_gpu_runtime.json')
    store.flush()


def load_part1_bundle(token, repo_id, repo_type='model'):
    """Read the final checkpoint/manifest together from a pinned ZIP catalog."""
    from pathlib import Path
    import tempfile
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from brats_bundles import ZipStore
    from brats_transfer import sha256_path
    api = HfApi(token=token)
    def download(name, revision):
        try:
            return hf_hub_download(repo_id=repo_id,repo_type=repo_type,filename=name,revision=revision,token=token)
        except EntryNotFoundError:
            return None
    store = ZipStore(Path(tempfile.gettempdir())/'brats_practical_v8_input'/repo_id.replace('/','_'),
        'stage1_practical_v8/final',dict(protocol=GRAPH_TRAINING_PROTOCOL),
        api=api,download=download,repo_id=repo_id,repo_type=repo_type)
    name = 'stage1_graph_model.manifest.json'
    manifest = store.read(name)
    if manifest is None:
        raise RuntimeError('Finish practical-v8 Part 1 and its final ZIP handoff before running Parts 2–5.')
    checkpoint_name = name.replace('.manifest.json','.pt')
    path = store.pull(checkpoint_name)
    if path is None or sha256_path(path) != manifest['sha256']:
        raise RuntimeError('Adaptive Part 1 bundled checkpoint checksum mismatch')
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    required = ('split','split_fingerprint','dataset_fingerprint','run_config','model_state_dict','class_weights')
    if any(key not in checkpoint for key in required):
        raise RuntimeError('Adaptive Part 1 checkpoint lacks the persisted split/configuration required downstream.')
    if (checkpoint['run_config']['train'].get('graph_training_protocol') != GRAPH_TRAINING_PROTOCOL
            or checkpoint['run_config']['identity']['structural'].get('normalization') != 'per-patient-v7'):
        raise RuntimeError('Adaptive package requires freshly trained v7 graph weights and normalization.')
    if not checkpoint.get('training_complete') or not manifest.get('training_complete'):
        raise RuntimeError('Adaptive Part 1 training is incomplete')
    receipt = manifest.get('graph_cache',{})
    if not receipt.get('complete') or not receipt.get('revision') or not receipt.get('sha256'):
        raise RuntimeError('Part 1 graph ZIP receipt is incomplete')
    return dict(checkpoint=checkpoint,manifest=manifest,checkpoint_name=checkpoint_name,
                manifest_name=name,sha256=manifest['sha256'],api=api,revision=store.revision,path=str(path))
