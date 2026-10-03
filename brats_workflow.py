"""Read-only Part 1 compatibility and shared continuation utilities."""
from pathlib import Path
import json
import os
import time
import torch
from brats_transfer import sha256_path
from brats_bundles import ZipStore


def download_or_none(token, repo_id, repo_type, name, revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        return hf_hub_download(repo_id=repo_id,repo_type=repo_type,filename=name,
                               revision=revision,token=token)
    except EntryNotFoundError:
        return None


def load_part1(token, repo_id, repo_type='model', manifest_name=''):
    from huggingface_hub import HfApi
    api=HfApi(token=token)
    head=api.repo_info(repo_id=repo_id,repo_type=repo_type).sha
    names=[manifest_name] if manifest_name else ['stage1_graph_model.manifest.json','stage1_graph_model_review_v3.manifest.json']
    for name in names:
        path=download_or_none(token,repo_id,repo_type,name,head)
        if path is not None: break
    else:
        raise RuntimeError('Part 1 final handoff is not present yet. Let Part 1 finish its upload; do not restart its training.')
    manifest=json.loads(Path(path).read_text())
    ckpt_name=name.replace('.manifest.json','.pt')
    revision=manifest.get('checkpoint_revision') or head
    ckpt_path=download_or_none(token,repo_id,repo_type,ckpt_name,revision)
    if ckpt_path is None or not manifest.get('sha256') or sha256_path(ckpt_path)!=manifest['sha256']:
        raise RuntimeError('Part 1 checkpoint/checksum does not match its handoff; wait for the completed upload.')
    checkpoint=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    required=('split','split_fingerprint','dataset_fingerprint','run_config','model_state_dict','class_weights')
    if any(k not in checkpoint for k in required):
        raise RuntimeError('This legacy checkpoint lacks audited split/configuration metadata. It cannot safely be relabelled as the current study; keep the running Part 1 intact and inspect its final artifact.')
    if not checkpoint.get('training_complete') or not manifest.get('training_complete'):
        raise RuntimeError('Part 1 is still incomplete; continue only after its final handoff.')
    receipt=manifest.get('graph_cache') or {}
    if not receipt.get('complete') or not receipt.get('revision') or not receipt.get('sha256'):
        raise RuntimeError('Part 1 graph ZIP handoff is incomplete. Wait for its final graph archive upload.')
    return dict(checkpoint=checkpoint,manifest=manifest,path=ckpt_path,sha256=manifest['sha256'],
                manifest_name=name,checkpoint_name=ckpt_name,head_revision=head,api=api)


def load_exact_model(model, base_model, checkpoint, identity, dataset_fingerprint, rec_module=None):
    recorded=checkpoint['run_config']['identity']
    if recorded!=identity:
        raise RuntimeError('Architecture/graph settings differ from running Part 1. Continuation must adopt its saved identity.')
    if checkpoint['dataset_fingerprint']!=dataset_fingerprint:
        raise RuntimeError('Dataset fingerprint differs from Part 1; refusing mixed patient data.')
    state=checkpoint['model_state_dict']
    # Prefix-only legacy packaging is supported when it represents every model
    # parameter. Never leave a structural head randomly initialized.
    expected=set(model.state_dict())
    if set(state)!=expected and set(state)==set(base_model.state_dict()) and model is base_model:
        model.load_state_dict(state,strict=True)
    else:
        model.load_state_dict(state,strict=True)
    if rec_module is not None:
        if checkpoint.get('rec_state_dict') is None:
            raise RuntimeError('Part 1 declares reconstruction but supplies no reconstruction weights.')
        rec_module.load_state_dict(checkpoint['rec_state_dict'],strict=True)
    return model


def make_store(context, key, root=None):
    c=context
    return ZipStore(root or Path(c['PERSISTENT_BASE'])/'continuation_v6'/key,
        'continuation_v6/'+c['STAGE1_MODEL_SHA256'][:20]+'/'+key,
        dict(stage1_sha256=c['STAGE1_MODEL_SHA256'],key=key),
        api=c['hf_api'] if c['HF_ENABLED'] else None,
        download=lambda name,rev:c['hf_try_download'](name,c['HF_MODEL_REPO_ID'],c['HF_MODEL_REPO_TYPE'],revision=rev),
        repo_id=c['HF_MODEL_REPO_ID'],repo_type=c['HF_MODEL_REPO_TYPE'],
        interval_seconds=c.get('ZIP_SYNC_MINUTES',120)*60)


def require_main_complete(store, name):
    state=store.read(name)
    if state is None or not state.get('training_complete'):
        raise RuntimeError('Finish/resume Part 2 before continuing. Its verified ZIP must contain a completed refinement checkpoint.')
    return state


def sync_main(store, base, checkpoint_name, stage1_hash):
    base=Path(base)
    if (base/checkpoint_name).is_file():
        store.add_file(base/checkpoint_name,checkpoint_name)
    store.collect(base/'node_probs'/stage1_hash[:16],'node_probs/'+stage1_hash[:16])
    store.collect(base/'figures','figures')
    if (base/'gpu_runtime.json').is_file(): store.add_file(base/'gpu_runtime.json','gpu_runtime.json')
    return store.flush()


def restore_main(store, base, checkpoint_name, stage1_hash):
    import shutil
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    names=[checkpoint_name]+[n for n in store.catalog['files'] if n.startswith('node_probs/'+stage1_hash[:16]+'/')]
    for name in names:
        path=store.pull(name)
        if path is not None:
            dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.resolve()!=path.resolve():shutil.copyfile(path,dest)


def export_reports_zip(root, destination):
    import zipfile
    root=Path(root);destination=Path(destination)
    with zipfile.ZipFile(str(destination)+'.tmp','w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.csv','.png','.md') and not any(part.startswith('.') for part in p.relative_to(root).parts):
                z.write(p,p.relative_to(root).as_posix())
    os.replace(str(destination)+'.tmp',destination)
    return destination
