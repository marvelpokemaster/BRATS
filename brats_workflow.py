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
    from brats_graph import load_part1_bundle
    if manifest_name:
        raise ValueError('Practical v8 reads its completed ZIP catalog; leave the manifest override blank.')
    return load_part1_bundle(token,repo_id,repo_type)


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
    # Local root is per Part 1 model, so a disk that kept an earlier run never mixes its index with this one.
    return ZipStore(root or Path(c['PERSISTENT_BASE'])/'continuation_v8'/c['STAGE1_MODEL_SHA256'][:20]/key,
        'continuation_v8/'+c['STAGE1_MODEL_SHA256'][:20]+'/'+key,
        dict(stage1_sha256=c['STAGE1_MODEL_SHA256'],key=key),
        api=c['hf_api'] if c['HF_ENABLED'] else None,
        download=lambda name,rev:c['hf_try_download'](name,c['HF_MODEL_REPO_ID'],c['HF_MODEL_REPO_TYPE'],revision=rev),
        repo_id=c['HF_MODEL_REPO_ID'],repo_type=c['HF_MODEL_REPO_TYPE'],
        interval_seconds=c.get('ZIP_SYNC_MINUTES',120)*60)


def discover_main_plan_key(main_store, research_store_factory, rows, seeds, override=None):
    """Find (not guess) the research store Part 3 trained into, from the Hub catalogs.

    Only small catalog ZIPs are read; no checkpoint is downloaded and no store is
    created for a key that does not already exist on the Hub.
    """
    from brats_protocol import select_main_plan_key
    api = main_store.api
    if api is None:
        raise RuntimeError('Hugging Face is disabled; Part 5 needs the stored Part 3 results. Fill the token and rerun.')
    if override:
        return override, {override: 'override supplied; verified against the stored protocol by the runner'}
    base = main_store.prefix.rsplit('/', 1)[0] + '/research'
    entries = api.list_repo_tree(repo_id=main_store.repo_id, repo_type=main_store.repo_type,
                                 path_in_repo=base, revision=main_store.revision)
    keys = sorted(e.path.rsplit('/', 1)[-1] for e in entries if type(e).__name__ == 'RepoFolder')
    candidates = {}
    for key in keys:
        try:
            candidates[key] = set(research_store_factory(key).catalog['files'])
        except RuntimeError as exc:
            print('[Part 5] skipping research store %s: %s' % (key, exc))
    key, report = select_main_plan_key(candidates, rows, seeds)
    print('[Part 5] research stores found on the Hub:')
    for k, r in sorted(report.items()):
        print('  %s  trained %d/%d  saved test evaluations %d%s' % (k, r['trained'], r['needed'], r['test_evaluations'], '   <== using this' if k == key else ''))
    return key, report


def open_existing_store(context, prefix, root=None):
    """Read-only handle on a ZIP store that already exists on the Hub (identity taken from its catalog)."""
    import tempfile
    import zipfile
    c = context
    api = c['hf_api']
    revision = api.repo_info(repo_id=c['HF_MODEL_REPO_ID'], repo_type=c['HF_MODEL_REPO_TYPE']).sha
    source = c['hf_try_download'](prefix + '/catalog.zip', c['HF_MODEL_REPO_ID'], c['HF_MODEL_REPO_TYPE'], revision=revision)
    if source is None:
        return None
    with zipfile.ZipFile(source) as z:
        identity = json.loads(z.read('catalog.json'))['identity']
    root = root or Path(tempfile.gettempdir()) / 'brats_readonly_stores' / prefix.replace('/', '_')
    return ZipStore(root, prefix, identity, api=api,
        download=lambda name, rev: c['hf_try_download'](name, c['HF_MODEL_REPO_ID'], c['HF_MODEL_REPO_TYPE'], revision=rev),
        repo_id=c['HF_MODEL_REPO_ID'], repo_type=c['HF_MODEL_REPO_TYPE'], interval_seconds=10**9)


def find_pilot_receipt(context, main_store, source_prefix=None):
    """Part 4's SLIC pilot receipt and a factory for its research store.

    Looks in the current main store first. After Part 1 is retrained the main store
    moves to a new prefix, so the earlier validation-only pilot (its own declared
    protocol) is then read, unchanged, from the earlier continuation on the Hub.
    Returns (receipt, store_factory, source_prefix); (None, None, None) if absent.
    """
    receipt = main_store.read('slic_pilot_receipt.json')
    here = main_store.prefix.rsplit('/', 1)[0]
    if receipt is not None and source_prefix in (None, here):
        return receipt, None, here
    if main_store.api is None:
        return None, None, None
    root = here.split('/')[0]
    if source_prefix:
        prefixes = [source_prefix.strip('/')]
    else:
        entries = main_store.api.list_repo_tree(repo_id=main_store.repo_id, repo_type=main_store.repo_type,
                                                path_in_repo=root, revision=main_store.revision)
        prefixes = sorted(e.path for e in entries if type(e).__name__ == 'RepoFolder' and e.path != here)
    found = []
    for prefix in prefixes:
        store = open_existing_store(context, prefix + '/main')
        value = store.read('slic_pilot_receipt.json') if store is not None else None
        if value is not None:
            found.append((prefix, value))
    complete = [row for row in found if row[1].get('complete')] or found
    if not complete:
        return None, None, None
    if len(complete) > 1:
        raise RuntimeError('Several earlier SLIC pilot receipts exist (%s); set PILOT_SOURCE_PREFIX to choose one.'
                           % ', '.join(p for p, _ in complete))
    prefix, receipt = complete[0]
    print('[Part 5] Using the SLIC pilot receipt carried over from', prefix, '(validation-only pilot, unchanged).')
    return receipt, (lambda key: open_existing_store(context, prefix + '/research/' + key)), prefix


def describe_pilot_receipt(main_store, receipt=None):
    """Part 4's receipt. Absent/incomplete is disclosed, never fatal."""
    receipt = main_store.read('slic_pilot_receipt.json') if receipt is None else receipt
    if receipt is None:
        print('[Part 5] No SLIC pilot receipt found; the report will say the pilot was not supplied. Run Part 4 first if you need it.')
    else:
        print('[Part 5] SLIC pilot receipt: plan=%s complete=%s jobs=%s' % (receipt.get('plan'), receipt.get('complete'),
              [j.get('experiment') for j in receipt.get('status', {}).get('finished_jobs', [])]))
        if not receipt.get('complete'):
            print('[Part 5] WARNING: pilot incomplete; it cannot support a SLIC-resolution claim.')
    return receipt


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
