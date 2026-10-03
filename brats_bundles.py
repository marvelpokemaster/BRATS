"""Incremental ZIP-only Hub persistence. One atomic commit per flush, no per-file uploads.

The catalog and changed files are ZIPs. Reads pin the catalog and all archive
downloads to one Hub commit. Checkpoints are local per epoch and remote at most
every two hours plus a clean exit/pause. A hard runtime kill can lose unflushed work.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import time
import uuid
import zipfile
from brats_transfer import sha256_path, write_json_atomic


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe bundle member: ' + name)
    return str(p)


class ZipStore:
    def __init__(self, root, prefix, identity, api=None, download=None, repo_id=None,
                 repo_type='model', interval_seconds=7200, max_zip_bytes=2*1024**3):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix.strip('/'); self.identity = identity
        self.api, self.download, self.repo_id, self.repo_type = api, download, repo_id, repo_type
        self.interval = interval_seconds; self.max_zip_bytes = max_zip_bytes
        self.last_flush = time.monotonic(); self.pending = set(); self.restored = set()
        self.catalog_name = self.prefix + '/catalog.zip'
        self.revision = None; self.remote_catalog_hash = None
        self.catalog = dict(schema=1, identity=identity, files={}, archives={})
        self.local_index = self.root / '.zip_store_index.json'
        if api is not None:
            self.revision = api.repo_info(repo_id=repo_id, repo_type=repo_type).sha
            source = self.download(self.catalog_name, self.revision)
            if source is not None:
                self.remote_catalog_hash = sha256_path(source)
                with zipfile.ZipFile(source) as z:
                    self.catalog = json.loads(z.read('catalog.json'))
                if self.catalog['schema'] != 1 or self.catalog['identity'] != identity:
                    raise RuntimeError('ZIP catalog identity mismatch')
        if self.local_index.exists():
            local = json.loads(self.local_index.read_text())
            if local['identity'] != identity:
                raise RuntimeError('Local ZIP store identity mismatch')
            # Unflushed local files survive a same-disk restart; do not replace
            # them with older remote state. A fresh session has no local index.
            self.pending.update(local.get('pending', []))
        self._index()

    def _index(self):
        write_json_atomic(self.local_index, dict(identity=self.identity, pending=sorted(self.pending)))

    def path(self, name):
        p = self.root / safe_name(name)
        if not p.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Bundle path escapes its root')
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def pull(self, name):
        name = safe_name(name); target = self.path(name)
        if name in self.pending and target.exists():
            return target
        record = self.catalog['files'].get(name)
        if record is None:
            return target if target.exists() else None
        if target.exists() and sha256_path(target) == record['sha256']:
            return target
        archive_name = record['archive']
        source = self.download(archive_name, self.revision) if self.download else None
        if source is None or sha256_path(source) != self.catalog['archives'][archive_name]['sha256']:
            raise RuntimeError('Missing or corrupt ZIP archive: ' + archive_name)
        with zipfile.ZipFile(source) as z:
            if z.testzip() is not None:
                raise RuntimeError('ZIP CRC verification failed')
            # Extract only the versions referenced by the pinned catalog.
            for member in z.infolist():
                item = safe_name(member.filename)
                if member.is_dir() or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise RuntimeError('Unexpected directory/symlink in state ZIP')
                expected = self.catalog['files'].get(item)
                if expected is None or expected['archive'] != archive_name or item in self.pending:
                    continue
                dest = self.path(item); tmp = dest.with_name(dest.name + '.download.tmp')
                with z.open(member) as src, tmp.open('wb') as sink:
                    shutil.copyfileobj(src, sink, 1 << 20)
                if sha256_path(tmp) != expected['sha256']:
                    tmp.unlink(); raise RuntimeError('Bundle member SHA256 mismatch: ' + item)
                os.replace(tmp, dest)
        return target

    def read(self, name):
        path = self.pull(name)
        if path is None:
            return None
        if name.endswith('.pt'):
            import torch
            return torch.load(path, map_location='cpu', weights_only=False)
        return json.loads(path.read_text(encoding='utf-8'))

    def save(self, name, value, push=False):
        path = self.path(name)
        if name.endswith('.pt'):
            import torch
            torch.save(value, str(path)+'.tmp'); os.replace(str(path)+'.tmp', path)
        else:
            write_json_atomic(path, value)
        self.pending.add(name); self._index()
        if push:
            self.maybe_flush()

    def push(self, name):
        if not self.path(name).is_file():
            raise FileNotFoundError(name)
        self.pending.add(safe_name(name)); self._index(); self.maybe_flush()

    def add_file(self, source, name):
        dest = self.path(name)
        if Path(source).resolve() != dest.resolve():
            shutil.copyfile(source, str(dest)+'.tmp'); os.replace(str(dest)+'.tmp', dest)
        self.push(name)

    def collect(self, folder, prefix):
        folder = Path(folder)
        if not folder.exists():
            return
        for path in folder.rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp'):
                name = safe_name(prefix.rstrip('/')+'/'+path.relative_to(folder).as_posix())
                if self.path(name).resolve() != path.resolve():
                    shutil.copyfile(path, self.path(name))
                self.pending.add(name)
        self._index()

    def restore_prefix(self, prefix):
        for name in self.catalog['files']:
            if name.startswith(prefix):
                self.pull(name)

    def maybe_flush(self):
        if time.monotonic()-self.last_flush >= self.interval:
            self.flush()

    def flush(self):
        if self.api is None:
            return None
        changed = {}
        for name in sorted(self.pending):
            path = self.path(name)
            if not path.is_file():
                raise FileNotFoundError(path)
            checksum = sha256_path(path)
            if self.catalog['files'].get(name, {}).get('sha256') != checksum:
                changed[name] = dict(sha256=checksum, size=path.stat().st_size)
        if not changed:
            self.pending.clear(); self._index(); self.last_flush=time.monotonic()
            return self.revision
        from huggingface_hub import CommitOperationAdd
        catalog = json.loads(json.dumps(self.catalog))
        scratch = self.root / '.zip_upload'; scratch.mkdir(exist_ok=True)
        if shutil.disk_usage(scratch).free < sum(r['size'] for r in changed.values()) + 256*1024**2:
            raise RuntimeError('Insufficient disk for ZIP upload; local checkpoints retained')
        groups=[]; group=[]; size=0
        for name, record in changed.items():
            if group and size+record['size'] > self.max_zip_bytes:
                groups.append(group);group=[];size=0
            group.append(name);size+=record['size']
        if group: groups.append(group)
        operations=[]; archives=[]
        for group in groups:
            leaf=uuid.uuid4().hex+'.zip'; local=scratch/leaf; remote=self.prefix+'/bundles/'+leaf
            with zipfile.ZipFile(local,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
                for name in group:
                    z.write(self.path(name),name)
                    catalog['files'][name]=dict(changed[name],archive=remote)
            catalog['archives'][remote]=dict(sha256=sha256_path(local),size=local.stat().st_size)
            operations.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(local)))
            archives.append(local)
        local_catalog=scratch/'catalog.zip'
        with zipfile.ZipFile(local_catalog,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('catalog.json',json.dumps(catalog,sort_keys=True,separators=(',',':')))
        operations.append(CommitOperationAdd(path_in_repo=self.catalog_name,path_or_fileobj=str(local_catalog)))
        head=self.api.repo_info(repo_id=self.repo_id,repo_type=self.repo_type).sha
        latest=self.download(self.catalog_name,head)
        if (sha256_path(latest) if latest else None) != self.remote_catalog_hash:
            raise RuntimeError('Another session changed this catalog. Stop concurrent runs and restore the latest catalog; local files retained.')
        for attempt in range(4):
            try:
                commit=self.api.create_commit(repo_id=self.repo_id,repo_type=self.repo_type,
                    operations=operations,commit_message='BraTS continuation: consolidated ZIP checkpoint',parent_commit=head)
                break
            except Exception as exc:
                response=getattr(exc,'response',None); status=getattr(response,'status_code',None)
                if status not in (429,502,503,504) or attempt==3:
                    raise
                raw=getattr(response,'headers',{}).get('Retry-After','')
                delay=min(60,max(5,int(raw) if str(raw).isdigit() else 15*(2**attempt)))
                print(f'Hub busy ({status}); retrying bundled commit in {delay}s. Local state retained.')
                time.sleep(delay)
        revision=getattr(commit,'oid',None)
        if not revision:
            raise RuntimeError('Hub commit has no immutable revision')
        downloaded=self.download(self.catalog_name,revision)
        if downloaded is None or sha256_path(downloaded)!=sha256_path(local_catalog):
            raise RuntimeError('Committed catalog readback mismatch; not marking handoff complete')
        infos=self.api.get_paths_info(repo_id=self.repo_id,repo_type=self.repo_type,
            paths=[op.path_in_repo for op in operations[:-1]],revision=revision)
        by_name={i.path:i for i in infos}
        for remote, info in catalog['archives'].items():
            if remote not in [op.path_in_repo for op in operations[:-1]]: continue
            actual=by_name.get(remote); lfs=getattr(actual,'lfs',None)
            sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
            if actual is None or actual.size!=info['size']:
                raise RuntimeError('Uploaded ZIP size verification failed')
            if sha:
                if sha!=info['sha256']: raise RuntimeError('Uploaded ZIP SHA256 mismatch')
            else:
                back=self.download(remote,revision)
                if back is None or sha256_path(back)!=info['sha256']: raise RuntimeError('ZIP readback mismatch')
        self.catalog=catalog; self.revision=revision
        self.remote_catalog_hash=sha256_path(local_catalog)
        self.pending.clear();self._index();self.last_flush=time.monotonic()
        for path in archives+[local_catalog]: path.unlink()
        print(f'VERIFIED ZIP HANDOFF: {len(changed)} changed files, {len(groups)} data ZIP(s), commit {revision}')
        return revision
