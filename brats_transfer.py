"""Verified Hub uploads and atomic ZIP extraction, with no token storage."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote(api, repo_id, repo_type, remote_name, revision, checksum, size, download):
    # The official API uses paths=, not path_in_repo=.
    infos = api.get_paths_info(repo_id=repo_id, repo_type=repo_type, paths=[remote_name], revision=revision)
    if len(infos) != 1 or infos[0].size != size:
        raise RuntimeError(f"Missing file or remote size mismatch: {remote_name}")
    lfs = getattr(infos[0], "lfs", None)
    remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if remote_sha:
        if remote_sha != checksum:
            raise RuntimeError(f"Remote SHA256 mismatch: {remote_name}")
    else:
        # Small Git files have a Git blob hash rather than an LFS SHA256.
        path = download(repo_id=repo_id, repo_type=repo_type, filename=remote_name,
                        revision=revision, force_download=True)
        if sha256_path(path) != checksum:
            raise RuntimeError(f"Remote readback checksum mismatch: {remote_name}")


def upload_verified(api, download, local_path, remote_name, repo_id, repo_type, message):
    checksum, size = sha256_path(local_path), os.path.getsize(local_path)
    commit = api.upload_file(path_or_fileobj=local_path, path_in_repo=remote_name,
                             repo_id=repo_id, repo_type=repo_type, commit_message=message)
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError("Hub upload returned no commit ID; refusing an unpinned handoff")
    verify_remote(api, repo_id, repo_type, remote_name, revision, checksum, size, download)
    return {"repo_id": repo_id, "repo_type": repo_type, "remote_name": remote_name,
            "revision": revision, "sha256": checksum, "size_bytes": size}


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def extract_verified_zip(archive, destination, prefix, expected_sha):
    """Check the complete ZIP before extraction. Reject escaping/symlink paths."""
    if sha256_path(archive) != expected_sha:
        raise RuntimeError("Graph archive SHA256 mismatch; no extraction performed")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/") + "/"
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
        if not members:
            raise RuntimeError("Graph archive has no entries under the expected cache prefix")
        targets = []
        for member in members:
            relative = PurePosixPath(member.filename[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
                raise RuntimeError("Unsafe graph archive path")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError("Symlink graph archive entry rejected")
            target = (destination / str(relative)).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("Graph archive path leaves cache directory")
            targets.append((member, target))
        for member, target in targets:
            # Always replace from the verified archive; do not trust file size.
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            try:
                with z.open(member) as source, open(temporary, "wb") as sink:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        sink.write(chunk)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
    return len(targets)
