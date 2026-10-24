"""Full JSON publication: immutable research results and replaceable metadata.

Results are staged and fsynced in the destination directory, then linked into
place atomically. Creating a hard link fails if the destination already exists;
there is deliberately no overwrite or partial-copy fallback. This requires a
local filesystem supporting hard links (including NTFS and normal POSIX filesystems).
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path


def _remove_own_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # A cleanup problem must not hide the publication error or invalidate a
        # complete published result. Never remove another writer's staging file.
        warnings.warn(f"Could not clean JSON staging file {path}: {exc}", RuntimeWarning)


def _stage_json(path: Path, payload: dict) -> Path:
    raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _remove_own_temporary(temporary)
        raise
    return temporary


def publish_json(path: Path, payload: dict) -> None:
    """Publish one complete result or fail; an existing target is never replaced."""
    path = Path(path)
    temporary = _stage_json(path, payload)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite an existing result: {path}") from exc
    finally:
        _remove_own_temporary(temporary)


def replace_json(path: Path, payload: dict) -> None:
    """Atomically replace intentionally mutable metadata, using a unique staging file."""
    path = Path(path)
    temporary = _stage_json(path, payload)
    try:
        os.replace(temporary, path)
    finally:
        _remove_own_temporary(temporary)
