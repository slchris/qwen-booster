"""APFS-aware persistence for block-aligned hybrid checkpoints.

APFS is copy-on-write: ``clonefile`` (``cp -c``) copies metadata instead of
bytes, and later writes to either side diverge cheaply. This store keeps one
safetensors checkpoint per block-aligned prefix, writes new files through a
temporary path plus an atomic rename, and replaces a small JSON index
atomically. ``snapshot`` clones the whole store for cheap backups.

Serialization defaults to mlx-lm's prompt-cache save/load helpers; callers may
inject their own pair for CPU tests or other formats.
"""

import ctypes
import json
import os
import shutil
import threading
import time
from pathlib import Path

from .hybrid_cache import block_chain


def clone_file(src, dst, *, follow_symlinks=True):
    """Clone a file on APFS; fall back to a byte copy elsewhere."""
    if os.path.exists(dst):
        raise FileExistsError(dst)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0:
            return "clonefile"
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    except (AttributeError, OSError):
        shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        return "copy"


def snapshot_tree(src, dst):
    """Clone a store directory file by file, preserving copy-on-write."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True)
    for path in sorted(src.iterdir()):
        if path.name.startswith(".tmp-"):
            continue
        clone_file(path, dst / path.name)


def _fsync_path(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _default_save(path, cache):
    from mlx_lm.models.cache import save_prompt_cache
    save_prompt_cache(str(path), cache)


def _default_load(path):
    from mlx_lm.models.cache import load_prompt_cache
    return load_prompt_cache(str(path))


class CheckpointDiskStore:
    """Content-keyed checkpoint files with byte-budgeted LRU eviction."""

    def __init__(self, root, max_bytes, max_entries=64, save=None, load=None,
                 clock=time.time):
        if max_bytes <= 0 or max_entries <= 0:
            raise ValueError("disk budgets must be positive")
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._save = save or _default_save
        self._load = load or _default_load
        self._clock = clock
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.skipped = self.exists = 0
        self.entries = self._read_index()

    def _index_path(self):
        return self.root / "index.json"

    def _entry_path(self, key):
        return self.root / f"{key}.safetensors"

    def _read_index(self):
        path = self._index_path()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        if data.get("version") != 1 or not isinstance(data.get("entries"), dict):
            raise ValueError("unsupported disk cache index")
        return data["entries"]

    def _write_index(self):
        path = self._index_path()
        temporary = self.root / f".tmp-index-{os.getpid()}"
        temporary.write_text(json.dumps({"version": 1, "entries": self.entries}))
        _fsync_path(temporary)
        os.replace(temporary, path)

    @property
    def nbytes(self):
        return sum(entry["bytes"] for entry in self.entries.values())

    def stats(self):
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "budget_bytes": self.max_bytes, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions,
                "skipped_oversize": self.skipped, "duplicate_stores": self.exists}

    def put(self, digest, tokens_len, cache, nbytes=None):
        """Persist one checkpoint; duplicate keys are kept as-is."""
        size = sum(c.nbytes for c in cache) if nbytes is None else nbytes
        key = f"{digest:016x}"
        with self._lock:
            if key in self.entries:
                self.exists += 1
                self.entries[key]["last_used"] = self._clock()
                self._write_index()
                return True
            if size > self.max_bytes:
                self.skipped += 1
                return False
            while self.entries and (self.nbytes + size > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                oldest = min(self.entries, key=lambda k: self.entries[k]["last_used"])
                self._remove(oldest)
            temporary = self.root / f".tmp-{key}-{os.getpid()}"
            try:
                self._save(temporary, cache)
                _fsync_path(temporary)
                os.replace(temporary, self._entry_path(key))
            finally:
                temporary.unlink(missing_ok=True)
            self.entries[key] = {"tokens": tokens_len, "bytes": size,
                                 "last_used": self._clock()}
            self._write_index()
            return True

    def _remove(self, key):
        self._entry_path(key).unlink(missing_ok=True)
        self.entries.pop(key, None)
        self.evictions += 1

    def get(self, digest):
        key = f"{digest:016x}"
        with self._lock:
            entry = self.entries.get(key)
            if entry is None or not self._entry_path(key).is_file():
                self.misses += 1
                return None
            cache = self._load(self._entry_path(key))
            if cache is None:
                self.misses += 1
                return None
            entry["last_used"] = self._clock()
            self._write_index()
            self.hits += 1
            return cache

    def find(self, tokens, block_size):
        """Longest persisted block-aligned prefix shorter than tokens.

        Chained hashes validate each entry independently, so missing
        intermediate boundaries do not stop an otherwise matching walk.
        """
        best = None
        for end, digest in block_chain(tokens, block_size):
            if end >= len(tokens):
                break
            entry = self.entries.get(f"{digest:016x}")
            if entry is not None and entry["tokens"] == end:
                best = digest
        if best is None:
            self.misses += 1
            return 0, None
        cache = self.get(best)
        if cache is None:
            return 0, None
        return self.entries[f"{best:016x}"]["tokens"], cache

    def snapshot(self, destination):
        """Copy-on-write backup of every checkpoint file and the index."""
        if any(path.name.startswith(".tmp-") for path in self.root.iterdir()):
            raise ValueError("refusing to snapshot a store with pending writes")
        snapshot_tree(self.root, destination)

    def clear(self):
        with self._lock:
            for key in list(self.entries):
                self._remove(key)
            self.evictions = 0
            self._write_index()
