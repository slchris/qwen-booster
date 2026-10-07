import json
from pathlib import Path

import pytest

from qwen_booster.apfs import CheckpointDiskStore, clone_file
from qwen_booster.hybrid_cache import final_block_hash


class FakeCache:
    def __init__(self, value=1, nbytes=8):
        self.value = value
        self.nbytes = nbytes
        self.state = {"recurrent": [value], "kv": [value + 1]}


def save(path, cache):
    Path(path).write_text(json.dumps([c.state for c in cache]))


def load(path):
    caches = []
    for state in json.loads(Path(path).read_text()):
        cache = FakeCache()
        cache.state = state
        cache.nbytes = 8
        caches.append(cache)
    return caches


def make_store(tmp_path, **kwargs):
    kwargs.setdefault("max_bytes", 1_000)
    return CheckpointDiskStore(tmp_path, save=save, load=load, **kwargs)


def put_tokens(store, tokens, value=1, block_size=4):
    store.put(final_block_hash(tokens, block_size), len(tokens), [FakeCache(value=value)])


def test_clone_file_copies_and_diverges(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "clone.bin"
    assert clone_file(source, destination) in ("clonefile", "copy")
    destination.write_bytes(b"changed")
    assert source.read_bytes() == b"payload"
    with pytest.raises(FileExistsError):
        clone_file(source, destination)


def test_snapshot_clones_files_and_refuses_pending_writes(tmp_path):
    store = make_store(tmp_path / "store")
    put_tokens(store, list(range(4)))
    store.snapshot(tmp_path / "backup")
    assert (tmp_path / "backup" / "index.json").is_file()
    assert len(list((tmp_path / "backup").glob("*.safetensors"))) == 1
    (store.root / ".tmp-pending-1").write_bytes(b"")
    with pytest.raises(ValueError):
        store.snapshot(tmp_path / "second-backup")


def test_put_get_roundtrip_and_index_reload(tmp_path):
    store = make_store(tmp_path)
    put_tokens(store, list(range(4)), value=5)
    digest = final_block_hash(list(range(4)), 4)
    restored = store.get(digest)
    assert restored[0].state == {"recurrent": [5], "kv": [6]}
    assert store.stats()["hits"] == 1
    reloaded = CheckpointDiskStore(tmp_path, max_bytes=1_000, save=save, load=load)
    assert reloaded.get(digest)[0].state == {"recurrent": [5], "kv": [6]}


def test_duplicate_put_keeps_existing_file(tmp_path):
    store = make_store(tmp_path)
    put_tokens(store, list(range(4)))
    before = {path.name: path.stat().st_mtime_ns for path in store.root.glob("*.safetensors")}
    put_tokens(store, list(range(4)), value=9)
    after = {path.name: path.stat().st_mtime_ns for path in store.root.glob("*.safetensors")}
    assert before == after and store.stats()["duplicate_stores"] == 1


def test_eviction_removes_oldest_file_and_metadata(tmp_path):
    store = make_store(tmp_path, max_bytes=16)
    put_tokens(store, list(range(4)), value=1)
    put_tokens(store, list(range(8)), value=2)
    put_tokens(store, list(range(12)), value=3)
    assert store.stats()["entries"] == 2
    assert store.stats()["evictions"] == 1
    assert not store._entry_path(f"{final_block_hash(list(range(4)), 4):016x}").is_file()
    assert store.get(final_block_hash(list(range(8)), 4))[0].state["recurrent"] == [2]


def test_oversize_checkpoint_is_skipped(tmp_path):
    store = make_store(tmp_path, max_bytes=4)
    assert store.put(1, 4, [FakeCache(nbytes=8)]) is False
    assert store.stats()["skipped_oversize"] == 1


def test_find_matches_across_missing_boundaries_and_skips_final_block(tmp_path):
    store = make_store(tmp_path)
    put_tokens(store, list(range(8)), value=7)
    count, cache = store.find(list(range(12)), 4)
    assert count == 8 and cache[0].state["recurrent"] == [7]
    put_tokens(store, list(range(12)), value=8)
    assert store.find(list(range(12)), 4)[0] == 8
    assert store.find(list(range(16)), 4)[0] == 12


def test_clear_removes_files(tmp_path):
    store = make_store(tmp_path)
    put_tokens(store, list(range(4)))
    store.clear()
    assert store.stats()["entries"] == 0
    assert not list(store.root.glob("*.safetensors"))
    assert json.loads((store.root / "index.json").read_text())["entries"] == {}
