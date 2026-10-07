import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from qwen_booster.runtime import HybridEngine


class FakeCache(KVCache):
    def __init__(self):
        self.offset = 0

    @property
    def nbytes(self):
        return 10

    @property
    def state(self):
        return (mx.array(self.offset, dtype=mx.uint32),)


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, cache):
        self.calls.append(int(tokens.shape[1]))
        for entry in cache:
            entry.offset += int(tokens.shape[1])


class EngineHarness(HybridEngine):
    def _fresh_cache(self):
        return [FakeCache()]


def save(path, cache):
    Path(path).write_text(json.dumps([c.offset for c in cache]))


def load(path):
    cache = FakeCache()
    cache.offset = json.loads(Path(path).read_text())[0]
    return [cache]


def test_prepare_checkpoints_a_boundary_and_replays_one_block():
    model = FakeModel()
    engine = EngineHarness(model, prefill_step_size=3, cache_bytes=1000, block_size=4)
    engine.prepare(list(range(9)))
    assert engine.last_cached_tokens == 0
    assert engine.prefixes.stats()["entries"] == 1
    model.calls.clear()
    engine.prepare(list(range(10)))
    assert engine.last_cached_tokens == 8
    assert model.calls == [1]
    model.calls.clear()
    engine.prepare(list(range(8)) + [104, 105])
    assert engine.last_cached_tokens == 8 and model.calls == [1]
    model.calls.clear()
    engine.prepare(list(range(10)), reuse=False)
    assert engine.last_cached_tokens == 0
    assert model.calls == [3, 3, 3]


def test_newer_boundary_is_added_and_matched():
    model = FakeModel()
    engine = EngineHarness(model, prefill_step_size=3, cache_bytes=1000, block_size=4)
    engine.prepare(list(range(9)))
    engine.prepare(list(range(13)))
    assert engine.prefixes.stats()["entries"] == 2
    model.calls.clear()
    engine.prepare(list(range(13)))
    assert engine.last_cached_tokens == 12 and model.calls == []


def test_commit_only_accepts_block_aligned_lengths():
    engine = EngineHarness(FakeModel(), prefill_step_size=3, cache_bytes=1000, block_size=4)
    cache = engine.prepare(list(range(9)))
    assert engine.commit(list(range(8)), cache) is True
    assert engine.commit(list(range(9)), cache) is False
    assert engine.commit(list(range(8)), [object()]) is False


def test_disk_checkpoint_survives_a_fresh_engine(tmp_path):
    engine = EngineHarness(FakeModel(), prefill_step_size=3, cache_bytes=1000, block_size=4,
                           disk_root=tmp_path, disk_bytes=1_000,
                           disk_save=save, disk_load=load)
    engine.prepare(list(range(9)))
    assert engine.disk.stats()["entries"] == 1
    engine.prepare(list(range(10)))
    assert engine.disk.stats()["hits"] == 0
    fresh = EngineHarness(FakeModel(), prefill_step_size=3, cache_bytes=1000, block_size=4,
                       disk_root=tmp_path, disk_bytes=1_000,
                       disk_save=save, disk_load=load)
    cache = fresh.prepare(list(range(10)))
    assert cache[0].offset == 9
    assert fresh.last_cached_tokens == 8
    assert fresh.disk.stats()["hits"] == 1


def test_disk_requires_explicit_budget():
    try:
        EngineHarness(FakeModel(), block_size=4, disk_root="/tmp/not-used", disk_bytes=0)
    except ValueError as error:
        assert "disk checkpointing" in str(error)
    else:
        raise AssertionError("disk_root without disk_bytes must fail")
