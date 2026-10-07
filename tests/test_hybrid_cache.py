from qwen_booster.hybrid_cache import (HybridPrefixCache, block_chain,
                                        block_hash, final_block_hash)
import pytest


class FakeCache:
    def __init__(self, nbytes=10):
        self.nbytes = nbytes
        self.state = {"recurrent": [1, 2], "kv": [3, 4]}


def test_block_hashes_chain_over_parent_and_position():
    chain = list(block_chain(list(range(8)), 4))
    assert [end for end, _ in chain] == [4, 8]
    assert chain[0][1] == block_hash(0, [0, 1, 2, 3])
    assert chain[1][1] == block_hash(chain[0][1], [4, 5, 6, 7])
    assert chain[1][1] != block_hash(0, [4, 5, 6, 7])


def test_store_requires_block_aligned_prefix():
    cache = HybridPrefixCache(block_size=4)
    with pytest.raises(ValueError):
        cache.store([1, 2, 3], [FakeCache()])
    with pytest.raises(ValueError):
        cache.store([], [FakeCache()])
    assert cache.store([1, 2, 3, 4], [FakeCache()])


def test_find_matches_across_missing_intermediate_boundaries():
    store = HybridPrefixCache(block_size=4)
    store.store(list(range(8)), [FakeCache()])
    count, restored = store.find(list(range(4)) + [99])
    assert (count, restored) == (0, None)
    count, restored = store.find(list(range(12)))
    assert count == 8 and restored[0].state["kv"] == [3, 4]


def test_find_never_returns_the_whole_request():
    store = HybridPrefixCache(block_size=4)
    store.store(list(range(8)), [FakeCache()])
    count, restored = store.find(list(range(12)))
    assert count == 8 and count <= len(range(12)) - 1
    assert restored is not None
    assert store.find(list(range(8))) == (0, None)


def test_restored_snapshot_is_isolated_from_store_and_caller():
    store = HybridPrefixCache(block_size=4)
    original = FakeCache()
    store.store(list(range(4)), [original])
    original.state["kv"][0] = 9
    _, restored = store.find(list(range(5)))
    restored[0].state["kv"][0] = 9
    _, again = store.find(list(range(5)))
    assert again[0].state["kv"][0] == 3


def test_lru_eviction_and_replace_by_same_key():
    store = HybridPrefixCache(max_bytes=20, block_size=4)
    assert store.store(list(range(4)), [FakeCache()]) is True
    assert store.store(list(range(4)), [FakeCache()]) is True
    assert store.stats()["entries"] == 1
    store.store(list(range(4, 8)), [FakeCache()])
    assert store.stats()["evictions"] == 0
    store.store(list(range(8, 12)), [FakeCache()])
    assert store.stats()["entries"] == 2
    assert store.stats()["bytes"] == 20
    assert store.find(list(range(4)))[0] == 0


def test_oversize_and_disabled_entries_are_skipped():
    assert HybridPrefixCache(max_bytes=9, block_size=4).store(list(range(4)), [FakeCache()]) is False
    assert HybridPrefixCache(max_entries=0, block_size=4).store(list(range(4)), [FakeCache()]) is False


def test_statistics_and_clear():
    store = HybridPrefixCache(block_size=4)
    store.store(list(range(4)), [FakeCache()])
    store.find(list(range(5)))
    store.find([7, 7, 7, 7, 7])
    stats = store.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["block_size"] == 4
    store.clear()
    assert store.nbytes == 0 and store.stats()["entries"] == 0
