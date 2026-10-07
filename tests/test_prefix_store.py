from qwen_booster.runtime import PrefixStore


class FakeCache:
    nbytes = 10

    def __init__(self):
        self.state = {"recurrent": [1, 2], "kv": [3, 4]}


def test_snapshot_restores_all_state_and_is_isolated():
    store = PrefixStore()
    original = FakeCache()
    store.put([1, 2], [original])
    original.state["recurrent"][0] = 9
    count, restored = store.get([1, 2, 3])
    assert count == 2
    assert restored[0].state["recurrent"][0] == 1
    restored[0].state["kv"][0] = 9
    assert store.get([1, 2, 4])[1][0].state["kv"][0] == 3


def test_no_recurrent_state_truncation_on_partial_match():
    store = PrefixStore()
    store.put([1, 2, 3], [FakeCache()])
    assert store.get([1, 2, 9]) == (0, None)
    assert store.get([1, 2, 3]) == (0, None)


def test_longest_prefix_and_lru_budget():
    store = PrefixStore(max_bytes=20)
    store.put([1], [FakeCache()])
    store.put([1, 2], [FakeCache()])
    assert store.get([1, 2, 3])[0] == 2
    store.put([7], [FakeCache()])
    assert store.get([1, 9]) == (0, None)
    assert store.nbytes == 20
    assert not PrefixStore(max_bytes=9).put([1], [FakeCache()])


def test_statistics_and_clear_release_entries():
    store = PrefixStore(max_bytes=10)
    store.put([1], [FakeCache()])
    store.get([1, 2])
    store.put([2], [FakeCache()])
    store.get([1, 2])
    assert store.stats()["hits"] == 1
    assert store.stats()["misses"] == 1
    assert store.stats()["evictions"] == 1
    store.clear()
    assert store.nbytes == 0 and store.stats()["entries"] == 0
