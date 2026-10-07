from threading import RLock
from types import SimpleNamespace

import pytest

from qwen_booster.admission_recheck import patch_admission_recheck
from qwen_booster.restore_admission import patch_restore_recheck, restore_recheck


@pytest.mark.parametrize('prefill,headrooms,expected', [
    (20, [39, 40], True),
    (50, [59, 60], True),
    (20, [39] * 5, False),
])
def test_restore_uses_max_reserve_not_store_sum(prefill, headrooms, expected):
    now, events, values = [0.0], [], iter(headrooms)
    manager = SimpleNamespace(
        lock=RLock(), memory_max_bytes=100, memory_reserve_bytes=10,
        _prefill_reserve_bytes=prefill, _resident_bytes_locked=lambda: 40,
        _memory_headroom=lambda: next(values))

    def sleep(delay):
        now[0] += delay

    assert restore_recheck(manager, 30, clock=lambda: now[0], sleep=sleep,
                           report=events.append) is expected
    assert all(e['required_headroom_bytes'] == 10 + max(prefill, 30) for e in events)
    assert len(events) <= 5 and now[0] <= .25


@pytest.mark.parametrize('change', ['capacity', 'reserve', 'oversleep'])
def test_changed_budget_or_expired_deadline_never_admits(change):
    now, events = [0.0], []
    manager = SimpleNamespace(
        lock=RLock(), memory_max_bytes=100, memory_reserve_bytes=10,
        _prefill_reserve_bytes=20, resident=40,
        _memory_headroom=lambda: 60)
    manager._resident_bytes_locked = lambda: manager.resident

    def sleep(delay):
        now[0] += 1 if change == 'oversleep' else delay
        if change == 'capacity':
            manager.resident = 101
        if change == 'reserve':
            manager._prefill_reserve_bytes = 80

    assert not restore_recheck(manager, 30, clock=lambda: now[0], sleep=sleep,
                               report=events.append)
    assert all(not e['admitted'] for e in events)
    if change == 'oversleep':
        assert not events


@pytest.mark.parametrize('evicted,original,allocation,retain,retries', [
    (True, False, 30, 0, 1),
    (False, False, 30, 0, 0),
    (True, True, 30, 0, 0),
    (True, False, 0, 0, 0),
    (True, False, 30, 30, 0),
])
def test_only_rejected_restore_after_actual_eviction_retries(
        evicted, original, allocation, retain, retries):
    calls = []

    class Manager:
        def __init__(self):
            self.stats = SimpleNamespace(memory_evictions=0)

        def _make_room(self, allocation_bytes=0, *, retain_bytes=0):
            self.stats.memory_evictions += int(evicted)
            return original

        def lookup_exact_cache(self):
            return self._make_room(allocation, retain_bytes=retain)

    def retry(*args, **kwargs):
        calls.append(args)
        return True

    patch_restore_recheck(Manager, retry=retry)
    patch_restore_recheck(Manager, retry=retry)
    manager = Manager()
    assert manager.lookup_exact_cache() is (original or bool(retries))
    assert len(calls) == retries
    assert manager._make_room(30) is original
    assert len(calls) == retries


@pytest.mark.parametrize('restore_first', [True, False])
def test_store_restore_scopes_compose_and_reset_after_exception(restore_first):
    calls = []

    class Manager:
        def __init__(self):
            self.stats = SimpleNamespace(memory_evictions=0)

        def _make_room(self, allocation_bytes=0, *, retain_bytes=0):
            self.stats.memory_evictions += 1
            return False

        def store_exact_cache(self, *args, **kwargs):
            return self._make_room(30, retain_bytes=30)

        def lookup_exact_cache(self, fail=False):
            if fail:
                raise RuntimeError('lookup failed')
            return self._make_room(60)

    def retry_store(*args, **kwargs):
        calls.append('store')
        return True

    def retry_restore(*args, **kwargs):
        calls.append('restore')
        return True

    installers = [
        lambda: patch_restore_recheck(Manager, retry=retry_restore),
        lambda: patch_admission_recheck(Manager, eligible_cache=lambda c: True,
                                       retry=retry_store),
    ]
    for install in installers if restore_first else reversed(installers):
        install()
    manager = Manager()
    assert manager.store_exact_cache([1], []) and manager.lookup_exact_cache()
    assert calls == ['store', 'restore']
    with pytest.raises(RuntimeError):
        manager.lookup_exact_cache(fail=True)
    assert not manager._make_room(60)
    assert not manager._make_room(30, retain_bytes=30)
    assert calls == ['store', 'restore']
