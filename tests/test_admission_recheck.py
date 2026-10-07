from threading import RLock
from types import SimpleNamespace
import pytest
from qwen_booster.admission_recheck import recheck, patch_admission_recheck


def setup(headrooms, resident=40):
    values = iter(headrooms)
    manager = SimpleNamespace(lock=RLock(), memory_max_bytes=100, memory_reserve_bytes=10,
        _prefill_reserve_bytes=20, _resident_bytes_locked=lambda: resident,
        _memory_headroom=lambda: next(values))
    now = [0.0]
    def sleep(delay): now[0] += delay
    reports = []
    return manager, now, sleep, reports


@pytest.mark.parametrize('headrooms,expected,attempts', [([59, 60], True, 2), ([59]*5, False, 5)])
def test_unchanged_threshold_and_bounded_recovery(headrooms, expected, attempts):
    m, now, sleep, reports = setup(headrooms)
    assert recheck(m, 30, 30, clock=lambda: now[0], sleep=sleep, report=reports.append) is expected
    assert len(reports) == attempts and now[0] <= .25
    assert all(r['required_headroom_bytes'] == 60 for r in reports)


def test_capacity_rejects_even_with_headroom():
    m, now, sleep, reports = setup([1000], resident=71)
    assert not recheck(m, 30, 30, clock=lambda: now[0], sleep=sleep, report=reports.append)
    assert len(reports) == 1 and not reports[0]['capacity_passed']


def test_oversleep_never_admits():
    m, now, _, reports = setup([])
    def sleep(delay): now[0] += 1
    assert not recheck(m, 30, 30, clock=lambda: now[0], sleep=sleep, report=reports.append)
    assert not reports


@pytest.mark.parametrize('eligible,evict,retain,original,expected_retries', [
    (True, True, 30, False, 1), (False, True, 30, False, 0),
    (True, False, 30, False, 0), (True, True, 0, False, 0),
    (True, True, 30, True, 0)])
def test_retry_scope(eligible, evict, retain, original, expected_retries):
    calls = []
    class Manager:
        def __init__(self): self.stats = SimpleNamespace(memory_evictions=0)
        def _make_room(self, allocation_bytes=0, *, retain_bytes=0):
            self.stats.memory_evictions += int(evict)
            return original
        def store_exact_cache(self, token_ids, prompt_cache, **kwargs):
            return self._make_room(30, retain_bytes=retain)
    def retry(*args, **kwargs): calls.append(args); return True
    patch_admission_recheck(Manager, eligible_cache=lambda c: eligible, retry=retry)
    m = Manager()
    assert m.store_exact_cache([1], []) is (original or bool(expected_retries))
    assert len(calls) == expected_retries
    # Scope must reset; an unrelated retain call must not retry.
    assert m._make_room(30, retain_bytes=30) is original
    assert len(calls) == expected_retries


@pytest.mark.parametrize('enabled,resident,eligible,retain,original,expected', [
    (True, 0, True, 30, False, 1),
    (False, 0, True, 30, False, 0),
    (True, 1, True, 30, False, 0),
    (True, 0, False, 30, False, 0),
    (True, 0, True, 0, False, 0),
    (True, 0, True, 30, True, 0),
])
def test_first_store_scope(enabled, resident, eligible, retain, original, expected):
    calls = []
    class Manager:
        def __init__(self):
            self.stats = SimpleNamespace(memory_evictions=0)
            self.lock = RLock()
        def _resident_bytes_locked(self): return resident
        def _make_room(self, allocation_bytes=0, *, retain_bytes=0): return original
        def store_exact_cache(self, token_ids, prompt_cache, **kwargs):
            return self._make_room(30, retain_bytes=retain)
    def retry(*args, **kwargs): calls.append(args); return True
    patch_admission_recheck(Manager, eligible_cache=lambda c: eligible,
                            retry=retry, retry_empty_store=enabled)
    m = Manager()
    assert m.store_exact_cache([1], []) is (original or bool(expected))
    assert len(calls) == expected
    assert m._make_room(30, retain_bytes=30) is original
    assert len(calls) == expected


def test_first_store_rechecks_current_capacity_not_original_empty_state():
    # Another retained entry appears during the bounded wait: never admit it
    # using the zero-resident observation taken before the original call.
    calls = []
    now = [0.0]
    class Manager:
        def __init__(self):
            self.stats = SimpleNamespace(memory_evictions=0)
            self.lock = RLock()
            self.resident = 0
            self.memory_max_bytes = 100
            self.memory_reserve_bytes = 10
            self._prefill_reserve_bytes = 20
        def _resident_bytes_locked(self): return self.resident
        def _memory_headroom(self): return 1000
        def _make_room(self, allocation_bytes=0, *, retain_bytes=0): return False
        def store_exact_cache(self, token_ids, prompt_cache, **kwargs):
            return self._make_room(30, retain_bytes=30)
    def retry(m, allocation, retain, **kwargs):
        def sleep(delay):
            now[0] += delay
            m.resident = 71
        return recheck(m, allocation, retain, clock=lambda: now[0], sleep=sleep, report=calls.append)
    patch_admission_recheck(Manager, eligible_cache=lambda c: True, retry=retry, retry_empty_store=True)
    assert not Manager().store_exact_cache([1], [])
    assert len(calls) == 1 and not calls[0]['capacity_passed']
