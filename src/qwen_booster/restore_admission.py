"""Opt-in exact-cache restore admission recheck after a lookup-time rejection.

A cold request can finish and release its MTP batch moments before the next
request looks up the same prefix. The released memory is not yet visible as
headroom, so the restored-state admission loses the race and the request
falls back to a full recompute. The store path already retries after an
actual eviction; this module also retries the restore path when no eviction
happened, first nudging MLX to reclaim freed buffers (synchronize, clear
allocator cache, collect Python cycles) and then polling the unchanged
admission guards for at most two seconds. Capacity failures still fail fast.
"""
import functools
import gc
import logging
import time
from contextvars import ContextVar


def _nudge_memory():
    try:
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def restore_recheck(manager, allocation_bytes, *, clock=time.monotonic,
                    sleep=time.sleep, report=lambda event: None, attempts=5,
                    budget_seconds=0.25, nudge=False):
    start = clock()
    for attempt in range(1, attempts + 1):
        if nudge and attempt == 1:
            _nudge_memory()
        elapsed = clock() - start
        if elapsed >= budget_seconds:
            return False
        sleep(min(0.05, max(0, budget_seconds - elapsed)))
        elapsed = clock() - start
        if elapsed > budget_seconds:
            return False
        with manager.lock:
            resident = manager._resident_bytes_locked()
            required = manager.memory_reserve_bytes + max(
                manager._prefill_reserve_bytes, allocation_bytes)
            headroom = manager._memory_headroom()
            capacity = resident <= manager.memory_max_bytes
            elapsed = clock() - start
            admitted = capacity and headroom >= required and elapsed <= budget_seconds
        report(dict(attempt=attempt, elapsed_seconds=elapsed, resident_bytes=resident,
                    allocation_bytes=allocation_bytes,
                    required_headroom_bytes=required, headroom_bytes=headroom,
                    capacity_passed=capacity, admitted=admitted))
        if admitted:
            return True
        if not capacity:
            return False
    return False


def restore_recheck_settled(manager, allocation_bytes, *, clock=time.monotonic,
                            sleep=time.sleep, report=lambda event: None):
    """Bounded two-second settle loop used by the Flash MTP restore recheck."""
    return restore_recheck(manager, allocation_bytes, clock=clock, sleep=sleep,
                           report=report, attempts=25, budget_seconds=2.0, nudge=True)


def patch_restore_recheck(manager_class, *, retry=restore_recheck_settled):
    if getattr(manager_class, '_qwen_restore_recheck', False):
        return
    scope = ContextVar('qwen_exact_lookup_recheck', default=None)
    lookup, make_room = manager_class.lookup_exact_cache, manager_class._make_room
    logger = logging.getLogger(__name__)

    @functools.wraps(lookup)
    def looked_up(self, *args, **kwargs):
        token = scope.set(self)
        try:
            return lookup(self, *args, **kwargs)
        finally:
            scope.reset(token)

    @functools.wraps(make_room)
    def admitted(self, allocation_bytes=0, *, retain_bytes=0):
        before = self.stats.memory_evictions
        result = make_room(self, allocation_bytes, retain_bytes=retain_bytes)
        if result or retain_bytes != 0 or allocation_bytes <= 0 or scope.get() is not self:
            return result
        evicted = self.stats.memory_evictions - before
        logger.info('APC_RESTORE_RECHECK_REJECTED allocation=%d evictions=%d',
                    allocation_bytes, evicted)
        result = retry(self, allocation_bytes,
                       report=lambda e: logger.info('APC_RESTORE_RECHECK_SAMPLE %s', e))
        logger.info('APC_RESTORE_RECHECK_RESULT admitted=%s', result)
        return result

    manager_class.lookup_exact_cache = looked_up
    manager_class._make_room = admitted
    manager_class._qwen_restore_recheck = True


def install_flash_restore_recheck():
    from mlx_vlm.apc import APCManager
    patch_restore_recheck(APCManager)
    logging.getLogger(__name__).warning(
        "APC_RESTORE_RECHECK_FLASH_INSTALLED attempts=25 max_wait_ms=2000")
