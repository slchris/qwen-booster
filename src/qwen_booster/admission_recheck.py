"""Opt-in bounded exact-snapshot admission recheck after actual idle eviction."""
import functools
import logging
import time
from contextvars import ContextVar


def recheck(manager, allocation_bytes, retain_bytes, *, clock=time.monotonic,
            sleep=time.sleep, report=lambda event: None):
    """No allocations or evictions: resample unchanged guards for at most 250 ms."""
    start = clock()
    for attempt in range(1, 6):
        if clock() - start >= 0.25:
            return False
        sleep(min(0.05, max(0, 0.25 - (clock() - start))))
        # Scheduler delays may exceed the deadline; never admit after it.
        elapsed = clock() - start
        if elapsed > 0.25:
            return False
        with manager.lock:
            resident = manager._resident_bytes_locked()
            required = manager.memory_reserve_bytes + manager._prefill_reserve_bytes + allocation_bytes
            headroom = manager._memory_headroom()
            capacity = resident + retain_bytes <= manager.memory_max_bytes
            elapsed = clock() - start
            admitted = capacity and headroom >= required and elapsed <= 0.25
        report({'attempt': attempt, 'elapsed_seconds': elapsed, 'resident_bytes': resident,
                'retain_bytes': retain_bytes, 'required_headroom_bytes': required,
                'headroom_bytes': headroom, 'capacity_passed': capacity, 'admitted': admitted})
        if admitted:
            return True
        if not capacity:
            return False
    return False


def patch_admission_recheck(manager_class, *, eligible_cache, retry=recheck, retry_empty_store=False):
    if getattr(manager_class, '_qwen_admission_recheck', False):
        return
    scope = ContextVar('qwen_exact_snapshot_recheck', default=None)
    store, make_room = manager_class.store_exact_cache, manager_class._make_room
    logger = logging.getLogger(__name__)

    @functools.wraps(store)
    def stored(self, token_ids, prompt_cache, **kwargs):
        token = scope.set(self if eligible_cache(prompt_cache) else None)
        try:
            return store(self, token_ids, prompt_cache, **kwargs)
        finally:
            scope.reset(token)

    @functools.wraps(make_room)
    def admitted(self, allocation_bytes=0, *, retain_bytes=0):
        before = self.stats.memory_evictions
        empty_store = False
        if retry_empty_store and retain_bytes > 0 and scope.get() is self:
            with self.lock:
                empty_store = self._resident_bytes_locked() == 0
        result = make_room(self, allocation_bytes, retain_bytes=retain_bytes)
        if result or retain_bytes <= 0 or scope.get() is not self:
            return result
        evicted = self.stats.memory_evictions - before
        if evicted <= 0 and not empty_store:
            return result
        logger.info('APC_RECHECK_REJECTED allocation=%d retain=%d evictions=%d',
                    allocation_bytes, retain_bytes, evicted)
        if empty_store and evicted <= 0:
            logger.info('APC_RECHECK_FIRST_STORE allocation=%d retain=%d', allocation_bytes, retain_bytes)
        result = retry(self, allocation_bytes, retain_bytes,
            report=lambda event: logger.info('APC_RECHECK_SAMPLE %s', event))
        logger.info('APC_RECHECK_RESULT admitted=%s', result)
        return result

    manager_class.store_exact_cache = stored
    manager_class._make_room = admitted
    manager_class._qwen_admission_recheck = True


def install_27b_admission_recheck(*, retry_empty_store=False):
    from mlx_vlm.apc import APCManager
    from mlx_vlm.models.cache import ArraysCache, KVCache

    def eligible(entries):
        if len(entries) != 64:
            return False
        for i, cache in enumerate(entries):
            if i % 4 != 3:
                if type(cache) is not ArraysCache or any(
                    a is None or a.ndim == 0 or a.shape[0] != 1 for a in cache.cache):
                    return False
            elif (type(cache) is not KVCache or cache.keys is None or cache.values is None
                  or cache.keys.ndim != 4 or cache.values.ndim != 4
                  or cache.keys.shape[0] != 1 or cache.values.shape[0] != 1):
                return False
        return True

    patch_admission_recheck(APCManager, eligible_cache=eligible, retry_empty_store=retry_empty_store)
    logging.getLogger(__name__).warning("APC_RECHECK_27B_INSTALLED attempts=5 max_wait_ms=250")
    if retry_empty_store:
        logging.getLogger(__name__).warning("APC_RECHECK_FIRST_STORE_INSTALLED attempts=5 max_wait_ms=250")


def install_flash_admission_recheck():
    from mlx_vlm.apc import APCManager
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    def eligible(entries):
        if len(entries) != 48:
            return False
        for i, cache in enumerate(entries):
            if i % 4 != 3:
                if type(cache) is not ArraysCache or any(
                    a is None or a.ndim == 0 or a.shape[0] != 1 for a in cache.cache
                ):
                    return False
            elif (type(cache) is not QSAKVCache or cache.keys is None or cache.values is None
                  or cache.keys.ndim != 4 or cache.values.ndim != 4
                  or cache.keys.shape[0] != 1 or cache.values.shape[0] != 1
                  or cache.index_keys is None or cache.index_position_ids is None):
                return False
        return True

    # Only an actual eviction followed by rejection can trigger a recheck.
    # The existing capacity/headroom guards and the 250 ms bound are unchanged.
    patch_admission_recheck(APCManager, eligible_cache=eligible)
    logging.getLogger(__name__).warning("APC_RECHECK_FLASH_INSTALLED attempts=5 max_wait_ms=250")
