"""Opt-in exact-cache restore admission recheck after actual idle eviction."""
import functools
import logging
import time
from contextvars import ContextVar


def restore_recheck(manager, allocation_bytes, *, clock=time.monotonic,
                    sleep=time.sleep, report=lambda event: None):
    start=clock()
    for attempt in range(1,6):
        elapsed=clock()-start
        if elapsed>=0.25:return False
        sleep(min(0.05,0.25-elapsed))
        if clock()-start>0.25:return False
        with manager.lock:
            resident=manager._resident_bytes_locked()
            required=manager.memory_reserve_bytes+max(manager._prefill_reserve_bytes,allocation_bytes)
            headroom=manager._memory_headroom()
            elapsed=clock()-start
            capacity=resident<=manager.memory_max_bytes
            admitted=capacity and headroom>=required and elapsed<=0.25
        report(dict(attempt=attempt,elapsed_seconds=elapsed,resident_bytes=resident,
            allocation_bytes=allocation_bytes,required_headroom_bytes=required,
            headroom_bytes=headroom,capacity_passed=capacity,admitted=admitted))
        if admitted:return True
        if not capacity:return False
    return False


def patch_restore_recheck(manager_class, *, retry=restore_recheck):
    if getattr(manager_class,'_qwen_restore_recheck',False):return
    scope=ContextVar('qwen_exact_lookup_recheck',default=None)
    lookup,make_room=manager_class.lookup_exact_cache,manager_class._make_room
    logger=logging.getLogger(__name__)
    @functools.wraps(lookup)
    def looked_up(self,*args,**kwargs):
        token=scope.set(self)
        try:return lookup(self,*args,**kwargs)
        finally:scope.reset(token)
    @functools.wraps(make_room)
    def admitted(self,allocation_bytes=0,*,retain_bytes=0):
        before=self.stats.memory_evictions
        result=make_room(self,allocation_bytes,retain_bytes=retain_bytes)
        evicted=self.stats.memory_evictions-before
        if result or retain_bytes!=0 or allocation_bytes<=0 or scope.get() is not self or evicted<=0:
            return result
        logger.info('APC_RESTORE_RECHECK_REJECTED allocation=%d evictions=%d',allocation_bytes,evicted)
        result=retry(self,allocation_bytes,report=lambda e:logger.info('APC_RESTORE_RECHECK_SAMPLE %s',e))
        logger.info('APC_RESTORE_RECHECK_RESULT admitted=%s',result)
        return result
    manager_class.lookup_exact_cache=looked_up
    manager_class._make_room=admitted
    manager_class._qwen_restore_recheck=True


def install_flash_restore_recheck():
    from mlx_vlm.apc import APCManager
    patch_restore_recheck(APCManager)
    logging.getLogger(__name__).warning("APC_RESTORE_RECHECK_FLASH_INSTALLED attempts=5 max_wait_ms=250")
