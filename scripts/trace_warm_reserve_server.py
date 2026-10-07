"""Observe post-restore APC reserve accounting without changing admission.

Logical cache byte counts do not prove physical ownership or safe memory credit.
Run separately from timing benchmarks; this adds diagnostic overhead.
"""

import functools
import inspect
import json
import logging
from contextvars import ContextVar


def install_observer(batch_class, coordinator_class, observe):
    current = ContextVar('qwen_warm_batch', default=None)
    initialize = batch_class.__init__
    signature = inspect.signature(initialize)
    prepare = coordinator_class.prepare_prefill

    @functools.wraps(initialize)
    def wrapped_init(self, *args, **kwargs):
        values = signature.bind(self, *args, **kwargs)
        values.apply_defaults()
        token = current.set(self if values.arguments.get('warm_cache') is not None else None)
        try:
            return initialize(self, *args, **kwargs)
        finally:
            current.reset(token)

    @functools.wraps(prepare)
    def wrapped_prepare(self, *args, **kwargs):
        batch = current.get()
        eligible = (batch is not None and getattr(batch, '_apc_coordinator', None) is self
                    and getattr(batch, 'prompt_cache', None) is not None)
        if eligible:
            observe('before', self, batch)
        result = prepare(self, *args, **kwargs)
        if eligible:
            observe('after', self, batch)
        return result

    batch_class.__init__ = wrapped_init
    coordinator_class.prepare_prefill = wrapped_prepare


def main():
    import mlx.core as mx
    from mlx_vlm import apc
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from trace_apc_server import install

    install()

    def observe(phase, coordinator, batch):
        manager = coordinator.manager
        if manager is None:
            return
        cache_bytes = apc._cache_nbytes(batch.prompt_cache)
        plan = manager.memory_plan
        logging.getLogger(__name__).warning('APC_WARM_RESERVE_DIAGNOSTIC %s', json.dumps({
            'phase': phase, 'cache_types': [type(c).__name__ for c in batch.prompt_cache],
            'logical_live_cache_bytes': cache_bytes,
            'plan_lengths': plan.lengths, 'plan_prefix_lengths': plan.prefix_lengths,
            'reserve_without_credit_bytes': plan.reserve_bytes(),
            'hypothetical_reserve_with_logical_credit_bytes': plan.reserve_bytes(cache_bytes),
            'applied_reserve_bytes': manager._prefill_reserve_bytes,
            'active_bytes': mx.get_active_memory(),
            'resident_snapshot_bytes': manager.resident_bytes(),
            'headroom_bytes': manager._memory_headroom(),
            'extra_reserve_bytes': manager.memory_reserve_bytes,
            'note': 'Observed only; no credit applied. Logical bytes are not physical ownership proof.'}))

    install_observer(PromptProcessingBatch, APCCoordinator, observe)
    from qwen_booster.vlm_server import main as serve
    serve()


if __name__ == '__main__':
    main()
