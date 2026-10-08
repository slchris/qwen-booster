"""Measure checkpoint extraction/clone allocation without changing admission.

Diagnostic runs only: allocator deltas include aliasing and lazy evaluation and
are not proof of exclusive ownership. Original calls and exceptions propagate.
"""

import functools
import json
import logging
from contextvars import ContextVar


def install_snapshot_observer(coordinator_class, apc, observe):
    current = ContextVar('snapshot_coordinator', default=None)
    original = coordinator_class.store_checkpoint

    @functools.wraps(original)
    def checkpoint(self, *args, **kwargs):
        token = current.set(self)
        try:
            observe('checkpoint.before', self)
            return original(self, *args, **kwargs)
        finally:
            try:
                observe('checkpoint.after', self)
            finally:
                current.reset(token)

    def wrap(name, label):
        function = getattr(apc, name)

        @functools.wraps(function)
        def traced(*args, **kwargs):
            coordinator = current.get()
            if coordinator is None:
                return function(*args, **kwargs)
            observe(label + '.before', coordinator)
            try:
                return function(*args, **kwargs)
            finally:
                observe(label + '.after', coordinator)

        setattr(apc, name, traced)

    wrap('extract_prompt_cache_from_batch', 'extract')
    wrap('_clone_prompt_cache_for_apc', 'clone')
    coordinator_class.store_checkpoint = checkpoint


def main():
    import mlx.core as mx
    from mlx_vlm import apc
    from mlx_vlm.apc_coordinator import APCCoordinator

    def observe(phase, coordinator):
        manager = coordinator.manager
        if manager is None:
            return
        logging.getLogger(__name__).warning('APC_SNAPSHOT_DIAGNOSTIC %s', json.dumps({
            'phase': phase,
            'active_bytes': mx.get_active_memory(),
            'allocator_cache_bytes': mx.get_cache_memory(),
            'free_ram_bytes': apc._free_ram_bytes(),
            'resident_bytes': manager.resident_bytes(),
            'headroom_bytes': manager._memory_headroom(),
            'prefill_reserve_bytes': manager._prefill_reserve_bytes,
        }))

    install_snapshot_observer(APCCoordinator, apc, observe)
    from trace_warm_reserve_server import main as serve
    serve()


if __name__ == '__main__':
    main()
