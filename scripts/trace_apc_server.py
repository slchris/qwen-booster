"""Run the unchanged server with APC memory-admission diagnostics.

Uses the pinned MLX-VLM private API. Records byte counts only; admission decisions,
model state, context and cache budgets remain unchanged. Not a timing benchmark.
"""

import json
import logging
import time
import traceback

import mlx.core as mx
from mlx_vlm import apc


def install():
    original = apc.APCManager._make_room

    def snapshot(manager):
        return {'active_bytes': mx.get_active_memory(),
                'allocator_cache_bytes': mx.get_cache_memory(),
                'free_ram_bytes': apc._free_ram_bytes(),
                'working_set_bytes': manager._working_set_bytes,
                'resident_bytes': manager.resident_bytes(),
                'headroom_bytes': manager._memory_headroom(),
                'memory_max_bytes': manager.memory_max_bytes,
                'memory_reserve_bytes': manager.memory_reserve_bytes,
                'prefill_reserve_bytes': manager._prefill_reserve_bytes,
                'memory_evictions': manager.stats.memory_evictions}

    def traced(manager, allocation_bytes=0, *, retain_bytes=0):
        before = snapshot(manager)
        admitted = original(manager, allocation_bytes, retain_bytes=retain_bytes)
        after = snapshot(manager)
        if retain_bytes or not admitted or after['memory_evictions'] > before['memory_evictions']:
            required = before['memory_reserve_bytes'] + (
                before['prefill_reserve_bytes'] + allocation_bytes if retain_bytes
                else max(before['prefill_reserve_bytes'], allocation_bytes))
            logging.getLogger(__name__).warning('APC_MEMORY_DIAGNOSTIC %s', json.dumps({
                'unix_time': time.time(), 'allocation_bytes': allocation_bytes,
                'retain_bytes': retain_bytes, 'required_headroom_bytes': required,
                'admitted': admitted, 'before': before, 'after': after,
                'call_path': [f'{frame.name}:{frame.lineno}'
                              for frame in traceback.extract_stack(limit=6)[:-1]]}))
        return admitted

    apc.APCManager._make_room = traced


if __name__ == '__main__':
    install()
    from qwen_booster.vlm_server import main
    main()
