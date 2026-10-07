"""Opt-in live-cache credit at the pinned backend's post-restore prepares.

Supports opt-in singleton Flash-Next and 27B paths. Cold admission and the
manager's eviction/headroom checks remain unchanged. No tensors are modified.
"""

import functools
import inspect
from contextvars import ContextVar


def patch_warm_reserve(batch_class, coordinator_class, *, accepts_model, cache_bytes, report=None, allow_mtp=False):
    if getattr(coordinator_class, '_qwen_live_reserve_installed', False):
        return
    context = ContextVar('qwen_live_reserve', default=None)
    init = batch_class.__init__
    merge = coordinator_class.merge_rows
    prepare = coordinator_class.prepare_prefill
    init_sig, merge_sig, prepare_sig = map(inspect.signature, (init, merge, prepare))

    @functools.wraps(init)
    def initialize(self, *args, **kwargs):
        bound = init_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        coordinator = v.get('apc_coordinator')
        eligible = (coordinator is not None and v.get('warm_cache') is not None
                    and len(v['input_ids']) == 1 and not any(v.get('right_pad_per_row') or [])
                    and ((v.get('draft_model') is None and v.get('draft_kind') is None)
                         or (allow_mtp and v.get('draft_model') is not None and v.get('draft_kind') == 'mtp'))
                    and v.get('kv_bits') is None and accepts_model(v['model']))
        token = context.set((coordinator, lambda: getattr(self, 'prompt_cache', None), 'batch')
                            if eligible else None)
        try:
            return init(self, *args, **kwargs)
        finally:
            context.reset(token)

    @functools.wraps(merge)
    def merge_rows(self, *args, **kwargs):
        bound = merge_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        picks, prefixes = v['picks'], v['prefix_lens']
        eligible = (self.is_checkpoint and accepts_model(self.model)
                    and len(picks) == len(prefixes) == 1 and prefixes[0] > 0
                    and picks[0] is not None and v.get('kv_quant_config') is None)
        token = context.set((self, lambda: picks[0].get('warm_cache'), 'merge') if eligible else None)
        try:
            return merge(self, *args, **kwargs)
        finally:
            context.reset(token)

    @functools.wraps(prepare)
    def prepare_prefill(self, *args, **kwargs):
        active = context.get()
        if active is None or active[0] is not self or not self.enabled:
            return prepare(self, *args, **kwargs)
        bound = prepare_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        lengths = v['prompt_lengths']
        lengths = [lengths] if isinstance(lengths, int) else lengths
        prefixes = v.get('prefix_lengths')
        if len(lengths) != 1 or prefixes is None or len(prefixes) != 1 or not 0 < prefixes[0] <= lengths[0]:
            return prepare(self, *args, **kwargs)
        live_bytes = cache_bytes(active[1]())
        if live_bytes is None or live_bytes <= 0:
            return prepare(self, *args, **kwargs)
        plan = self.manager.memory_plan
        full = plan.prepare(lengths, chunk_size=v.get('prefill_step_size'), prefix_lengths=prefixes)
        # Credit at most one row footprint, retaining the other copy/growth
        # reserve even if restored capacity is larger than current demand.
        credit = min(live_bytes, full // 2)
        reserve = plan.reserve_bytes(credit)
        self.manager.prepare_prefill(reserve)
        if report is not None:
            report(active[2], live_bytes, credit, full, reserve)

    batch_class.__init__ = initialize
    coordinator_class.merge_rows = merge_rows
    coordinator_class.prepare_prefill = prepare_prefill
    coordinator_class._qwen_live_reserve_installed = True


def install_warm_reserve(*, allow_mtp=False):
    import logging
    import mlx.core as mx
    from mlx_vlm.apc import _cache_nbytes, _collect_mx_arrays
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, QSAKVCache, BatchQSAKVCache

    def measured(entries):
        if not entries or len(entries) != 48:
            return None
        allowed = (ArraysCache, QSAKVCache, BatchQSAKVCache)
        if any(type(c) not in allowed for c in entries):
            return None
        if sum(type(c) is ArraysCache for c in entries) != 36:
            return None
        arrays = []
        for entry in entries:
            if type(entry) is ArraysCache:
                if any(a is not None and (a.ndim == 0 or a.shape[0] != 1) for a in entry.cache):
                    return None
            else:
                kv = entry.kv_cache if type(entry) is BatchQSAKVCache else entry
                if kv.keys is None or kv.keys.shape[0] != 1:
                    return None
            _collect_mx_arrays(entry.state, arrays)
        if not arrays:
            return None
        mx.eval(arrays)
        return _cache_nbytes(entries)

    def report(phase, live, credit, before, after):
        marker = 'WARM_FLASH_RESERVE' if allow_mtp else 'Warm reserve credit'
        logging.getLogger(__name__).info(
            '%s phase=%s logical_live=%d credit=%d before=%d after=%d',
            marker, phase, live, credit, before, after)

    patch_warm_reserve(PromptProcessingBatch, APCCoordinator,
                       accepts_model=lambda m: type(getattr(m, 'language_model', m)) is LanguageModel,
                       cache_bytes=measured, report=report, allow_mtp=allow_mtp)


def install_27b_warm_reserve():
    import logging
    import mlx.core as mx
    from mlx_vlm.apc import _cache_nbytes, _collect_mx_arrays
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache, KVCache, BatchKVCache
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    def measured(entries):
        if not entries or len(entries) != 64:
            return None
        arrays = []
        for i, entry in enumerate(entries):
            if i % 4 != 3:
                if type(entry) is not ArraysCache or any(
                    a is None or a.ndim == 0 or a.shape[0] != 1 for a in entry.cache
                ):
                    return None
            else:
                if type(entry) not in (KVCache, BatchKVCache):
                    return None
                if any(a is None or a.ndim != 4 or a.shape[0] != 1
                       for a in (entry.keys, entry.values)):
                    return None
                if type(entry) is BatchKVCache and (
                    entry.left_padding.size != 1 or int(entry.left_padding[0].item()) != 0
                ):
                    return None
            _collect_mx_arrays(entry.state, arrays)
        mx.eval(arrays)
        return _cache_nbytes(entries)

    def report(phase, live, credit, before, after):
        logging.getLogger(__name__).info(
            'WARM_27B_RESERVE phase=%s live=%d credit=%d before=%d after=%d',
            phase, live, credit, before, after)

    patch_warm_reserve(PromptProcessingBatch, APCCoordinator,
                       accepts_model=lambda m: type(getattr(m, 'language_model', m)) is LanguageModel,
                       cache_bytes=measured, report=report, allow_mtp=True)
