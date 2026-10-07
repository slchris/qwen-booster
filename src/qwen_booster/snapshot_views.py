"""Opt-in borrowed singleton row views during synchronous checkpoint storage.

The original manager still admits and clones the snapshot before returning.
Borrowed rows must never be retained as snapshots or used outside that call.
"""

import functools
from contextvars import ContextVar


def borrow_27b_row(entries, idx, *, arrays_class, batch_kv_class, kv_class):
    """Borrow the known 64-layer singleton layout for synchronous cloning only."""
    if idx != 0 or len(entries) != 64:
        return None
    for i, c in enumerate(entries):
        if i % 4 != 3:
            if type(c) is not arrays_class or any(
                a is None or a.ndim == 0 or a.shape[0] != 1 for a in c.cache
            ):
                return None
        else:
            if type(c) is not batch_kv_class:
                return None
            if (c.keys is None or c.values is None
                    or c.keys.ndim != 4 or c.values.ndim != 4
                    or c.keys.shape[0] != 1 or c.values.shape[0] != 1
                    or c.left_padding.size != 1 or int(c.left_padding[0].item()) != 0
                    or not 0 < c._idx <= min(c.keys.shape[2], c.values.shape[2])):
                return None
    rows = []
    for c in entries:
        if type(c) is arrays_class:
            row = arrays_class(len(c.cache))
            row.cache = [a[0:1] for a in c.cache]
        else:
            row = kv_class()
            row.keys = c.keys[0:1, :, :c._idx]
            row.values = c.values[0:1, :, :c._idx]
            row.offset = c._idx
        rows.append(row)
    return rows


def borrow_flash_row(entries, idx, *, arrays_class, batch_qsa_class, qsa_class):
    # Only the complete, known Flash-Next float singleton layout is supported.
    if idx != 0 or len(entries) != 48:
        return None
    if sum(type(c) is arrays_class for c in entries) != 36:
        return None
    for c in entries:
        if type(c) is arrays_class:
            if any(a is None or a.ndim == 0 or a.shape[0] != 1 for a in c.cache):
                return None
        elif type(c) is batch_qsa_class:
            kv = c.kv_cache
            if (kv.keys is None or kv.values is None or kv.keys.shape[0] != 1
                    or c.left_padding.size != 1 or int(c.left_padding[0].item()) != 0
                    or not 0 < kv._idx <= kv.keys.shape[2]):
                return None
            if c.index_keys is None or c.index_position_ids is None:
                return None
            if c.index_keys.shape[0] != 1 or c.index_position_ids.ndim not in (2, 3):
                return None
            if c.index_position_ids.shape[-2] != 1:
                return None
        else:
            return None
    rows = []
    for c in entries:
        if type(c) is arrays_class:
            row = arrays_class(len(c.cache))
            row.cache = [a[0:1] for a in c.cache]
        else:
            row = qsa_class()
            kv = c.kv_cache
            # Keep strided views here. The manager's detached clone performs
            # the one required materialization after its original admission.
            row.keys = kv.keys[0:1, :, :kv._idx]
            row.values = kv.values[0:1, :, :kv._idx]
            row.offset = kv._idx
            row.index_keys = c.index_keys[0:1, :c.index_offset]
            p = c.index_position_ids
            row.index_position_ids = (p[:, 0:1, :c.index_offset] if p.ndim == 3
                                      else p[0:1, :c.index_offset])
            if c.index_block_keys is not None:
                row.index_block_keys = c.index_block_keys[0:1]
                row.index_block_ratio = c.index_block_ratio
        rows.append(row)
    return rows


def patch_snapshot_views(coordinator_class, apc, *, accepts_model, borrow):
    if getattr(coordinator_class, '_qwen_snapshot_views_installed', False):
        return
    current = ContextVar('qwen_snapshot_views', default=False)
    store = coordinator_class.store_checkpoint
    extract = apc.extract_prompt_cache_from_batch

    @functools.wraps(store)
    def checkpoint(self, *args, **kwargs):
        eligible = self.enabled and self.is_checkpoint and accepts_model(self.model)
        token = current.set(eligible)
        try:
            return store(self, *args, **kwargs)
        finally:
            current.reset(token)

    @functools.wraps(extract)
    def extracted(entries, idx):
        if current.get():
            row = borrow(entries, idx)
            if row is not None:
                return row
        return extract(entries, idx)

    coordinator_class.store_checkpoint = checkpoint
    apc.extract_prompt_cache_from_batch = extracted
    coordinator_class._qwen_snapshot_views_installed = True


def install_snapshot_views():
    import logging
    from mlx_vlm import apc
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, QSAKVCache, BatchQSAKVCache

    def borrow(entries, idx):
        row = borrow_flash_row(entries, idx, arrays_class=ArraysCache,
                               batch_qsa_class=BatchQSAKVCache, qsa_class=QSAKVCache)
        if row is not None:
            logging.getLogger(__name__).info('BORROW_FLASH_CHECKPOINT_ROW layers=%d', len(row))
        return row

    patch_snapshot_views(
        APCCoordinator, apc,
        accepts_model=lambda m: type(getattr(m, 'language_model', m)) is LanguageModel,
        borrow=borrow)


def install_27b_snapshot_views():
    import logging
    from mlx_vlm import apc
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    def borrow(entries, idx):
        row = borrow_27b_row(entries, idx, arrays_class=ArraysCache,
                             batch_kv_class=BatchKVCache, kv_class=KVCache)
        if row is not None:
            logging.getLogger(__name__).info('BORROW_27B_CHECKPOINT_ROW layers=%d', len(row))
        return row

    patch_snapshot_views(
        APCCoordinator, apc,
        accepts_model=lambda m: type(getattr(m, 'language_model', m)) is LanguageModel,
        borrow=borrow)
