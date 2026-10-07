"""Experimental singleton warm-cache normalization for pinned Qwen3.8-27B.

The backend restores APC rows as batch caches. Qwen3_5's singleton forward
extracts and merges those caches on every token, discarding spare KV capacity.
Convert once before suffix prefill, matching its ordinary singleton cold path.
"""

import functools
import inspect


def patch_prompt_batch(batch_class, *, normalize, accepts_model):
    if getattr(batch_class, '_qwen_booster_single_warm_installed', False):
        return
    original = batch_class.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        warm = values.get('warm_cache')
        if (warm is not None and len(values['input_ids']) == 1
                and values.get('draft_model') is None
                and values.get('draft_kind') is None
                and values.get('kv_bits') is None
                and not any(values.get('right_pad_per_row') or [])
                and accepts_model(values['model'])):
            replacement = normalize(warm)
            if replacement is not None:
                values['warm_cache'] = replacement
        original(*bound.args, **bound.kwargs)

    batch_class.__init__ = initialize
    batch_class._qwen_booster_single_warm_installed = True


def normalize_qwen_cache(entries):
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache

    # Fail closed for quantized, sparse, custom or genuinely batched caches.
    if not entries or any(type(c) not in (ArraysCache, BatchKVCache) for c in entries):
        return None
    kv = [c for c in entries if type(c) is BatchKVCache]
    if not kv or any(c.empty() or c.batch_size != 1
                     or c.left_padding.tolist() != [0] for c in kv):
        return None
    for c in entries:
        if type(c) is ArraysCache:
            if any(a is not None and a.shape[0] != 1 for a in c.cache):
                return None
            if c.lengths is not None:
                return None
            if c.left_padding is not None and c.left_padding.tolist() != [0]:
                return None
    converted = [c.extract(0) for c in entries]
    mx.eval([c.state for c in converted])
    return converted


def install_single_warm_cache():
    import logging
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    def normalize(entries):
        converted = normalize_qwen_cache(entries)
        if converted is not None:
            logging.getLogger(__name__).info(
                'Normalized singleton Qwen warm cache once: layers=%d', len(converted))
        return converted

    patch_prompt_batch(PromptProcessingBatch, normalize=normalize,
        accepts_model=lambda m: type(getattr(m, 'language_model', m)) is LanguageModel)
