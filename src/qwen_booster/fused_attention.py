"""Opt-in full-attention prefill dispatch for the pinned Qwen3.5 VLM backend."""


def patch_attention(module, fused_sdpa):
    """Preserve decode, speculative small blocks, and quantized cache paths."""
    if getattr(module, '_qwen_fused_prefill_stats', None) is not None:
        return module._qwen_fused_prefill_stats
    original = module.scaled_dot_product_attention
    stats = {'fused_calls': 0, 'fallback_calls': 0}

    def attention(queries, keys, values, cache, scale, mask, sinks=None):
        if (queries.shape[0] == 1 and queries.shape[-2] > 8
                and queries.shape[-1] == 256
                and not hasattr(cache, 'bits')
                and not hasattr(cache, 'dequantize')
                and hasattr(keys, 'shape') and hasattr(values, 'shape')):
            stats['fused_calls'] += 1
            return fused_sdpa(queries, keys, values, scale=scale, mask=mask,
                              sinks=sinks, force_fused=True)
        stats['fallback_calls'] += 1
        return original(queries, keys, values, cache, scale, mask, sinks)

    module.scaled_dot_product_attention = attention
    module._qwen_fused_prefill_stats = stats
    return stats


def install_fused_prefill():
    import logging
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5 import language
    def fused(*args, **kwargs):
        if stats['fused_calls'] == 1:
            logging.getLogger(__name__).warning(
                'FUSED_PREFILL_ACTIVE query_shape=%s key_shape=%s',
                args[0].shape, args[1].shape)
        return mx.fast.scaled_dot_product_attention(*args, **kwargs)
    stats = patch_attention(language, fused)
    return stats
