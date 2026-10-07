from types import SimpleNamespace

import pytest

from qwen_booster.fused_attention import patch_attention


@pytest.mark.parametrize('batch,length,width,cache,fused', [
    (1, 2048, 256, None, True),
    (1, 1, 256, None, False),
    (1, 3, 256, None, False),
    (1, 8, 256, None, False),
    (2, 2048, 256, None, False),
    (1, 2048, 128, None, False),
    (1, 2048, 256, SimpleNamespace(bits=4), False),
    (1, 2048, 256, SimpleNamespace(dequantize=lambda: None), False),
])
def test_dispatch_preserves_backend_paths(batch, length, width, cache, fused):
    calls = []
    module = SimpleNamespace(scaled_dot_product_attention=lambda *a: calls.append(('base', a)))
    stats = patch_attention(module, lambda *a, **kw: calls.append(('fused', a, kw)))
    q = SimpleNamespace(shape=(batch, 24, length, width))
    k = v = SimpleNamespace(shape=(batch, 4, 204800, width))
    mask, sinks = object(), object()
    module.scaled_dot_product_attention(q, k, v, cache, .0625, mask, sinks)
    assert calls[0][0] == ('fused' if fused else 'base')
    if fused:
        assert calls[0][1] == (q, k, v)
        assert calls[0][2] == dict(scale=.0625, mask=mask, sinks=sinks, force_fused=True)
    else:
        assert calls[0][1] == (q, k, v, cache, .0625, mask, sinks)
    assert stats['fused_calls'] == int(fused)
    assert patch_attention(module, None) is stats
