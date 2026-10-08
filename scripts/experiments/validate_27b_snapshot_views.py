"""Real-array singleton 27B normalization and detached-snapshot contract probe."""
import argparse
import gc
import json
from pathlib import Path
import mlx.core as mx
from mlx_vlm import apc
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from qwen_booster.snapshot_views import borrow_27b_row


def borrow(entries, idx=0):
    return borrow_27b_row(entries, idx, arrays_class=ArraysCache,
                          batch_kv_class=BatchKVCache, kv_class=KVCache)


def equal(a, b):
    if isinstance(a, mx.array):
        return isinstance(b, mx.array) and a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b))
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def arrays(entries):
    out = []
    for c in entries:
        apc._collect_mx_arrays(c.state, out)
    return out


def make(n):
    out = []
    for i in range(64):
        if i % 4 != 3:
            c = ArraysCache(2)
            c.cache = [mx.full((1, 3, 8), i, dtype=mx.bfloat16),
                       mx.full((1, 2, 8, 8), i, dtype=mx.float32)]
        else:
            c = BatchKVCache([0])
            c.keys = mx.arange(4 * 256 * 16).reshape(1, 4, 256, 16).astype(mx.bfloat16)
            c.values = c.keys + i
            c._idx = n
            c.offset = mx.array([n])
        out.append(c)
    mx.eval(arrays(out))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    records = []
    for n in (17, 255, 256):
        entries = make(n)
        measurements = {}
        for name, normalize in [('original', lambda x: apc.extract_prompt_cache_from_batch(x, 0)), ('borrowed', borrow)]:
            gc.collect(); mx.clear_cache()
            before = mx.get_active_memory()
            row = normalize(entries)
            mx.eval(arrays(row))
            after = mx.get_active_memory()
            measurements[name] = after - before
            del row
        original = apc.extract_prompt_cache_from_batch(entries, 0)
        candidate = borrow(entries)
        assert candidate is not None
        assert all(equal(a.state, b.state) and equal(a.meta_state, b.meta_state) for a, b in zip(original, candidate))
        reference = apc._clone_prompt_cache_for_apc(original)
        detached = apc._clone_prompt_cache_for_apc(candidate)
        assert all(equal(a.state, b.state) for a, b in zip(reference, detached))
        source_arrays = [a for c in entries for a in (c.cache if type(c) is ArraysCache else [c.keys, c.values])]
        for a in source_arrays:
            a[...] = 123
        mx.eval(source_arrays)
        assert all(equal(a.state, b.state) for a, b in zip(reference, detached))
        for c in detached:
            for a in arrays([c]):
                a[...] = 77
        mx.eval(arrays(detached))
        assert all(bool(mx.all(a == 123)) for a in source_arrays)
        assert not equal(reference[3].state, detached[3].state)
        assert borrow(entries, 1) is None and borrow(entries[:-1]) is None
        entries[3].left_padding = mx.array([1])
        assert borrow(entries) is None
        records.append({'offset': n, 'capacity': 256, 'state_metadata_parity': True,
                        'source_and_snapshot_mutation_isolation': True,
                        'normalization_active_delta_bytes': measurements})
        del entries, original, candidate, reference, detached
    report = {'device': args.device, 'cases': records, 'full_model_quality_validated': False,
              'native_admission_validated': False}
    with Path(args.output).open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
