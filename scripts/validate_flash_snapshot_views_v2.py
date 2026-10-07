"""CPU/GPU validation of the pinned Flash cache contract and detached snapshots."""

import argparse
import json
import gc
from pathlib import Path

import mlx.core as mx


from mlx_vlm import apc
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache
from qwen_booster.snapshot_views import borrow_flash_row


def borrow(entries, idx=0):
    return borrow_flash_row(entries, idx, arrays_class=ArraysCache,
                            batch_qsa_class=BatchQSAKVCache, qsa_class=QSAKVCache)


def equal(a, b):
    if isinstance(a, mx.array):
        return isinstance(b, mx.array) and a.dtype == b.dtype and a.shape == b.shape and bool(mx.array_equal(a, b))
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def make_entries(multimodal, n):
    entries = []
    for i in range(48):
        if i % 4 != 3:
            c = ArraysCache(4)
            # Convolution, recurrence, PLE convolution and recent token IDs.
            c.cache = [mx.full((1, 3, 8), i, dtype=mx.bfloat16),
                       mx.full((1, 2, 8, 8), i, dtype=mx.float32),
                       mx.full((1, 3, 4), i, dtype=mx.bfloat16),
                       mx.full((1, 7), i, dtype=mx.int32)]
        else:
            c = BatchQSAKVCache([0])
            kv = c.kv_cache
            kv.keys = mx.arange(4096).reshape(1, 2, 256, 8).astype(mx.bfloat16)
            kv.values = kv.keys + i
            kv._idx = n
            kv.offset = mx.array([n])
            c.index_keys = mx.arange(n * 8).reshape(1, n, 8).astype(mx.bfloat16)
            c.index_position_ids = (mx.arange(3 * n).reshape(3, 1, n) if multimodal
                                    else mx.arange(n).reshape(1, n))
            c.index_offset = n
            c.index_block_keys = mx.full((1, (n + 3) // 4, 8), i, dtype=mx.bfloat16)
            c.index_block_ratio = 4
        entries.append(c)
    return entries


def arrays(entries):
    out = []
    for c in entries:
        apc._collect_mx_arrays(c.state, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    cases = []
    for multimodal, n in [(m, n) for m in (False, True) for n in (17, 255, 256)]:
        entries = make_entries(multimodal, n)
        mx.eval(arrays(entries))
        measurements = {}
        for name, normalize in [("original", lambda e: apc.extract_prompt_cache_from_batch(e, 0)), ("borrowed", borrow)]:
            gc.collect(); mx.clear_cache()
            before = mx.get_active_memory()
            row = normalize(entries)
            mx.eval(arrays(row))
            measurements[name] = mx.get_active_memory() - before
            del row
        original = apc.extract_prompt_cache_from_batch(entries, 0)
        candidate = borrow(entries)
        assert candidate is not None
        assert all(equal(a.state, b.state) and equal(a.meta_state, b.meta_state)
                   for a, b in zip(original, candidate))
        reference = apc._clone_prompt_cache_for_apc(original)
        detached = apc._clone_prompt_cache_for_apc(candidate)
        assert all(equal(a.state, b.state) for a, b in zip(reference, detached))
        source_arrays = []
        for entry in entries:
            if type(entry) is ArraysCache:
                source_arrays.extend(entry.cache)
            else:
                source_arrays.extend([entry.kv_cache.keys, entry.kv_cache.values,
                               entry.index_keys, entry.index_position_ids,
                               entry.index_block_keys])
        for array in source_arrays:
            array[...] = 123
        mx.eval(source_arrays)
        assert all(bool(mx.all(array == 123)) for array in source_arrays)
        assert all(equal(a.state, b.state) for a, b in zip(reference, detached))
        # Change the saved copy too: neither the source nor independent baseline
        # may be mutated by updates to a detached snapshot.
        for target in arrays(detached):
            target[...] = 77
        mx.eval(arrays(detached))
        assert all(bool(mx.all(a == 123)) for a in source_arrays)
        assert not equal(reference[3].state, detached[3].state)
        assert bool(mx.all(entries[0].cache[0] == 123))
        assert bool(mx.all(reference[0].cache[0] == 0))
        assert borrow(entries, 1) is None
        assert borrow(entries[:-1]) is None
        entries[3].left_padding = mx.array([1])
        assert borrow(entries) is None
        cases.append({'offset': n, 'capacity': 256, 'normalization_active_delta_bytes': measurements, 'position_dimensions': 3 if multimodal else 2,
                      'state_and_metadata_equal': True, 'detached_mutation_isolation': True,
                      'unsupported_layouts_rejected': True})
        del entries, original, candidate, reference, detached, source_arrays
    report = {'device': args.device, 'contract_validation_passed': True, 'cases': cases,
              'full_model_quality_validated': False, 'full_model_admission_validated': False}
    with Path(args.output).open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
