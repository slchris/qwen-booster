"""Small CPU-only check of complete Flash hybrid-state disk round trips.

This does not establish model inference parity, large-cache latency or memory
admission. Uses a disposable local directory and the pinned backend serializer.
"""

import argparse
import json
import tempfile
from pathlib import Path

from validate_snapshot_views import apc, borrow, equal, make_entries, mx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    observations = []
    for multimodal in (False, True):
        source = make_entries(multimodal)
        borrowed = borrow(source)
        reference = apc._clone_prompt_cache_for_apc(borrowed)
        with tempfile.TemporaryDirectory(prefix='qwen-disk-state-') as directory:
            store = apc.DiskBlockStore(Path(directory), namespace='cpu-contract',
                                       max_bytes=10_000_000)
            try:
                # The fallback path serializes borrowed state synchronously,
                # before the producer is allowed to modify any live state.
                assert store.save_exact_cache(123, list(range(17)), 9,
                                              borrowed, synchronous=True)
            finally:
                store.close()
            for c in source:
                arrays = c.cache if hasattr(c, 'cache') else [
                    c.kv_cache.keys, c.kv_cache.values, c.index_keys,
                    c.index_position_ids, c.index_block_keys]
                for array in arrays:
                    array[...] = 123
                mx.eval(arrays)
            # Reopen the index to avoid mistaking in-memory retention for disk.
            restored_store = apc.DiskBlockStore(Path(directory), namespace='cpu-contract',
                                                max_bytes=10_000_000)
            try:
                loaded = restored_store.load_exact_cache(123)
                assert loaded is not None
                token_ids, extra_hash, restored = loaded
                assert token_ids == tuple(range(17)) and extra_hash == 9
                assert len(restored) == len(reference) == 48
                assert all(type(a) is type(b) and equal(a.state, b.state)
                           and equal(a.meta_state, b.meta_state)
                           for a, b in zip(reference, restored))
                size = sum(p.stat().st_size for p in Path(directory).rglob('*.safetensors'))
            finally:
                restored_store.close()
        observations.append({'position_dimensions': 3 if multimodal else 2,
                             'complete_state_round_trip_equal': True,
                             'reopened_disk_index': True, 'disk_bytes': size})
    report = {'cpu_disk_contract_passed': True, 'cases': observations,
              'full_model_inference_tested': False, 'large_cache_memory_tested': False,
              'default_disk_cache_enabled': False}
    with Path(args.output).open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
