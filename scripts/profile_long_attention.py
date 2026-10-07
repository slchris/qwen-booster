"""Measure existing 27B attention shapes; no runtime patch or speedup claim."""

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    path = Path(args.output)
    if path.exists():
        parser.error('refusing to overwrite output')

    import mlx.core as mx
    from mlx_vlm.models.base import scaled_dot_product_attention

    mx.set_memory_limit(110_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    source = Path(inspect.getfile(scaled_dot_product_attention))
    with path.open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(json.dumps(row), flush=True)

        emit({'type': 'metadata', 'scope': __doc__, 'dtype': 'bfloat16',
              'query_shape': [1, 24, 2048, 256], 'kv_heads': 4,
              'mask': 'causal', 'seed': 1729, 'repeats': args.repeats,
              'source_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
                                __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'limitations': 'Random activations, isolated SDPA only. Excludes model projections, '
                             'GDN, MLP, cache updates and scheduling. Not a full-model speed estimate.'})
        for klen in [51200, 102400, 153600, 204800]:
            mx.random.seed(1729)
            q = mx.random.normal((1, 24, 2048, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 4, klen, 256)).astype(mx.bfloat16)
            v = mx.random.normal((1, 4, klen, 256)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            elapsed = []
            for repeat in range(args.repeats + 1):
                mx.reset_peak_memory()
                start = time.perf_counter()
                out = scaled_dot_product_attention(q, k, v, cache=None,
                                                   scale=256 ** -0.5, mask='causal')
                mx.eval(out)
                seconds = time.perf_counter() - start
                peak = mx.get_peak_memory()
                assert out.shape == q.shape
                assert bool(mx.all(mx.isfinite(out)).item())
                emit({'type': 'measurement', 'kv_tokens': klen, 'repeat': repeat,
                      'warmup': repeat == 0, 'seconds': seconds, 'peak_mlx_bytes': peak})
                if repeat:
                    elapsed.append(seconds)
                del out
            emit({'type': 'summary', 'kv_tokens': klen,
                  'median_seconds': statistics.median(elapsed)})
            del q, k, v
            mx.clear_cache()


if __name__ == '__main__':
    main()
