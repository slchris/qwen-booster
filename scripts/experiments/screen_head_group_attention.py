"""Paired SDPA head-group screen; full KV retained, runtime unchanged."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or Path(args.output).exists():
        parser.error('positive repeats and a new output path required')
    import mlx.core as mx
    from mlx_vlm.models.base import scaled_dot_product_attention as attention

    mx.set_memory_limit(110_000_000_000)
    mx.set_cache_limit(2_000_000_000)

    def compute(q, k, v, group):
        if group == 4:
            return attention(q, k, v, cache=None, scale=256 ** -0.5, mask='causal')
        parts = []
        for start in range(0, 4, group):
            part = attention(q[:, start * 6:(start + group) * 6],
                             k[:, start:start + group], v[:, start:start + group],
                             cache=None, scale=256 ** -0.5, mask='causal')
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, axis=1)

    with Path(args.output).open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            print(json.dumps(row), flush=True)

        emit({'type': 'metadata', 'scope': __doc__, 'query_shape': [1, 24, 2048, 256],
              'kv_heads': 4, 'dtype': 'bfloat16', 'seed': 1729,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': 'Random activations, one seed, isolated attention. '
                             'Numerical comparison is not full-model quality acceptance.'})
        for length in [51200, 102400, 153600, 204800]:
            mx.random.seed(1729)
            q = mx.random.normal((1, 24, 2048, 256)).astype(mx.bfloat16)
            k = mx.random.normal((1, 4, length, 256)).astype(mx.bfloat16)
            v = mx.random.normal((1, 4, length, 256)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            reference = compute(q, k, v, 4)
            mx.eval(reference)
            measurements = {g: [] for g in [4, 2, 1]}
            for repeat in range(args.repeats + 1):
                for group in ([4, 2, 1] if repeat % 2 == 0 else [1, 2, 4]):
                    mx.clear_cache()
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    out = compute(q, k, v, group)
                    mx.eval(out)
                    seconds = time.perf_counter() - start
                    peak = mx.get_peak_memory()
                    diff = mx.abs(out.astype(mx.float32) - reference.astype(mx.float32))
                    exact = bool(mx.all(out == reference).item())
                    max_error = float(mx.max(diff).item())
                    assert out.shape == reference.shape and bool(mx.all(mx.isfinite(out)).item())
                    row = {'type': 'measurement', 'kv_tokens': length, 'kv_heads_per_call': group,
                           'repeat': repeat, 'warmup': repeat == 0, 'seconds': seconds,
                           'peak_mlx_bytes': peak, 'bitwise_equal': exact, 'max_abs_error': max_error}
                    emit(row)
                    if repeat:
                        measurements[group].append(row)
                    del out, diff
            for group, rows in measurements.items():
                emit({'type': 'summary', 'kv_tokens': length, 'kv_heads_per_call': group,
                      'median_seconds': statistics.median(r['seconds'] for r in rows),
                      'peak_mlx_bytes': max(r['peak_mlx_bytes'] for r in rows),
                      'all_bitwise_equal': all(r['bitwise_equal'] for r in rows),
                      'max_abs_error': max(r['max_abs_error'] for r in rows)})
            del q, k, v, reference
            mx.clear_cache()


if __name__ == '__main__':
    main()
