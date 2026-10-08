"""Verify a finished real backend batch releases retained MLX buffers immediately."""

import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_vlm.generate.ar import SpeculativeGenerationBatch
    from qwen_booster.mtp_lifecycle import release_finished_batch

    if Path(args.output).exists():
        parser.error('refusing to overwrite results')
    mx.clear_cache()
    baseline = mx.get_active_memory()
    batch = SpeculativeGenerationBatch.__new__(SpeculativeGenerationBatch)
    batch._finished = [True]
    batch.prompt_cache = [mx.ones((8192, 8192), dtype=mx.float32)]
    batch.hidden = batch.shared_kv_states = batch.prompt_tokens = batch.first_tokens = None
    mx.eval(batch.prompt_cache)

    def suspended(owner):
        yield None
        yield len(owner.prompt_cache)

    batch._rounds_iter = suspended(batch)
    next(batch._rounds_iter)
    retained = mx.get_active_memory()
    assert len(batch) == 0 and retained - baseline >= 8192 * 8192 * 4
    assert release_finished_batch(batch)
    gc.collect()
    mx.clear_cache()
    after = mx.get_active_memory()
    report = {'passed': retained - after >= 8192 * 8192 * 4,
              'baseline_active_bytes': baseline, 'finished_batch_retained_bytes': retained,
              'after_release_active_bytes': after, 'reclaimed_bytes': retained - after,
              'scope': 'real backend batch class with synthetic 256MiB MLX state; API parity and 200K reuse tested separately'}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not report['passed']:
        raise SystemExit('buffer release verification failed')


if __name__ == '__main__':
    main()
