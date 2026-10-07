"""Full-prefix diagnostics with block synchronization only at selected chunks.

Timings are diagnostic, not an end-to-end speedup benchmark. No prefix is skipped.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--length', type=int, default=204800)
    parser.add_argument('--step', type=int, default=2048)
    parser.add_argument('--windows', type=int, nargs='+', default=[0, 51200, 102400, 196608])
    args = parser.parse_args()
    if args.length < 2 or args.step < 1 or any(
        n < 0 or n >= args.length - 1 or n % args.step for n in args.windows
    ) or len(set(args.windows)) != len(args.windows):
        parser.error('Invalid dimensions or duplicate/unaligned window starts')
    output = Path(args.output)
    if output.exists():
        parser.error('Refusing to overwrite evidence')

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from qwen_booster.cli import environment, local_model, prompt_tokens
    from qwen_booster.context import ContextBudget

    path, config = local_model(args.model)
    native = config.get('text_config', config)['max_position_embeddings']
    ContextBudget.from_config(config, native).admit(args.length, 1)
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    token_ids = prompt_tokens(tokenizer, args.length)
    tokens = mx.array([token_ids], dtype=mx.uint32)
    mx.eval(tokens)
    cache = make_prompt_cache(model)
    chunk_start = 0
    chunk_tokens = 0
    active = False
    records = []
    originals = []

    class TimedBlock(nn.Module):
        def __init__(self, inner, label, index):
            super().__init__()
            self.inner, self.label, self.index = inner, label, index

        def __call__(self, *inputs, **kwargs):
            if not active:
                return self.inner(*inputs, **kwargs)
            state = kwargs.get('cache')
            if state is None and len(inputs) >= 3:
                state = inputs[2]
            kv_before = getattr(state, 'offset', None) if self.label == 'self_attn' else None
            mx.eval(inputs[0])
            mx.synchronize()
            started = time.perf_counter()
            result = self.inner(*inputs, **kwargs)
            mx.eval(result)
            if state is not None:
                mx.eval(state.state)
            mx.synchronize()
            records.append({'type': 'block', 'block': self.label, 'layer': self.index,
                            'chunk_start': chunk_start, 'chunk_tokens': chunk_tokens,
                            'kv_before': kv_before,
                            'kv_after': getattr(state, 'offset', None) if kv_before is not None else None,
                            'seconds': time.perf_counter() - started})
            return result

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()

        sources = [Path(__file__)] + sorted(Path('src/qwen_booster').glob('*.py'))
        emit({'type': 'metadata', 'environment': environment(), 'prompt_tokens': len(token_ids),
              'prompt_sha256': hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
              'step': args.step, 'windows': args.windows, 'native_context_limit': native,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'model_config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
              'scope': __doc__, 'acceptance_complete': False})
        try:
            for index, layer in enumerate(model.layers):
                for label in ['mlp', 'linear_attn' if layer.is_linear else 'self_attn']:
                    original = getattr(layer, label)
                    originals.append((layer, label, original))
                    setattr(layer, label, TimedBlock(original, label, index))
            mx.clear_cache()
            mx.reset_peak_memory()
            start_time = time.perf_counter()
            for chunk_start in range(0, args.length - 1, args.step):
                end = min(chunk_start + args.step, args.length - 1)
                chunk_tokens = end - chunk_start
                active = chunk_start in args.windows
                started = time.perf_counter()
                model(tokens[:, chunk_start:end], cache=cache)
                mx.eval([c.state for c in cache])
                mx.synchronize()
                seconds = time.perf_counter() - started
                for row in records:
                    emit(row)
                records.clear()
                row = {'type': 'chunk', 'start': chunk_start, 'end': end,
                       'instrumented': active, 'seconds': seconds,
                       'peak_memory_gb': mx.get_peak_memory() / 1e9}
                emit(row)
                print(json.dumps(row), flush=True)
                mx.clear_cache()
            active = False
            logits = model(tokens[:, -1:], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token, [c.state for c in cache])
            mx.synchronize()
            emit({'type': 'complete', 'processed_input_tokens': args.length,
                  'first_token_id': int(token.item()), 'diagnostic_elapsed_seconds': time.perf_counter() - start_time,
                  'peak_memory_gb': mx.get_peak_memory() / 1e9, 'acceptance_complete': False})
        finally:
            for layer, label, original in originals:
                setattr(layer, label, original)


if __name__ == '__main__':
    main()
