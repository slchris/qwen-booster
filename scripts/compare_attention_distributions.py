"""Compare attention numerics on identical real-model teacher-forced prefixes.

Baseline greedy continuation is diagnostic input, not a ground-truth answer.
No timing or semantic acceptance claim is made by this experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokens', type=int, default=32)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error('Need at least two continuation tokens')
    output = Path(args.output)
    if output.exists():
        parser.error('Refusing to overwrite evidence')
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import qwen3_next
    from mlx_lm.models.cache import make_prompt_cache
    from qwen_booster.cli import environment, local_model
    from qwen_booster.context import ContextBudget

    path, config = local_model(args.model)
    native = config.get('text_config', config)['max_position_embeddings']
    budget = ContextBudget.from_config(config, native)
    mx.set_memory_limit(64_000_000_000)
    mx.set_cache_limit(2_000_000_000)
    model, tokenizer = load(path)
    mx.eval(model.parameters())
    original = qwen3_next.scaled_dot_product_attention
    calls = 0

    def fused(q, k, v, cache, scale, mask, sinks=None):
        nonlocal calls
        if q.shape[2] > 8 and q.shape[-1] == 256 and not hasattr(cache, 'bits'):
            calls += 1
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                mask=mask, sinks=sinks, force_fused=True)
        return original(q, k, v, cache, scale, mask, sinks)

    suite = Path(args.suite)
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    with output.open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row) + '\n'); handle.flush()
        emit({'type': 'metadata', 'scope': __doc__, 'environment': environment(),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), suite, Path(qwen3_next.__file__), path / 'config.json']},
              'continuation_tokens': args.tokens, 'acceptance_complete': False,
              'runtime_default_enabled': False})
        try:
            for case in cases:
                opts = {k: case[k] for k in ['enable_thinking', 'preserve_thinking', 'reasoning_effort'] if k in case}
                tokens = tokenizer.apply_chat_template(case['messages'], tokenize=True,
                    add_generation_prompt=True, return_dict=False, **opts)
                prompt_hash = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                if 'prompt_token_sha256' in case:
                    assert prompt_hash == case['prompt_token_sha256']
                budget.admit(len(tokens), args.tokens)
                baseline_probs, continuation, comparisons = [], [], []
                for mode in ['default', 'fused']:
                    qwen3_next.scaled_dot_product_attention = original if mode == 'default' else fused
                    calls = 0
                    cache = make_prompt_cache(model)
                    mx.clear_cache()
                    prompt = mx.array([tokens], dtype=mx.uint32)
                    print(json.dumps({'case': case['id'], 'mode': mode, 'input_tokens': len(tokens)}), flush=True)
                    for start in range(0, len(tokens) - 1, 2048):
                        model(prompt[:, start:min(start + 2048, len(tokens) - 1)], cache=cache)
                        mx.eval([c.state for c in cache]); mx.clear_cache()
                    feed = tokens[-1]
                    for position in range(args.tokens):
                        logits = model(mx.array([[feed]], dtype=mx.uint32), cache=cache)[:, -1, :].astype(mx.float32)
                        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                        mx.eval(logp, [c.state for c in cache])
                        assert bool(mx.all(mx.isfinite(logp)).item())
                        top = int(mx.argmax(logp).item())
                        if mode == 'default':
                            continuation.append(top); baseline_probs.append(logp)
                        else:
                            ref = baseline_probs[position]
                            target = continuation[position]
                            comparisons.append({'position': position, 'baseline_top': target,
                                'candidate_top': top, 'top_equal': target == top,
                                'baseline_token_nll': float(-ref[0, target].item()),
                                'candidate_token_nll': float(-logp[0, target].item()),
                                'kl_baseline_to_candidate': float(mx.sum(mx.exp(ref) * (ref - logp)).item()),
                                'max_abs_logprob_delta': float(mx.max(mx.abs(ref - logp)).item())})
                        feed = continuation[position]
                    del cache
                    if mode == 'fused':
                        assert calls > 0
                emit({'type': 'comparison', 'case_id': case['id'], 'input_tokens': len(tokens),
                      'prompt_sha256': prompt_hash, 'continuation': continuation,
                      'candidate_calls': calls, 'positions': comparisons,
                      'top_equal_count': sum(p['top_equal'] for p in comparisons),
                      'mean_kl': sum(p['kl_baseline_to_candidate'] for p in comparisons) / len(comparisons),
                      'mean_nll_delta': sum(p['candidate_token_nll'] - p['baseline_token_nll'] for p in comparisons) / len(comparisons)})
                del baseline_probs, logp, logits
                mx.clear_cache()
            emit({'type': 'complete', 'cases': len(cases), 'acceptance_complete': False})
        finally:
            qwen3_next.scaled_dot_product_attention = original


if __name__ == '__main__':
    main()
