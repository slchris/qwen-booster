"""Cold, paired free-generation task screen; no HTTP/MTP/cache acceptance."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output path and positive repeats')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models import qwen3_next
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from qwen_booster.cli import environment, local_model
    from qwen_booster.context import ContextBudget
    from qwen_booster.grading import grade_answer, grade_json_format
    from qwen_booster.reasoning import ReasoningParser

    cases = [json.loads(line) for line in args.suite.read_text().splitlines()]
    assert cases and len({c['id'] for c in cases}) == len(cases)
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

    with args.output.open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
        emit({'type': 'metadata', 'scope': __doc__, 'environment': environment(),
              'source_sha256': {str(p): sha(p) for p in
                  (Path(__file__), args.suite, Path(qwen3_next.__file__), path / 'config.json')},
              'repeats': args.repeats, 'prefill_step_size': 2048,
              'cache': 'fresh per request', 'warmup_pairs': 0,
              'acceptance_complete': False, 'runtime_default_enabled': False})
        try:
            for case in cases:
                opts = {k: case[k] for k in ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if k in case}
                tokens = tokenizer.apply_chat_template(case['messages'], tokenize=True,
                    add_generation_prompt=True, return_dict=False, **opts)
                prompt_sha = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                assert len(tokens) == case['prompt_tokens']
                assert prompt_sha == case['prompt_token_sha256']
                budget.admit(len(tokens), case['max_tokens'])
                for repeat in range(args.repeats):
                    paired = {}
                    order = ['default', 'fused'] if repeat % 2 == 0 else ['fused', 'default']
                    for mode in order:
                        qwen3_next.scaled_dot_product_attention = original if mode == 'default' else fused
                        calls = 0
                        gc.collect()
                        mx.synchronize()
                        mx.clear_cache()
                        mx.reset_peak_memory()
                        mx.random.seed(case['seed'])
                        sampler = make_sampler(temp=case['temperature'], top_p=case['top_p'], top_k=case['top_k'])
                        processors = make_logits_processors(presence_penalty=case.get('presence_penalty', 0),
                                                            presence_context_size=None)
                        split = ReasoningParser(thinking=case.get('enable_thinking', False))
                        content, reasoning, ids = [], [], []
                        first = visible = None
                        last_progress = 0
                        print(json.dumps({'case': case['id'], 'repeat': repeat, 'mode': mode, 'stage': 'started'}), flush=True)
                        started = time.perf_counter()

                        def progress(done, total):
                            nonlocal last_progress
                            if done - last_progress >= 16384 or done == total:
                                print(json.dumps({'case': case['id'], 'mode': mode, 'stage': 'prefill',
                                                  'processed': done, 'total': total}), flush=True)
                                last_progress = done

                        iterator = stream_generate(model, tokenizer, tokens, max_tokens=case['max_tokens'],
                            sampler=sampler, logits_processors=processors, prefill_step_size=2048,
                            prompt_progress_callback=progress)
                        try:
                            for response in iterator:
                                now = time.perf_counter()
                                if first is None:
                                    first = now - started
                                ids.append(int(response.token))
                                parts = split.feed(response.text, final=response.finish_reason is not None)
                                answer = parts.get('content', '')
                                content.append(answer)
                                reasoning.append(parts.get('reasoning_content', ''))
                                if visible is None and answer.strip():
                                    visible = now - started
                        finally:
                            iterator.close()
                            mx.synchronize()
                        elapsed = time.perf_counter() - started
                        answer = ''.join(content)
                        row = {'type': 'measurement', 'case_id': case['id'], 'repeat': repeat, 'mode': mode,
                            'input_tokens': len(tokens), 'prompt_sha256': prompt_sha,
                            'max_tokens': case['max_tokens'], 'seed': case['seed'],
                            'sampling': {k: case.get(k) for k in ('temperature', 'top_p', 'top_k', 'presence_penalty')},
                            'candidate_calls': calls, 'total_seconds': elapsed, 'first_token_seconds': first,
                            'first_visible_seconds': visible, 'prefill_tokens_per_second': response.prompt_tps,
                            'decode_tokens_per_second': response.generation_tps,
                            'peak_mlx_gb': response.peak_memory, 'finish_reason': response.finish_reason,
                            'output_tokens': response.generation_tokens, 'output_token_ids': ids,
                            'content': answer, 'reasoning': ''.join(reasoning),
                            'answer_correct': grade_answer(answer, case['grading']),
                            'json_format_correct': grade_json_format(answer, case['grading'])}
                        emit(row)
                        assert response.prompt_tokens == len(tokens)
                        assert mode != 'fused' or calls > 0
                        paired[mode] = row
                        del response, iterator
                    emit({'type': 'pair', 'case_id': case['id'], 'repeat': repeat,
                          'token_parity': paired['default']['output_token_ids'] == paired['fused']['output_token_ids'],
                          'both_answers_correct': all(r['answer_correct'] and r['finish_reason'] == 'stop' for r in paired.values()),
                          'both_json_formats_correct': all(r['json_format_correct'] is not False for r in paired.values()),
                          'first_token_speedup': paired['default']['first_token_seconds'] / paired['fused']['first_token_seconds']})
            emit({'type': 'complete', 'cases': len(cases), 'pairs': len(cases) * args.repeats,
                  'acceptance_complete': False})
        except BaseException as exc:
            emit({'type': 'error', 'error_type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            qwen3_next.scaled_dot_product_attention = original


if __name__ == '__main__':
    main()
