"""Independent review of complete alternating 200K histories, preserving failures."""
import copy
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prefix = 'results/credit-27b-switching-conversation-200k-v1'
    stem = prefix + '-27b-combined'
    config_path = prefix + '-config.json'
    config = json.loads(Path(config_path).read_text())
    assert all(sha(p) == h for p, h in config['source_sha256'].items())
    status_path = prefix + '-status.json'
    assert json.loads(Path(status_path).read_text())['stage'] == 'observations_finished_pending_review'
    raw_path = stem + '.jsonl'
    raw = [json.loads(x) for x in Path(raw_path).read_text().splitlines()]
    assert len(raw) == 30 and raw[0]['type'] == 'metadata' and raw[-1]['type'] == 'summary'
    names = ['a_initial', 'b_initial', 'a_revisit', 'b_revisit']
    names += [f'{key}_turn_{turn}' for turn in range(1, 11) for key in ('a', 'b')]
    names += ['a_final_revisit', 'b_final_revisit', 'a_final_cold', 'b_final_cold']
    data = raw[1:-1]
    assert [r['name'] for r in data] == names
    by = {r['name']: r for r in data}
    suite = 'results/document-switching-200k-suite-v1.jsonl'
    assert sha(suite) == raw[0]['suite_sha256']
    cases = [json.loads(x) for x in Path(suite).read_text().splitlines()]
    assert [c['id'] for c in cases] == config['case_ids']
    from transformers import AutoTokenizer
    model = Path(config['runs'][0]['model'])
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    model_config = json.loads((model / 'config.json').read_text())
    native = model_config.get('text_config', model_config)['max_position_embeddings']
    assert native == 262144
    audits, expected_cache = {}, {}

    def audit(name, history):
        tokens = tokenizer.apply_chat_template(history['messages'], tokenize=True,
            add_generation_prompt=True, return_dict=False, **{
                k: history[k] for k in ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if k in history})
        assert 204800 <= len(tokens) and len(tokens) + history['max_tokens'] <= native
        audits[name] = {'input_tokens': len(tokens),
            'token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'messages_sha256': hashlib.sha256(json.dumps(history['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}

    for key, case in zip(('a', 'b'), cases):
        history = copy.deepcopy(case)
        audit(key + '_initial', history)
        audit(key + '_revisit', history)
        expected_cache[key + '_revisit'] = audits[key + '_initial']['input_tokens'] - 1
        previous = key + '_initial'
        for turn in range(1, 11):
            prior = by[previous]['result']
            assistant = {'role': 'assistant', 'content': prior['text']}
            if prior.get('reasoning'):
                assistant['reasoning_content'] = prior['reasoning']
            history['messages'].extend([assistant, {'role': 'user',
                'content': '请重新回答最初的用户任务，严格遵守最初要求的输出格式。'}])
            name = f'{key}_turn_{turn}'
            audit(name, history)
            expected_cache[name] = audits[previous]['input_tokens'] - 1
            previous = name
        audit(key + '_final_revisit', history)
        audit(key + '_final_cold', history)
        expected_cache[key + '_final_revisit'] = audits[previous]['input_tokens'] - 1
    observations, reuse = [], []
    for row in data:
        name, r = row['name'], row['result']
        audited = audits[name]
        correct = json.loads(r['text']) == cases[0 if name[0] == 'a' else 1]['grading']['expected']
        assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
        assert row['messages_sha256'] == audited['messages_sha256']
        assert row['input_audit']['prompt_token_sha256'] == audited['token_sha256']
        u, t = r['usage'], r['server_timings']
        assert u['prompt_tokens'] == audited['input_tokens'] == row['input_audit']['actual_prompt_tokens']
        assert u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
        assert t['prompt_n'] + t['cache_n'] == u['prompt_tokens']
        assert 0 <= t['cache_n'] <= u['prompt_tokens']
        assert t['cache_n'] == u['prompt_tokens_details']['cached_tokens']
        assert t['predicted_n'] == u['completion_tokens'] and t['draft_kind'] == 'mtp'
        for n, ms, rate in [('prompt_n', 'prompt_ms', 'prompt_per_second'),
                           ('predicted_n', 'predicted_ms', 'predicted_per_second')]:
            assert math.isclose(t[n] * 1000 / t[ms], t[rate], rel_tol=1e-9)
        cold = name.endswith('_initial') or name.endswith('_cold')
        if cold:
            assert t['cache_n'] == 0
        else:
            reuse.append({'request': name, 'expected_cached_tokens': expected_cache[name],
                'actual_cached_tokens': t['cache_n'],
                'exact_latest_prefix_reused': t['cache_n'] == expected_cache[name]})
        assert row['cache_tier'] == ('cold' if cold else 'memory')
        assert row['disk_hits_during_request'] == 0
        for stats in [row['cache_stats_before_request'], row['cache_stats_after_response']]:
            assert stats['disk_hits'] == stats['disk_writes'] == stats['disk_write_failures'] == 0
            assert 'disk_max_bytes' not in stats
        if name.endswith('_cold'):
            assert row['cache_stats_before_request']['resident_bytes'] == 0
        observations.append({'name': name, 'correct': correct, 'bare_json_valid': True,
            'input_tokens': u['prompt_tokens'], 'output_tokens': u['completion_tokens'], 'cached_tokens': t['cache_n'],
            'seconds': r['seconds'], 'first_visible_answer_seconds': r['first_visible_answer_seconds'],
            'cold_prefill_tokens_per_second': t['prompt_per_second'] if cold else None,
            'decode_tokens_per_second': t['predicted_per_second'], 'peak_mlx_gb': t['peak_memory'],
            'cache_stats_after_response': row['cache_stats_after_response']})
    pairs = []
    for key in ('a', 'b'):
        for warm, cold in [(key + '_revisit', key + '_initial'),
                           (key + '_final_revisit', key + '_final_cold')]:
            a, b = by[warm], by[cold]
            x, y = a['result'], b['result']
            same = audits[warm] == audits[cold]
            parity = (x['text'], x['reasoning'], x['usage']['completion_tokens']) == (y['text'], y['reasoning'], y['usage']['completion_tokens'])
            pairs.append({'warm': warm, 'cold': cold, 'full_history_identical': same, 'output_parity': parity})
    log_path = stem + '-server.log'
    log = Path(log_path).read_text()
    assert log.count('Request completed: endpoint=/chat/completions') == 28
    assert log.count('Request started: endpoint=/chat/completions') == 28
    assert 'Application shutdown complete.' in log
    assert all(marker in log for marker in ['FUSED_PREFILL_ACTIVE', 'BORROW_27B_CHECKPOINT_ROW', 'WARM_27B_RESERVE'])
    traces = [json.loads(line.split('APC_MEMORY_DIAGNOSTIC ', 1)[1]) for line in log.splitlines() if 'APC_MEMORY_DIAGNOSTIC ' in line]
    for trace in traces:
        trace['sampled_after_capacity_passed'] = trace['after']['resident_bytes'] + trace['retain_bytes'] <= trace['after']['memory_max_bytes']
        trace['sampled_after_headroom_margin_bytes'] = trace['after']['headroom_bytes'] - trace['required_headroom_bytes']
    failures = [r for r in reuse if not r['exact_latest_prefix_reused']]
    assert len(reuse) == 24
    assert reuse == raw[-1]['reuse_checks']
    quality = all(o['correct'] for o in observations)
    bounded = quality and not failures and all(p['full_history_identical'] and p['output_parity'] for p in pairs)
    assert bounded == raw[-1]['bounded_switching_passed']
    sources = [config_path, status_path, raw_path, suite, log_path, stem + '-memory.jsonl', __file__, prefix + '-review-criteria.json']
    report = {'status': 'completed_bounded_review_retention_failure' if failures else 'completed_bounded_review',
        'acceptance_complete': False, 'diagnostic_only': True,
        'correct_answers': sum(o['correct'] for o in observations), 'bare_json_valid': 28,
        'exact_latest_prefix_hits': len(reuse) - len(failures), 'reuse_checks_count': len(reuse),
        'full_history_retokenization_passed': True, 'frozen_runtime_files_unchanged': True,
        'bounded_switching_passed': bounded, 'sustained_switching_retention_proven': False,
        'observations': observations, 'pairs': pairs, 'reuse_checks': reuse, 'reuse_failures': failures,
        'admission_trace': traces, 'input_audits': audits,
        'limitations': ['Two documents with ten repeated-question followups each; not broad dialogue reasoning quality.',
            'Traced run, one observation per request; not an isolated optimization speedup or P95 measurement.',
            'Short outputs do not establish sustained decode performance.',
            'MLX peak may be cumulative across requests.',
            'After-call free RAM samples do not prove instantaneous admission headroom.',
            'No native-maximum quality acceptance; earlier failures remain.'],
        'source_sha256': {str(p): sha(p) for p in sources}}
    with Path(prefix + '-review.json').open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'correct': report['correct_answers'], 'exact_prefix_hits': report['exact_latest_prefix_hits'],
                     'reuse_failures': failures, 'pairs': pairs, 'observations': observations[-4:]}))


if __name__ == '__main__':
    main()
