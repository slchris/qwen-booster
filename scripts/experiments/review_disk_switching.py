"""Independently audit completed two-document RAM+SSD results and full histories."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prefix = args.prefix
    config_path = prefix + '-config.json'
    config = json.loads(Path(config_path).read_text())
    assert all(digest(p) == h for p, h in config['source_sha256'].items())
    raw_path = prefix + '-responses.jsonl'
    raw = [json.loads(x) for x in Path(raw_path).read_text().splitlines()]
    assert len(raw) == 10 and raw[0]['type'] == 'metadata' and raw[-1]['type'] == 'summary'
    assert raw[0]['suite_sha256'] == digest(args.suite)
    data = raw[1:-1]
    names = ['a_initial', 'b_initial', 'a_revisit', 'b_revisit',
             'a_followup', 'b_followup', 'a_followup_cold', 'b_followup_cold']
    assert [r['name'] for r in data] == names
    by = {r['name']: r for r in data}
    cases = [json.loads(x) for x in Path(args.suite).read_text().splitlines()]
    assert [c['id'] for c in cases] == config['case_ids']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['run']['model'], local_files_only=True)
    audits = {}
    for key, case in zip(('a', 'b'), cases):
        follow = copy.deepcopy(case)
        initial = by[key + '_initial']['result']
        assistant = {'role': 'assistant', 'content': initial['text']}
        if initial['reasoning']:
            assistant['reasoning_content'] = initial['reasoning']
        follow['messages'].extend([assistant, {'role': 'user',
            'content': '请重新回答最初的用户任务，严格遵守最初要求的输出格式。'}])
        for kind, item in (('initial', case), ('followup', follow)):
            tokens = tokenizer.apply_chat_template(item['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=False, **{
                    k: item[k] for k in ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if k in item})
            assert 204800 <= len(tokens) and len(tokens) + item['max_tokens'] <= 262144
            audits[key + '_' + kind] = {
                'input_tokens': len(tokens), 'token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                'messages_sha256': hashlib.sha256(json.dumps(item['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    observations = []
    for row in data:
        name, result = row['name'], row['result']
        key = name[0]
        audit = audits[key + ('_followup' if 'followup' in name else '_initial')]
        expected = cases[0 if key == 'a' else 1]['grading']['expected']
        assert json.loads(result['text']) == expected
        assert result['complete'] and result['finish_reason'] == 'stop' and not result['server_errors']
        assert row['messages_sha256'] == audit['messages_sha256']
        assert row['input_audit']['prompt_token_sha256'] == audit['token_sha256']
        usage, timings = result['usage'], result['server_timings']
        assert usage['prompt_tokens'] == audit['input_tokens']
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        assert timings['prompt_n'] + timings['cache_n'] == usage['prompt_tokens']
        assert timings['cache_n'] == usage['prompt_tokens_details']['cached_tokens']
        assert timings['predicted_n'] == usage['completion_tokens']
        for n, ms, rate in (('prompt_n', 'prompt_ms', 'prompt_per_second'),
                            ('predicted_n', 'predicted_ms', 'predicted_per_second')):
            assert math.isclose(timings[n] * 1000 / timings[ms], timings[rate], rel_tol=1e-9)
        before, after = row['cache_stats_before_request'], row['cache_stats_after_response']
        assert before['observed_server_pid'] == after['observed_server_pid']
        delta = after['disk_hits'] - before['disk_hits']
        assert delta == row['disk_hits_during_request'] >= 0
        cold = name.endswith('initial') or name.endswith('_cold')
        assert timings['cache_n'] == (0 if cold else audits[key + '_initial']['input_tokens'] - 1)
        assert row['cache_tier'] == ('cold' if cold else ('disk' if delta else 'memory'))
        if cold:
            assert delta == 0
        if name.endswith('_cold'):
            assert 'disk_max_bytes' not in before and 'disk_max_bytes' not in after
        assert after['disk_write_failures'] == 0
        if 'disk_max_bytes' in after:
            assert after['disk_bytes'] <= after['disk_max_bytes'] == config['disk_budget_mb'] * 1000000
        observations.append({'name': name, 'input_tokens': usage['prompt_tokens'],
            'output_tokens': usage['completion_tokens'], 'cached_tokens': timings['cache_n'],
            'tier': row['cache_tier'], 'disk_hit_delta': delta, 'seconds': result['seconds'],
            'first_visible_answer_seconds': result['first_visible_answer_seconds'],
            'cold_prefill_tokens_per_second': timings['prompt_per_second'] if cold else None,
            'decode_tokens_per_second': timings['predicted_per_second'],
            'peak_mlx_gb': timings['peak_memory'], 'server_pid': after['observed_server_pid']})
    pairs = []
    for key in ('a', 'b'):
        for warm, cold in ((key + '_revisit', key + '_initial'), (key + '_followup', key + '_followup_cold')):
            a, b = by[warm], by[cold]
            assert a['messages_sha256'] == b['messages_sha256']
            x, y = a['result'], b['result']
            assert (x['text'], x['reasoning']) == (y['text'], y['reasoning'])
            assert x['usage']['completion_tokens'] == y['usage']['completion_tokens']
            pairs.append({'warm': warm, 'cold': cold, 'complete_history_and_output_parity': True})
    lifecycle_path = prefix + '-lifecycle.jsonl'
    lifecycle = [json.loads(x) for x in Path(lifecycle_path).read_text().splitlines()]
    ready = [r for r in lifecycle if r['stage'] == 'ready']
    assert len(ready) == 3 and len({r['pid'] for r in ready}) == 3
    assert [r['disk_enabled'] for r in ready] == [True, False, False]
    assert [o['server_pid'] for o in observations] == [ready[0]['pid']] * 6 + [ready[1]['pid'], ready[2]['pid']]
    sources = [config_path, raw_path, lifecycle_path, args.suite]
    for i, count in ((1, 6), (2, 1), (3, 1)):
        log = prefix + f'-server-{i}.log'
        text = Path(log).read_text()
        assert text.count('Request completed: endpoint=/chat/completions') == count
        assert text.count('Request started: endpoint=/chat/completions') == count
        assert 'Application shutdown complete.' in text
        sources.extend([log, prefix + f'-server-{i}-memory.jsonl'])
    assert sum(o['disk_hit_delta'] for o in observations) >= 1
    report = {'status': 'completed_bounded_review', 'acceptance_complete': False,
        'bounded_disk_switching_passed': True, 'correct_answers': 8, 'bare_json_valid': 8,
        'full_history_retokenization_passed': True, 'frozen_runtime_files_unchanged': True,
        'observations': observations, 'pairs': pairs, 'server_lifecycle': ready,
        'source_sha256': {p: digest(p) for p in sources},
        'scope': 'Two full documents with repeated-question followups; one observation per request. '
                 'SSD restore is part of request latency, model startup is separate. '
                 'CPU fixture preparation overlapped first prefill. Not stable speedup, broad quality, '
                 'many-document capacity, native-maximum acceptance, or cross-model equivalence.'}
    with Path(args.output).open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'bounded_passed': True, 'correct': 8, 'pairs': 4,
                      'disk_restores': sum(o['disk_hit_delta'] for o in observations)}))


if __name__ == '__main__':
    main()
