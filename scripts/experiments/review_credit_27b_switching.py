"""Independent full-history review of the completed RAM-only combined 27B run."""
import copy
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prefix = 'results/credit-27b-switching-200k-v1'
    stem = prefix + '-27b-combined'
    config_path = prefix + '-config.json'
    config = json.loads(Path(config_path).read_text())
    assert all(sha(p) == h for p, h in config['source_sha256'].items())
    status_path = prefix + '-status.json'
    assert json.loads(Path(status_path).read_text())['stage'] == 'observations_finished_pending_review'
    raw_path = stem + '.jsonl'
    raw = [json.loads(x) for x in Path(raw_path).read_text().splitlines()]
    assert len(raw) == 10 and raw[0]['type'] == 'metadata' and raw[-1]['type'] == 'summary'
    names = ['a_initial', 'b_initial', 'a_revisit', 'b_revisit',
             'a_followup', 'b_followup', 'a_followup_cold', 'b_followup_cold']
    data = raw[1:-1]
    assert [r['name'] for r in data] == names
    by = {r['name']: r for r in data}
    suite = 'results/document-switching-200k-suite-v1.jsonl'
    assert sha(suite) == raw[0]['suite_sha256']
    cases = [json.loads(x) for x in Path(suite).read_text().splitlines()]
    assert [c['id'] for c in cases] == config['case_ids']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['runs'][0]['model'], local_files_only=True)
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
            audits[key + '_' + kind] = {'input_tokens': len(tokens),
                'token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                'messages_sha256': hashlib.sha256(json.dumps(item['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    observations = []
    for row in data:
        name, r = row['name'], row['result']
        key = name[0]
        audit = audits[key + ('_followup' if 'followup' in name else '_initial')]
        assert json.loads(r['text']) == cases[0 if key == 'a' else 1]['grading']['expected']
        assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
        assert row['messages_sha256'] == audit['messages_sha256']
        assert row['input_audit']['prompt_token_sha256'] == audit['token_sha256']
        u, t = r['usage'], r['server_timings']
        assert u['prompt_tokens'] == audit['input_tokens']
        assert u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
        assert t['prompt_n'] + t['cache_n'] == u['prompt_tokens']
        assert t['cache_n'] == u['prompt_tokens_details']['cached_tokens']
        assert t['predicted_n'] == u['completion_tokens'] and t['draft_kind'] == 'mtp'
        for n, ms, rate in [('prompt_n', 'prompt_ms', 'prompt_per_second'),
                             ('predicted_n', 'predicted_ms', 'predicted_per_second')]:
            assert math.isclose(t[n] * 1000 / t[ms], t[rate], rel_tol=1e-9)
        cold = name.endswith('initial') or name.endswith('_cold')
        assert t['cache_n'] == (0 if cold else audits[key + '_initial']['input_tokens'] - 1)
        assert row['cache_tier'] == ('cold' if cold else 'memory')
        assert row['disk_hits_during_request'] == 0
        for stats in [row['cache_stats_before_request'], row['cache_stats_after_response']]:
            assert stats['disk_hits'] == stats['disk_writes'] == stats['disk_write_failures'] == 0
            assert 'disk_max_bytes' not in stats
        observations.append({'name': name, 'correct': True, 'input_tokens': u['prompt_tokens'],
            'output_tokens': u['completion_tokens'], 'cached_tokens': t['cache_n'],
            'seconds': r['seconds'], 'first_visible_answer_seconds': r['first_visible_answer_seconds'],
            'cold_prefill_tokens_per_second': t['prompt_per_second'] if cold else None,
            'decode_tokens_per_second': t['predicted_per_second'], 'peak_mlx_gb': t['peak_memory'],
            'cache_stats_after_response': row['cache_stats_after_response']})
    pairs = []
    for key in ('a', 'b'):
        for warm, cold in [(key + '_revisit', key + '_initial'), (key + '_followup', key + '_followup_cold')]:
            a, b = by[warm], by[cold]
            assert a['messages_sha256'] == b['messages_sha256']
            x, y = a['result'], b['result']
            assert (x['text'], x['reasoning'], x['usage']['completion_tokens']) == (y['text'], y['reasoning'], y['usage']['completion_tokens'])
            pairs.append({'warm': warm, 'cold': cold, 'full_history_and_output_parity': True})
    log_path = stem + '-server.log'
    log = Path(log_path).read_text()
    assert log.count('Request completed: endpoint=/chat/completions') == 8
    assert log.count('Request started: endpoint=/chat/completions') == 8
    assert 'Application shutdown complete.' in log
    assert all(marker in log for marker in ['FUSED_PREFILL_ACTIVE', 'BORROW_27B_CHECKPOINT_ROW', 'WARM_27B_RESERVE'])
    sources = [config_path, status_path, raw_path, suite, log_path, stem + '-memory.jsonl', __file__]
    report = {'status': 'completed_bounded_review', 'acceptance_complete': False,
        'correct_answers': 8, 'bare_json_valid': 8, 'exact_original_prefix_hits': 4,
        'full_history_retokenization_passed': True, 'frozen_runtime_files_unchanged': True,
        'observations': observations, 'pairs': pairs, 'input_audits': audits,
        'sustained_switching_retention_proven': False,
        'retention_issue': 'B followup used original prefix but ended with one memory eviction and one skipped snapshot store. Further alternating followups are untested.',
        'limitations': ['Two documents, repeated-question followups; one observation per request.',
            'Cold controls reset cache in the same server process; not fresh-process controls.',
            'MLX peak may be cumulative across requests; not an isolated request allocation estimate.',
            'No broad quality, sustained decode, native-maximum quality, or isolated optimization speedup claim.'],
        'source_sha256': {p: sha(p) for p in sources}}
    with Path(prefix + '-review.json').open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'correct': 8, 'exact_prefix_hits': 4, 'parity_pairs': 4,
                      'retention_issue_remains': True}))


if __name__ == '__main__':
    main()
