"""Regrade full-input conversation evidence without loading the model."""
import hashlib
import json
from pathlib import Path
import re
import statistics

PREFIX = 'results/fused-mtp-conversation-200k-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def review():
    config = json.loads(Path(PREFIX + '-config.json').read_text())
    assert json.loads(Path(PREFIX + '-status.json').read_text())['stage'] == 'finished_pending_review'
    for name, expected in config['source_sha256'].items():
        assert sha(name) == expected, name
    command = config['commands']['candidate']
    assert all(flag in command for flag in ['--draft', '--release-finished-mtp', '--fused-prefill'])
    assert command[command.index('--context') + 1] == '262144'
    log = Path(PREFIX + '-candidate-server.log')
    assert 'FUSED_PREFILL_ACTIVE' in log.read_text()
    suite = Path(config['suite'])
    fixture = next(c for c in map(json.loads, suite.read_text().splitlines()) if c['fact_position_fraction'] == 0.95)
    records = list(re.finditer(r'记录 (\d{6}): 项目编号 (\d{8});', fixture['messages'][0]['content']))
    questions = [records[int((len(records)-1)*f)] for f in (.05, .5, .95)]
    rows = [json.loads(x) for x in Path(PREFIX + '-200k.jsonl').read_text().splitlines()]
    assert len(rows) == 12 and rows[0]['type'] == 'metadata'
    assert rows[0]['suite_sha256'] == sha(suite) and rows[0]['turns'] == 10
    assert rows[0]['fixture_id'] == fixture['id']
    measurements = rows[1:11]
    observations = []
    for i, row in enumerate(measurements):
        assert row['type'] == 'measurement' and row['turn'] == i
        expected = fixture['expected_substrings'][0] if not i else questions[(i-1)%3][2]
        if i:
            assert row['question']['record_id'] == questions[(i-1)%3][1]
        assert row['expected'] == expected
        r = row['result']; t = r['server_timings']; u = r['usage']
        assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
        assert r['task_check_passed'] == (r['text'].strip() == expected)
        assert 204800 <= u['prompt_tokens'] and u['prompt_tokens'] + 64 <= 262144
        cache = 0 if not i else measurements[i-1]['result']['usage']['prompt_tokens']-1
        assert t['cache_n'] == u['prompt_tokens_details']['cached_tokens'] == cache
        assert t['cache_n'] + t['prompt_n'] == u['prompt_tokens']
        assert t['draft_kind'] == 'mtp'
        observations.append(dict(turn=i, content_correct=r['task_check_passed'], input_tokens=u['prompt_tokens'],
            cached_tokens=cache, seconds=r['seconds'], first_answer_seconds=r['first_visible_answer_seconds'],
            decode_tokens_per_second=t['predicted_per_second'], process_peak_mlx_gb=t['peak_memory']))
    last = measurements[-1]['result']; replay = rows[-1]; cold = replay['result']
    assert replay['type'] == 'cold_replay' and replay['turn'] == 9
    assert cold['complete'] and cold['finish_reason'] == 'stop' and not cold['server_errors']
    assert cold['server_timings']['cache_n'] == 0
    assert cold['usage']['prompt_tokens'] == last['usage']['prompt_tokens']
    parity = cold['text'] == last['text'] and cold['reasoning'] == last['reasoning']
    assert replay['parity_passed'] == parity
    assert cold['task_check_passed'] == (cold['text'].strip() == questions[2][2])
    short_path = Path(PREFIX + '-short.jsonl')
    short = [json.loads(x) for x in short_path.read_text().splitlines()]
    assert len(short) == 4
    for i, row in enumerate(short):
        assert row['turn'] == i and row['parity_passed'] and row['reuse_check_passed']
        a, b = row['warm'], row['cold']
        assert a['complete'] and b['complete'] and a['finish_reason'] == b['finish_reason'] == 'stop'
        assert a['text'] == b['text'] and a['reasoning'] == b['reasoning']
        assert b['server_timings']['cache_n'] == 0
        if i:
            assert a['server_timings']['cache_n'] > 0
    paths = [suite, short_path, log, Path(PREFIX + '-200k.jsonl'), Path(PREFIX + '-config.json'), Path(PREFIX + '-status.json')]
    return dict(status='completed_bounded_conversation_review', acceptance_complete=False, runtime_default_enabled=False,
        source_sha256={str(p):sha(p) for p in paths}, observed_turns=10,
        content_correct=sum(r['content_correct'] for r in observations), warm_reuse_turns=9,
        short_parity_pairs=4, turns=observations,
        warm_first_answer_median_seconds=statistics.median(r['first_answer_seconds'] for r in observations[1:]),
        final_cold_replay=dict(content_correct=cold['task_check_passed'], output_parity=parity,
            seconds=cold['seconds'], input_tokens=cold['usage']['prompt_tokens'], cached_tokens=0,
            prefill_tokens_per_second=cold['server_timings']['prompt_per_second']),
        limitations=['Four cycled retrieval questions in one synthetic document, not ten independent tasks.',
                    'Only final history has a paired cold replay; no P95 or controlled historical optimization ratio.',
                    'No native-boundary, broad reasoning, multi-document or concurrent-user acceptance.',
                    'Short checks establish protocol parity, not independent semantic grading; 9–12 output tokens are not a stable decode benchmark.'])

if __name__ == '__main__':
    r=review()
    with Path(PREFIX + '-review.json').open('x') as f:
        json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ['status','content_correct','warm_reuse_turns','warm_first_answer_median_seconds','final_cold_replay']}))
