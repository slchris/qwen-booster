"""Review the bounded admission candidate's short integration gate."""
import hashlib
import json
import math
from pathlib import Path
from qwen_booster.grading import grade_answer, grade_json_format


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    prefix = 'results/recheck-27b-http-short-v2'
    config_path = prefix + '-config.json'
    config = json.loads(Path(config_path).read_text())
    assert all(sha(p) == h for p, h in config['source_sha256'].items())
    status = prefix + '-status.json'
    assert json.loads(Path(status).read_text())['stage'] == 'observations_finished_pending_review'
    suite = Path('results/reasoning-diverse-short-suite-v1.jsonl')
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    observations, pairs, sources, by = [], [], [config_path, status, str(suite), __file__], {}
    for mode in ('baseline', 'borrowed'):
        stem = prefix + '-short-' + mode
        log_path = stem + '-server.log'
        log = Path(log_path).read_text()
        assert 'Application shutdown complete.' in log
        assert log.count('Request completed: endpoint=/chat/completions') == 6
        if mode == 'borrowed':
            assert 'APC_RECHECK_27B_INSTALLED' in log
        else:
            assert 'APC_RECHECK_27B_INSTALLED' not in log
        sources += [log_path, stem + '-memory.jsonl']
        for case in cases:
            p = stem + '-' + case['id'] + '.jsonl'
            sources.append(p)
            rows = [json.loads(line) for line in Path(p).read_text().splitlines()]
            assert len(rows) == 4
            cold, warm = [row['result'] for row in rows[1:3]]
            for regime, r in [('cold', cold), ('warm', warm)]:
                assert r['complete'] and r['finish_reason'] == 'stop' and not r['server_errors']
                assert grade_answer(r['text'], case['grading']) is True
                assert grade_json_format(r['text'], case['grading']) is not False
                t, u = r['server_timings'], r['usage']
                assert t['prompt_n'] + t['cache_n'] == u['prompt_tokens']
                assert t['cache_n'] == (0 if regime == 'cold' else u['prompt_tokens'] - 1)
                assert t['cache_n'] == u['prompt_tokens_details']['cached_tokens']
                assert t['predicted_n'] == u['completion_tokens']
                for n, ms, rate in [('prompt_n', 'prompt_ms', 'prompt_per_second'), ('predicted_n', 'predicted_ms', 'predicted_per_second')]:
                    assert math.isclose(t[n] * 1000 / t[ms], t[rate], rel_tol=1e-9)
                observations.append({'mode': mode, 'case': case['id'], 'regime': regime, 'seconds': r['seconds'],
                    'first_visible_answer_seconds': r['first_visible_answer_seconds'], 'timings': t, 'usage': u})
            def output(r): return (r['text'], r.get('reasoning'), r['usage']['completion_tokens'])
            assert output(cold) == output(warm)
            by[mode, case['id']] = cold
            pairs.append({'mode': mode, 'case': case['id'], 'output_parity': True})
    cross = []
    for case in cases:
        a, b = by['baseline', case['id']], by['borrowed', case['id']]
        cross.append({'case': case['id'], 'body_equal': a['text'] == b['text'],
                      'reasoning_equal': a.get('reasoning') == b.get('reasoning'),
                      'output_count_equal': a['usage']['completion_tokens'] == b['usage']['completion_tokens']})
    log = Path(prefix + '-short-borrowed-server.log').read_text()
    count = log.count('APC_RECHECK_REJECTED')
    report = {'status': 'completed_short_integration_review', 'acceptance_complete': False,
        'correct_answers': 12, 'cold_warm_parity_pairs': 6, 'candidate_installed': True,
        'recheck_rejections': count, 'recheck_samples': log.count('APC_RECHECK_SAMPLE'),
        'recovery_exercised': count > 0, 'frozen_sources_unchanged': True,
        'observations': observations, 'pairs': pairs, 'cross_configuration': cross,
        'limitations': ['Short gate only; no 200K retention acceptance.',
            'Single observation per condition; no causal speedup claim.',
            'No rejection events means recovery branch was not exercised.'],
        'source_sha256': {p: sha(p) for p in sources}}
    with Path(prefix + '-review.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'correct': 12, 'pairs': 6, 'rejections': count, 'cross': cross}))


if __name__ == '__main__': main()
