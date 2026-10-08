"""Regrade the bounded HTTP/MTP/fused-prefill experiment without loading MLX."""
import hashlib
import json
from pathlib import Path
import runpy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review():
    prefix = 'results/fused-http-mtp-v1'
    config_path = Path(prefix + '-config.json')
    config = json.loads(config_path.read_text())
    status_path = Path(prefix + '-status.json')
    assert json.loads(status_path.read_text())['stage'] == 'observations_finished_pending_review'
    for path, expected in config['source_sha256'].items():
        assert digest(path) == expected, f'Changed experiment source: {path}'
    baseline = config['commands']['baseline']
    assert config['commands']['fused'] == baseline + ['--fused-prefill']
    assert baseline[baseline.index('--context') + 1] == '262144'
    assert '--draft' in baseline and '--release-finished-mtp' in baseline
    grading = runpy.run_path('src/qwen_booster/grading.py')
    sources = [config_path, status_path]
    observations, comparisons = [], []
    for tier in ['short', '200k']:
        suite = Path(f'results/reasoning-diverse-{tier}-suite-v1.jsonl')
        sources.append(suite)
        cases = [json.loads(line) for line in suite.read_text().splitlines()]
        assert len(cases) == 3
        for mode in ['baseline', 'fused']:
            log = Path(f'{prefix}-{tier}-{mode}-server.log')
            sources.append(log)
            if mode == 'fused':
                assert 'FUSED_PREFILL_ACTIVE' in log.read_text()
        for case in cases:
            pairs = {}
            for mode in ['baseline', 'fused']:
                raw = Path(f'{prefix}-{tier}-{mode}-{case["id"]}.jsonl')
                sources.append(raw)
                rows = [json.loads(line) for line in raw.read_text().splitlines()]
                assert len(rows) == 4
                meta = rows[0]
                assert meta['type'] == 'metadata' and meta['case_id'] == case['id']
                assert meta['suite_sha256'] == digest(suite) and meta['backend'] == 'vlm'
                assert meta['model'] == baseline[baseline.index('--model') + 1]
                pair = {}
                for row, regime in zip(rows[1:3], ['cold', 'warm']):
                    assert row['type'] == 'measurement' and row['regime'] == regime
                    result = row['result']
                    timing, usage = result['server_timings'], result['usage']
                    assert result['complete'] and result['finish_reason'] == 'stop'
                    assert not result['server_errors']
                    n = usage['prompt_tokens']
                    assert n == case['prompt_tokens'] and n + case['max_tokens'] <= 262144
                    if tier == '200k':
                        assert n >= 204800
                    expected_cache = 0 if regime == 'cold' else n - 1
                    assert timing['cache_n'] == usage['prompt_tokens_details']['cached_tokens'] == expected_cache
                    assert timing['prompt_n'] + timing['cache_n'] == n
                    assert timing['predicted_n'] == usage['completion_tokens'] > 0
                    assert timing['draft_kind'] == 'mtp'
                    correct = grading['grade_answer'](result['text'], case['grading'])
                    formatted = grading['grade_json_format'](result['text'], case['grading'])
                    assert correct == result['task_check_passed']
                    assert formatted == result['bare_json_format_passed']
                    observations.append(dict(tier=tier, mode=mode, case_id=case['id'],
                        regime=regime, input_tokens=n, output_tokens=usage['completion_tokens'],
                        cached_tokens=expected_cache, content_correct=correct,
                        bare_json_correct=formatted, seconds=result['seconds'],
                        prefill_tokens_per_second=timing['prompt_per_second'] if regime == 'cold' else None,
                        decode_tokens_per_second=timing['predicted_per_second'],
                        process_peak_mlx_gb=timing['peak_memory']))
                    pair[regime] = result
                actual = dict(type='comparison', quality_passed=all(r['task_check_passed'] for r in pair.values()),
                    answer_parity=pair['cold']['text'] == pair['warm']['text'],
                    reasoning_parity=pair['cold']['reasoning'] == pair['warm']['reasoning'],
                    cold_cache_empty=True, warm_cache_reused=True)
                assert rows[-1] == actual
                pairs[mode] = pair
            for regime in ['cold', 'warm']:
                b, f = pairs['baseline'][regime], pairs['fused'][regime]
                bt, ft = b['server_timings'], f['server_timings']
                comparisons.append(dict(tier=tier, case_id=case['id'], regime=regime,
                    answer_parity=b['text'] == f['text'], reasoning_parity=b['reasoning'] == f['reasoning'],
                    output_count_parity=bt['predicted_n'] == ft['predicted_n'],
                    prefill_gain_percent=(ft['prompt_per_second'] / bt['prompt_per_second'] - 1) * 100 if regime == 'cold' else None,
                    decode_gain_percent=(ft['predicted_per_second'] / bt['predicted_per_second'] - 1) * 100,
                    total_latency_reduction_percent=(1 - f['seconds'] / b['seconds']) * 100,
                    peak_memory_reduction_percent=(1 - ft['peak_memory'] / bt['peak_memory']) * 100))
    return dict(status='completed_bounded_integration_review', acceptance_complete=False,
        runtime_default_enabled=False, reviewed_requests=len(observations),
        content_correct=sum(r['content_correct'] for r in observations),
        bare_json_correct=sum(r['bare_json_correct'] for r in observations),
        source_sha256={str(p): digest(p) for p in sources}, observations=observations,
        optimization_comparisons=comparisons,
        limitations=['Three independent synthetic tasks at two lengths, one cold/warm pair per mode; no broad quality acceptance.',
                    'Baseline precedes fused within each tier; order is not balanced. No repeated HTTP performance confidence or P95.',
                    'Both modes already use MTP and APC; differences isolate adding fused prefill under these conditions, not all acceleration versus stock.',
                    'Regrades original gold answers, not an independent oracle audit. HTTP text equality is not exact token-ID parity.',
                    'Native boundary, multi-turn and multi-document behavior with this combined configuration remain unverified.',
                    'Memory is process-cumulative MLX peak, not whole-system RAM or isolated per-request peak.'])


if __name__ == '__main__':
    report = review()
    target = Path('results/fused-http-mtp-v1-review.json')
    with target.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ['status', 'reviewed_requests', 'content_correct', 'bare_json_correct', 'acceptance_complete']}))
