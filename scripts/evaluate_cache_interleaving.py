"""Check full-prefix retention across two unrelated short requests.

Use an otherwise idle, memory-only VLM server. This clears its disposable cache.
Cold original -> short A -> short B -> identical original, without truncation.
This is a bounded retention/quality check, not a throughput benchmark.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from evaluate_cache_branches import require_memory_only_cache
from evaluate_sessions import reset
from qwen_booster.grading import grade_answer


def screen(case, request, clear, stats, save, minimum_input_tokens):
    if case.get('temperature', 0) != 0:
        raise ValueError('Use a greedy fixture for exact cold/warm parity')
    rule = case.get('grading')
    if rule is None:
        expected = case.get('expected_substrings', [])
        if len(expected) != 1:
            raise ValueError('Need independent grading or a single exact retrieval answer')
        rule = {'kind': 'exact_text', 'expected': expected[0]}
    case = {**case, 'grading': rule}
    unrelated = [
        {'messages': [{'role': 'user', 'content':
                       '独立测试请求。请只输出下列标记，不解释，不加引号，不添加其他字符：' + marker}],
         'temperature': 0, 'enable_thinking': False, 'max_tokens': 64,
         'grading': {'kind': 'exact_text', 'expected': marker}}
        for marker in ('BIRCH-724', 'MAPLE-519')
    ]
    require_memory_only_cache(stats())
    clear()
    measurements = []
    for name, fixture in [('original_cold', case), ('unrelated_a', unrelated[0]),
                          ('unrelated_b', unrelated[1]), ('original_after_unrelated', case)]:
        result = request(fixture)
        after = stats()
        require_memory_only_cache(after)
        row = {
            'type': 'measurement', 'name': name, 'result': result,
            'case_sha256': hashlib.sha256(json.dumps(fixture, ensure_ascii=False,
                                                    sort_keys=True).encode()).hexdigest(),
            'independent_grade': grade_answer(result.get('text', ''), fixture['grading']) is True,
            'cache_stats_after_response': after,
        }
        save(row)
        measurements.append(row)
        if not (result.get('complete') is True and result.get('finish_reason') == 'stop'
                and result.get('text', '').strip() and not result.get('server_errors')):
            raise RuntimeError('Incomplete response; retain partial evidence')
    cold, warm = measurements[0]['result'], measurements[-1]['result']
    count = cold['usage']['prompt_tokens']
    checks = {
        'all_answers_correct': all(row['independent_grade'] for row in measurements),
        'full_input_floor': count >= minimum_input_tokens,
        'identical_full_fixture': measurements[0]['case_sha256'] == measurements[-1]['case_sha256'],
        'input_counts_equal': count == warm['usage']['prompt_tokens'],
        'output_counts_equal': cold['usage']['completion_tokens'] == warm['usage']['completion_tokens'],
        'body_and_reasoning_equal': (cold['text'], cold['reasoning']) == (warm['text'], warm['reasoning']),
        'cold_cache_empty': cold['server_timings']['cache_n'] == 0,
        'original_prefix_retained': warm['server_timings']['cache_n'] >= count - 1,
        'unrelated_prefixes_distinct': all(row['result']['server_timings']['cache_n'] == 0
                                         for row in measurements[1:3]),
    }
    summary = {'type': 'summary', 'bounded_interleaving_passed': all(checks.values()),
               'checks': checks, 'actual_input_tokens': count, 'acceptance_complete': False}
    save(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-input-tokens', type=int, default=204800)
    args = parser.parse_args()
    raw = Path(args.suite).read_bytes()
    matches = [json.loads(line) for line in raw.splitlines()
               if line.strip() and json.loads(line)['id'] == args.case_id]
    if len(matches) != 1:
        parser.error('case-id must identify one complete fixture')

    def stats():
        with urllib.request.urlopen(args.endpoint.rstrip('/') + '/cache/stats', timeout=30) as response:
            return json.load(response)

    with Path(args.output).open('x') as output:
        def save(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()

        save({'type': 'metadata', 'suite_sha256': hashlib.sha256(raw).hexdigest(),
              'case_id': args.case_id, 'model': args.model, 'backend': 'vlm',
              'minimum_input_tokens': args.minimum_input_tokens, 'scope': __doc__})
        result = screen(matches[0], lambda fixture: evaluate(args.endpoint, fixture, 7200,
                        model=args.model, include_usage=True), lambda: reset(args.endpoint),
                        stats, save, args.minimum_input_tokens)
    print(json.dumps(result))
    raise SystemExit(0 if result['bounded_interleaving_passed'] else 1)


if __name__ == '__main__':
    main()
