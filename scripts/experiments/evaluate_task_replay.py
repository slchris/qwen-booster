"""Evaluate one full quality fixture cold, then repeat with its exact prefix cached.

This records one paired observation, not a repeated speed benchmark. All input and
sampling settings come from the same fixture, including thinking/output budgets.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_api import evaluate
from evaluate_sessions import reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['native', 'vlm'], required=True)
    args = parser.parse_args()
    raw = Path(args.suite).read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    matches = [case for case in cases if case['id'] == args.case_id]
    if len(matches) != 1:
        parser.error('case-id must match exactly one fixture')
    case = matches[0]
    cache_key = 'cache_n' if args.backend == 'vlm' else 'cached_tokens'
    results = {}
    with Path(args.output).open('x') as output:
        def save(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()

        save({'type': 'metadata', 'scope': __doc__, 'case_id': args.case_id,
              'suite_sha256': hashlib.sha256(raw).hexdigest(), 'model': args.model,
              'backend': args.backend, 'prompt_tokens': case.get('prompt_tokens')})
        reset(args.endpoint)
        for regime in ('cold', 'warm'):
            result = evaluate(args.endpoint, case, 7200, model=args.model,
                              include_usage=args.backend == 'vlm')
            results[regime] = result
            cached = (result.get('server_timings') or {}).get(cache_key)
            save({'type': 'measurement', 'regime': regime, 'result': result})
            print(json.dumps({'regime': regime, 'quality_passed': result['task_check_passed'],
                              'cached_tokens': cached, 'seconds': result['seconds']}), flush=True)
            if not result['complete'] or result['finish_reason'] != 'stop':
                raise RuntimeError('incomplete task response; inspect saved output')
        cold, warm = results['cold'], results['warm']
        save({'type': 'comparison', 'quality_passed': all(r['task_check_passed'] for r in results.values()),
              'answer_parity': cold['text'] == warm['text'],
              'reasoning_parity': cold['reasoning'] == warm['reasoning'],
              'cold_cache_empty': (cold.get('server_timings') or {}).get(cache_key) == 0,
              'warm_cache_reused': (warm.get('server_timings') or {}).get(cache_key, 0) > 0})


if __name__ == '__main__':
    main()
