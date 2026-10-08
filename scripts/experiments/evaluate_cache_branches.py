"""Screen singleton APC restoration against cold references and sibling branches.

Run on an otherwise idle server, once per baseline/candidate configuration.
This clears the server cache. It measures bounded greedy parity, not broad quality
or a controlled speedup. All fixture content is retained without truncation.
"""

import argparse
import copy
import hashlib
import json
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from evaluate_sessions import reset


def require_memory_only_cache(stats, backend='vlm'):
    # The backend's reset endpoint does not remove persisted snapshots.
    if backend == 'native':
        if not isinstance(stats, dict) or not {'entries', 'bytes', 'budget_bytes'} <= stats.keys():
            raise ValueError('expected native prefix-store stats; cannot verify backend')
        return
    if backend != 'vlm':
        raise ValueError('unknown backend')
    if not isinstance(stats, dict) or 'disk_hits' not in stats:
        raise ValueError('expected VLM cache stats; cannot verify cold-reference isolation')
    if 'disk_max_bytes' in stats:
        raise ValueError('disk cache is enabled: memory-only reset cannot create cold references; '
                         'use fresh disk stores and server restarts for disk branch testing')


def screen(case, request, clear, save, *, branch_mode='edit_last_user', backend='vlm'):
    if backend not in ('native', 'vlm'):
        raise ValueError('unknown backend')
    if case.get('temperature', 0) != 0:
        raise ValueError('cache parity screen requires greedy temperature=0')
    if branch_mode not in ('edit_last_user', 'followup'):
        raise ValueError('unknown branch mode')
    if case['messages'][-1]['role'] != 'user' or not isinstance(
            case['messages'][-1]['content'], str):
        raise ValueError('fixture must end with a plain text user message')
    rows = {}

    def run(name, fixture, cold=False):
        if cold:
            clear()
        result = request(fixture)
        rows[name] = result
        save({'type': 'measurement', 'name': name,
              'messages_sha256': hashlib.sha256(json.dumps(
                  fixture['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              'result': result})
        if (not result['complete'] or result['finish_reason'] != 'stop'
                or not result.get('text', '').strip() or result.get('server_errors')):
            raise RuntimeError('incomplete task response; evidence retained')

    run('base_cold', case, cold=True)
    branches = {}
    for key, suffix in [('a', '\n请严格遵守上述输出格式，完成上述任务。'),
                        ('b', '\n请核对上述要求后，完成上述任务。')]:
        branch = copy.deepcopy(case)
        if branch_mode == 'edit_last_user':
            branch['messages'][-1]['content'] += suffix
        else:
            answer = {'role': 'assistant', 'content': rows['base_cold']['text']}
            if rows['base_cold'].get('reasoning'):
                answer['reasoning_content'] = rows['base_cold']['reasoning']
            branch['messages'].extend([
                answer, {'role': 'user', 'content': '请重新回答最初的用户任务。' + suffix}])
        branches[key] = branch
    run('base_warm', case)
    run('branch_a_warm', branches['a'])
    run('base_after_branch', case)
    run('branch_b_warm', branches['b'])
    run('branch_a_cold', branches['a'], cold=True)
    run('branch_b_cold', branches['b'], cold=True)
    def counts(result):
        timing = result.get('server_timings') or {}
        usage = result.get('usage') or {}
        if backend == 'native':
            return timing.get('prompt_tokens'), timing.get('generation_tokens'), timing.get('cached_tokens')
        return usage.get('prompt_tokens'), usage.get('completion_tokens'), timing.get('cache_n')

    prompt_tokens = counts(rows['base_cold'])[0]
    minimum_reuse = max(1, int(prompt_tokens * .95)) if prompt_tokens else None
    comparisons = []
    for warm_name, cold_name in [('base_warm', 'base_cold'),
                                  ('base_after_branch', 'base_cold'),
                                  ('branch_a_warm', 'branch_a_cold'),
                                  ('branch_b_warm', 'branch_b_cold')]:
        warm, cold = rows[warm_name], rows[cold_name]
        warm_prompt, warm_completion, cached = counts(warm)
        cold_prompt, cold_completion, cold_cached = counts(cold)
        parity = all(warm.get(k) == cold.get(k) for k in ('text', 'reasoning'))
        counts_match = warm_completion == cold_completion
        prompt_match = warm_prompt == cold_prompt
        telemetry = all(value is not None for value in (
            warm_prompt, cold_prompt, warm_completion, cold_completion))
        passed = (parity and counts_match and prompt_match and telemetry and cold_cached == 0
                  and minimum_reuse is not None and cached is not None
                  and cached >= minimum_reuse)
        comparisons.append({'warm': warm_name, 'cold': cold_name,
                            'answer_and_reasoning_parity': parity,
                            'completion_token_count_parity': counts_match,
                            'prompt_token_count_parity': prompt_match,
                            'cached_tokens': cached, 'cold_cached_tokens': cold_cached,
                            'minimum_reuse_tokens': minimum_reuse, 'passed': passed})
    summary = {'type': 'summary', 'backend': backend, 'branch_mode': branch_mode, 'comparisons': comparisons,
               'cache_parity_passed': all(c['passed'] for c in comparisons),
               'quality_passed': all(r.get('task_check_passed') is True for r in rows.values()),
               'scope': 'Single observation per branch; output equality is not token-ID/logit equality.'}
    save(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=['native', 'vlm'], default='vlm')
    parser.add_argument('--branch-mode', choices=['edit_last_user', 'followup'],
                        default='edit_last_user')
    args = parser.parse_args()
    raw = Path(args.suite).read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    matches = [c for c in cases if c['id'] == args.case_id]
    if len(matches) != 1 or not matches[0].get('grading'):
        parser.error('case-id must select exactly one graded fixture')
    with urllib.request.urlopen(args.endpoint.rstrip('/') + '/cache/stats', timeout=30) as response:
        initial_stats = json.load(response)
    require_memory_only_cache(initial_stats, args.backend)
    with Path(args.output).open('x') as output:
        def save(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
        save({'type': 'metadata', 'label': args.label, 'model': args.model,
              'suite_sha256': hashlib.sha256(raw).hexdigest(), 'case_id': args.case_id,
              'branch_mode': args.branch_mode, 'backend': args.backend,
              'initial_cache_stats': initial_stats,
              'scope': __doc__})
        summary = screen(matches[0], lambda c: evaluate(args.endpoint, c, 7200,
            model=args.model, include_usage=args.backend == 'vlm'), lambda: reset(args.endpoint), save,
            branch_mode=args.branch_mode, backend=args.backend)
    print(json.dumps(summary), flush=True)
    if not summary['cache_parity_passed'] or not summary['quality_passed']:
        raise RuntimeError('branch parity or quality failed; inspect saved evidence')


if __name__ == '__main__':
    main()
