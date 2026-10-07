"""Independently review seven recorded branch requests against their full fixture."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from qwen_booster.grading import grade_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--minimum-input-tokens', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.minimum_input_tokens <= 0:
        parser.error('minimum input must be positive')
    source, suite = Path(args.input), Path(args.suite)
    rows = [json.loads(l) for l in source.read_text().splitlines()]
    metadata = rows[0]
    if metadata['type'] != 'metadata' or metadata['suite_sha256'] != hashlib.sha256(suite.read_bytes()).hexdigest():
        raise ValueError('fixture provenance mismatch')
    cases = [c for c in map(json.loads, suite.read_text().splitlines()) if c['id'] == metadata['case_id']]
    if len(cases) != 1:
        raise ValueError('ambiguous fixture')
    case = cases[0]
    names = ['base_cold', 'base_warm', 'branch_a_warm', 'base_after_branch',
             'branch_b_warm', 'branch_a_cold', 'branch_b_cold']
    measurements = [r for r in rows if r['type'] == 'measurement']
    if [r['name'] for r in measurements] != names:
        raise ValueError('incomplete or reordered experiment')
    actual = {r['name']: r['result'] for r in measurements}
    branches = {}
    for key, suffix in [('a', '\n请严格遵守上述输出格式，完成上述任务。'),
                        ('b', '\n请核对上述要求后，完成上述任务。')]:
        messages = copy.deepcopy(case['messages'])
        if metadata['branch_mode'] == 'followup':
            answer = {'role': 'assistant', 'content': actual['base_cold']['text']}
            if actual['base_cold'].get('reasoning'):
                answer['reasoning_content'] = actual['base_cold']['reasoning']
            messages.extend([answer, {'role': 'user', 'content': '请重新回答最初的用户任务。' + suffix}])
        elif metadata['branch_mode'] == 'edit_last_user':
            messages[-1]['content'] += suffix
        else:
            raise ValueError('unknown branch mode')
        branches[key] = messages
    reviewed = []
    for row in measurements:
        name, r = row['name'], row['result']
        messages = branches[name[7]] if name.startswith('branch_') else case['messages']
        expected_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        t, u = r.get('server_timings') or {}, r.get('usage') or {}
        if metadata['backend'] == 'vlm':
            prompt, completion, cached = u.get('prompt_tokens'), u.get('completion_tokens'), t.get('cache_n')
        elif metadata['backend'] == 'native':
            prompt, completion, cached = t.get('prompt_tokens'), t.get('generation_tokens'), t.get('cached_tokens')
        else:
            raise ValueError('unknown backend')
        passed = (r['complete'] and r['finish_reason'] == 'stop' and not r.get('server_errors')
                  and bool(r['text'].strip()) and grade_answer(r['text'], case['grading']) is True
                  and row['messages_sha256'] == expected_hash
                  and isinstance(prompt, int) and prompt >= args.minimum_input_tokens
                  and isinstance(completion, int) and completion > 0)
        reviewed.append({'name': name, 'quality_and_input_passed': bool(passed),
                         'prompt_tokens': prompt, 'completion_tokens': completion,
                         'cached_tokens': cached, 'seconds': r['seconds']})
    by_name = {r['name']: r for r in reviewed}
    pairs = []
    for warm, cold in [('base_warm', 'base_cold'), ('base_after_branch', 'base_cold'),
                       ('branch_a_warm', 'branch_a_cold'), ('branch_b_warm', 'branch_b_cold')]:
        a, b = by_name[warm], by_name[cold]
        passed = (a['quality_and_input_passed'] and b['quality_and_input_passed']
                  and b['cached_tokens'] == 0 and isinstance(a['cached_tokens'], int)
                  and a['cached_tokens'] >= int(by_name['base_cold']['prompt_tokens'] * .95)
                  and all(a[k] == b[k] for k in ('prompt_tokens', 'completion_tokens'))
                  and all(actual[warm].get(k) == actual[cold].get(k) for k in ('text', 'reasoning')))
        pairs.append({'warm': warm, 'cold': cold, 'passed': bool(passed)})
    report = {'bounded_branch_check_passed': all(p['passed'] for p in pairs),
              'observations': reviewed, 'comparisons': pairs, 'minimum_input_tokens': args.minimum_input_tokens,
              'scope': 'One retrieval fixture; full-message hashes, exact grading and four cold/warm pairs. No broad quality or repeated latency claim.',
              'acceptance_complete': False,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, suite)}}
    with Path(args.output).open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
