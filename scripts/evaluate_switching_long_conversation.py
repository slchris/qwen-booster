"""Two complete 200K documents with ten growing followups each and final same-history cold controls.

Repeated-question workload measures cache retention, not broad semantic dialogue quality.
No trimming or compression; preserves actual assistant responses in both histories."""

import argparse
import copy
import hashlib
import json
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from evaluate_cache_branches import require_memory_only_cache
from evaluate_sessions import reset
from qwen_booster.grading import grade_answer, grade_json_format


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def screen(cases, request, clear, stats, save, minimum_input_tokens=204800, audit_input=None,
           allow_disk=False):
    if len(cases) != 2 or digest(cases[0]['messages']) == digest(cases[1]['messages']):
        raise ValueError('Need two distinct complete document fixtures')
    if any(c.get('temperature', 0) != 0 or not c.get('grading') for c in cases):
        raise ValueError('Need greedy fixtures with independent grading')
    def check_stats(value):
        if not allow_disk:
            require_memory_only_cache(value)
        if not isinstance(value.get('disk_hits'), int) or value['disk_hits'] < 0:
            raise RuntimeError('Missing disk-hit telemetry')
    check_stats(stats())
    rows = {}

    def run(name, case, cold=False):
        audit = audit_input(case) if audit_input else None
        if audit and not (audit['fits_native_context'] and audit['meets_minimum_input']):
            raise RuntimeError('Independent input audit failed before request')
        if cold:
            clear()
        before = stats()
        check_stats(before)
        result = request(case)
        after = stats()
        check_stats(after)
        disk_hits = after['disk_hits'] - before['disk_hits']
        if disk_hits < 0:
            raise RuntimeError('Cache counters reset during request')
        cached = (result.get('server_timings') or {}).get('cache_n', 0)
        row = {'type': 'measurement', 'name': name, 'result': result,
               'messages_sha256': digest(case['messages']),
               'independent_grade': grade_answer(result.get('text', ''), case['grading']) is True,
               'bare_json_format_passed': grade_json_format(result.get('text', ''), case['grading']),
               'input_audit': audit,
               'cache_stats_before_request': before,
               'disk_hits_during_request': disk_hits,
               'cache_tier': ('disk' if disk_hits else 'memory') if cached else 'cold',
               'cache_stats_after_response': after}
        save(row)
        rows[name] = row
        if not (result.get('complete') is True and result.get('finish_reason') == 'stop'
                and result.get('text', '').strip() and not result.get('server_errors')):
            raise RuntimeError('Incomplete response; retain partial evidence')
        usage, timing = result.get('usage') or {}, result.get('server_timings') or {}
        n, cached = usage.get('prompt_tokens'), timing.get('cache_n')
        if not (isinstance(n, int) and n >= minimum_input_tokens
                and isinstance(cached, int) and 0 <= cached <= n
                and timing.get('prompt_n', -1) + cached == n
                and usage.get('prompt_tokens_details', {}).get('cached_tokens') == cached):
            raise RuntimeError('Missing/inconsistent full-input or cache telemetry')
        if (cold and (cached != 0 or disk_hits)) or (disk_hits and cached == 0):
            raise RuntimeError('Cold control reused cache')
        if case.get('prompt_tokens') is not None and n != case['prompt_tokens']:
            raise RuntimeError('Input count differs from independently audited complete fixture')
        if audit and n != audit['actual_prompt_tokens']:
            raise RuntimeError('Served input differs from independent full-history token audit')
        if n + case.get('max_tokens', 256) > case.get('context_tier', 262144):
            raise RuntimeError('Input and reserved output exceed native context')

    run('a_initial', cases[0], cold=True)
    run('b_initial', cases[1])
    run('a_revisit', cases[0])
    run('b_revisit', cases[1])
    histories = [copy.deepcopy(case) for case in cases]
    previous = {key: key + '_initial' for key in ('a', 'b')}
    reuse_checks = []
    def check_reuse(name, expected):
        actual = rows[name]['result']['server_timings']['cache_n']
        reuse_checks.append({'request': name, 'expected_cached_tokens': expected,
            'actual_cached_tokens': actual, 'exact_latest_prefix_reused': actual == expected})
    for key in ('a', 'b'):
        check_reuse(key + '_revisit', rows[key + '_initial']['result']['usage']['prompt_tokens'] - 1)
    for turn in range(1, 11):
        for key, history in zip(('a', 'b'), histories):
            prior = rows[previous[key]]['result']
            history.pop('prompt_tokens', None)
            history.pop('prompt_token_sha256', None)
            assistant = {'role': 'assistant', 'content': prior['text']}
            if prior.get('reasoning'):
                assistant['reasoning_content'] = prior['reasoning']
            history['messages'].extend([assistant, {'role': 'user',
                'content': '请重新回答最初的用户任务，严格遵守最初要求的输出格式。'}])
            name = key + '_turn_' + str(turn)
            run(name, history)
            check_reuse(name, prior['usage']['prompt_tokens'] - 1)
            previous[key] = name
    for key, history in zip(('a', 'b'), histories):
        run(key + '_final_revisit', history)
        check_reuse(key + '_final_revisit', rows[previous[key]]['result']['usage']['prompt_tokens'] - 1)
    # Reset RAM separately for each final full-history control; disk is disabled.
    for key, history in zip(('a', 'b'), histories):
        run(key + '_final_cold', history, cold=True)
    comparisons = []
    for key in ('a', 'b'):
        for warm_name, reference in [(key + '_revisit', key + '_initial'),
                                      (key + '_final_revisit', key + '_final_cold')]:
            warm, cold = rows[warm_name], rows[reference]
            w, c = warm['result'], cold['result']
            comparisons.append({'request': warm_name, 'reference': reference,
                'full_messages_identical': warm['messages_sha256'] == cold['messages_sha256'],
                'output_parity': (w['text'], w.get('reasoning'), w['usage']['completion_tokens']) ==
                    (c['text'], c.get('reasoning'), c['usage']['completion_tokens']),
                'input_counts_equal': w['usage']['prompt_tokens'] == c['usage']['prompt_tokens']})
    quality = all(r['independent_grade'] for r in rows.values())
    format_ok = all(r['bare_json_format_passed'] is not False for r in rows.values())
    distinct = rows['b_initial']['result']['server_timings']['cache_n'] == 0
    summary = {'type': 'summary', 'comparisons': comparisons, 'reuse_checks': reuse_checks,
        'response_count': len(rows), 'followups_per_document': 10,
        'all_answers_correct': quality, 'all_required_formats_correct': format_ok,
        'distinct_initial_prefixes': distinct,
        'bounded_switching_passed': quality and format_ok and distinct and all(
            all(c[k] for k in ('full_messages_identical', 'output_parity', 'input_counts_equal'))
            for c in comparisons) and all(c['exact_latest_prefix_reused'] for c in reuse_checks),
        'acceptance_complete': False, 'diagnostic_only': True,
        'cold_followup_controls_in_this_run': True,
        'limit': 'Two documents and ten repeated-question followups per document, with actual assistant histories. '
                 'Each final complete history has a same-process reset-cache control. Traced run, not an isolated '
                 'speed benchmark; no broad quality, P95, or native-maximum acceptance.'}
    save(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('suite', 'endpoint', 'model', 'output'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--case-ids', nargs=2, required=True)
    parser.add_argument('--minimum-input-tokens', type=int, default=204800)
    args = parser.parse_args()
    raw = Path(args.suite).read_bytes()
    suite = [json.loads(line) for line in raw.splitlines() if line.strip()]
    cases = []
    for identifier in args.case_ids:
        matches = [c for c in suite if c['id'] == identifier]
        if len(matches) != 1:
            parser.error('Each case ID must identify exactly one fixture')
        cases.append(matches[0])
    from transformers import AutoTokenizer
    from audit_workload_tokens import count_case
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model) / 'config.json').read_text())
    native = config.get('text_config', config)['max_position_embeddings']

    def stats():
        with urllib.request.urlopen(args.endpoint.rstrip('/') + '/cache/stats', timeout=30) as response:
            return json.load(response)

    with Path(args.output).open('x') as output:
        def save(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
        save({'type': 'metadata', 'scope': __doc__, 'case_ids': args.case_ids,
              'suite_sha256': hashlib.sha256(raw).hexdigest(), 'model': args.model,
              'minimum_input_tokens': args.minimum_input_tokens})
        summary = screen(cases, lambda c: evaluate(args.endpoint, c, 7200,
            model=args.model, include_usage=True), lambda: reset(args.endpoint), stats,
            save, args.minimum_input_tokens,
            audit_input=lambda c: count_case(tokenizer, c, native, args.minimum_input_tokens))
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary['bounded_switching_passed'] else 1)


if __name__ == '__main__':
    main()
