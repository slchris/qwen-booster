"""Validate a completed attention task screen without loading MLX or model weights."""
import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def review(raw, suite):
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    check(bool(cases) and len({c['id'] for c in cases}) == len(cases), 'invalid cases')
    check(len(rows) >= 2 and rows[0]['type'] == 'metadata', 'missing metadata')
    meta = rows[0]
    repeats = meta['repeats']
    check(type(repeats) is int and repeats > 0, 'invalid repeats')
    check(rows[-1] == {'type': 'complete', 'cases': len(cases),
                      'pairs': len(cases) * repeats, 'acceptance_complete': False},
          'incomplete or mismatched completion')
    check(len(rows) == 2 + 3 * len(cases) * repeats, 'unexpected row count')
    check(meta['acceptance_complete'] is False and meta['runtime_default_enabled'] is False,
          'screen cannot claim acceptance')
    check(meta['cache'] == 'fresh per request' and meta['warmup_pairs'] == 0
          and meta['prefill_step_size'] == 2048, 'unexpected execution regime')
    sources = dict(meta['source_sha256'])
    check(any(Path(p).resolve() == suite.resolve() and h == sha(suite)
              for p, h in sources.items()), 'suite not bound to metadata')
    sources.update({str(Path('src/qwen_booster') / p): h
                    for p, h in meta['environment']['source_sha256'].items()})
    for path, expected in sources.items():
        check(sha(path) == expected, f'source changed: {path}')
    grading = runpy.run_path('src/qwen_booster/grading.py')
    cursor = 1
    summaries = []
    for case in cases:
        for repeat in range(repeats):
            pair = {}
            order = ['default', 'fused'] if repeat % 2 == 0 else ['fused', 'default']
            for mode in order:
                row = rows[cursor]
                cursor += 1
                check((row['type'], row['case_id'], row['repeat'], row['mode']) ==
                      ('measurement', case['id'], repeat, mode), 'measurement order mismatch')
                for key, expected in [('input_tokens', case['prompt_tokens']),
                                      ('prompt_sha256', case['prompt_token_sha256']),
                                      ('max_tokens', case['max_tokens']), ('seed', case['seed'])]:
                    check(row[key] == expected, f'{case["id"]}: {key} mismatch')
                check(row['sampling'] == {k: case.get(k) for k in
                      ('temperature', 'top_p', 'top_k', 'presence_penalty')}, 'sampling mismatch')
                calls = row['candidate_calls']
                check(type(calls) is int and (calls == 0 if mode == 'default' else calls > 0),
                      'candidate dispatch mismatch')
                for metric in ('total_seconds', 'first_token_seconds', 'prefill_tokens_per_second',
                               'decode_tokens_per_second', 'peak_mlx_gb'):
                    check(math.isfinite(row[metric]) and row[metric] > 0, f'invalid {metric}')
                check(row['first_token_seconds'] <= row['total_seconds'], 'invalid first token time')
                visible = row['first_visible_seconds']
                check((visible is None and not row['content'].strip()) or
                      (visible is not None and math.isfinite(visible) and
                       row['first_token_seconds'] <= visible <= row['total_seconds']),
                      'invalid visible time')
                ids = row['output_token_ids']
                check(len(ids) == row['output_tokens'] and 0 < len(ids) <= case['max_tokens']
                      and all(type(i) is int and i >= 0 for i in ids), 'invalid output tokens')
                check(row['finish_reason'] in ('stop', 'length'), 'unexpected finish reason')
                check(row['answer_correct'] == grading['grade_answer'](row['content'], case['grading']),
                      'answer grade mismatch')
                check(row['json_format_correct'] == grading['grade_json_format'](row['content'], case['grading']),
                      'format grade mismatch')
                pair[mode] = row
            a, b = pair['default'], pair['fused']
            expected = {'type': 'pair', 'case_id': case['id'], 'repeat': repeat,
                        'token_parity': a['output_token_ids'] == b['output_token_ids'],
                        'both_answers_correct': all(r['answer_correct'] and r['finish_reason'] == 'stop'
                                                    for r in pair.values()),
                        'both_json_formats_correct': all(r['json_format_correct'] is not False
                                                        for r in pair.values()),
                        'first_token_speedup': a['first_token_seconds'] / b['first_token_seconds']}
            check(rows[cursor] == expected, 'pair calculation mismatch')
            cursor += 1
            summaries.append({**expected, 'measurements': [{k: v for k, v in r.items()
                              if k not in ('output_token_ids', 'content', 'reasoning')}
                              for r in (a, b)]})
    return {'status': 'completed_bounded_task_screen', 'acceptance_complete': False,
            'runtime_default_enabled': False,
            'source_sha256': {str(p): sha(p) for p in (raw, suite, Path(__file__))},
            'pairs': summaries,
            'limits': ['Original suite gold answers regraded; gold correctness not independently established here.',
                       'Prompt token hashes checked against suite; tokenizer not rerun.',
                       'No HTTP, MTP, warm-cache, native-boundary or broad quality acceptance.',
                       'No warmup; single AB observations do not establish stable speedup.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.input, args.suite)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
