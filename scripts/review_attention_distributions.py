"""Validate completed numerical diagnostics without claiming task quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(raw, suite):
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    cases = [json.loads(line) for line in suite.read_text().splitlines()]
    assert len(rows) == len(cases) + 2, 'Incomplete or unexpected rows'
    meta, terminal = rows[0], rows[-1]
    assert meta['type'] == 'metadata' and terminal['type'] == 'complete'
    assert terminal['cases'] == len(cases)
    assert meta['acceptance_complete'] is False
    assert meta['runtime_default_enabled'] is False
    assert terminal['acceptance_complete'] is False
    assert str(suite) in meta['source_sha256'], 'Suite must be bound to metadata'
    for path, expected in meta['source_sha256'].items():
        assert digest(path) == expected, f'Source changed: {path}'
    for name, expected in meta['environment']['source_sha256'].items():
        assert digest(Path('src/qwen_booster') / name) == expected, name
    count = meta['continuation_tokens']
    assert type(count) is int and count >= 2
    assert len({case['id'] for case in cases}) == len(cases)
    summaries = []
    for case, row in zip(cases, rows[1:-1]):
        assert row['type'] == 'comparison' and row['case_id'] == case['id']
        assert row['input_tokens'] == case['prompt_tokens']
        assert row['prompt_sha256'] == case['prompt_token_sha256']
        assert type(row['candidate_calls']) is int and row['candidate_calls'] > 0
        positions, continuation = row['positions'], row['continuation']
        assert len(positions) == len(continuation) == count
        for index, (position, token) in enumerate(zip(positions, continuation)):
            assert position['position'] == index
            assert type(token) is int and token >= 0
            assert position['baseline_top'] == token
            assert type(position['candidate_top']) is int and position['candidate_top'] >= 0
            assert position['top_equal'] is (token == position['candidate_top'])
            for field in ('baseline_token_nll', 'candidate_token_nll',
                          'kl_baseline_to_candidate', 'max_abs_logprob_delta'):
                assert math.isfinite(position[field]), field
            assert position['max_abs_logprob_delta'] >= 0
        equal = sum(p['top_equal'] for p in positions)
        kl = sum(p['kl_baseline_to_candidate'] for p in positions) / count
        delta = sum(p['candidate_token_nll'] - p['baseline_token_nll']
                    for p in positions) / count
        assert row['top_equal_count'] == equal
        assert math.isclose(row['mean_kl'], kl, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(row['mean_nll_delta'], delta, rel_tol=0, abs_tol=1e-12)
        summaries.append({key: row[key] for key in ('case_id', 'input_tokens',
            'prompt_sha256', 'candidate_calls', 'top_equal_count', 'mean_kl', 'mean_nll_delta')})
    return {'status': 'completed_numerical_diagnostic', 'acceptance_complete': False,
            'runtime_default_enabled': False, 'cases': summaries,
            'positions_per_case': count,
            'source_sha256': {str(p): digest(p) for p in (raw, suite, Path(__file__))},
            'limitations': [
                'Teacher-forced baseline continuation is not a ground-truth answer.',
                'Numerical agreement does not establish free-generation task quality.',
                'No speed, HTTP, MTP, or maximum-context acceptance is inferred.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.input, args.suite)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
