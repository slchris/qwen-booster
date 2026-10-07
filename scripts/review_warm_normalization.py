"""Independently review a completed singleton-cache cold/warm paired experiment.

This accepts one bounded fixture, not repeated performance or daily-use quality.
It deliberately refuses unfinished runs and does not trust saved comparison flags.
"""

import argparse
import hashlib
import json
from pathlib import Path

from qwen_booster.grading import grade_answer


def review(prefix, minimum_input_tokens):
    sources = {}

    def read(suffix):
        path = Path(prefix + suffix)
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    status = json.loads(read('-status.json'))
    require(status['stage'] == 'finished_pending_review', 'Run is not terminal and complete')
    config = json.loads(read('-config.json'))
    baseline = config['commands']['baseline']
    require(config['commands']['candidate'] == baseline + ['--normalize-single-warm-cache'],
            'Candidate must differ only by the normalization flag')
    suite = Path(config['suite'])
    raw = suite.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sources[str(suite)] = digest
    require(config['source_sha256'][str(suite)] == digest, 'Fixture changed after launch')
    fixtures = [json.loads(line) for line in raw.splitlines() if line.strip()]
    matches = [case for case in fixtures if case['id'] == config['case_id']]
    require(len(matches) == 1, 'Expected exactly one fixture')
    case = matches[0]
    require(bool(case.get('grading')), 'Independent grading specification is required')
    model = baseline[baseline.index('--model') + 1]
    context = int(baseline[baseline.index('--context') + 1])
    observations = {}
    responses = []
    for mode in ('baseline', 'candidate'):
        rows = [json.loads(line) for line in read('-' + mode + '.jsonl').splitlines() if line.strip()]
        require([row['type'] for row in rows] == ['metadata', 'measurement', 'measurement', 'comparison'],
                mode + ': incomplete or unexpected record sequence')
        metadata = rows[0]
        require(metadata['suite_sha256'] == digest and metadata['case_id'] == case['id']
                and metadata['model'] == model and metadata['backend'] == 'vlm',
                mode + ': provenance mismatch')
        log = read('-' + mode + '-server.log').decode()
        marker = 'Normalized singleton Qwen warm cache once: layers=64'
        require((marker in log) == (mode == 'candidate'), mode + ': candidate execution mismatch')
        read('-' + mode + '-memory.jsonl')
        for row, regime in zip(rows[1:3], ('cold', 'warm')):
            require(row['regime'] == regime, mode + ': request order mismatch')
            result = row['result']
            require(result['complete'] and result['finish_reason'] == 'stop'
                    and not result['server_errors'] and bool(result['text'].strip()),
                    mode + '/' + regime + ': incomplete response')
            require(grade_answer(result['text'], case['grading']) is True,
                    mode + '/' + regime + ': answer failed independent grading')
            usage, timing = result['usage'], result['server_timings']
            count, output = usage['prompt_tokens'], usage['completion_tokens']
            require(count >= minimum_input_tokens and count == case['prompt_tokens'],
                    mode + '/' + regime + ': input token count mismatch')
            require(count + case['max_tokens'] <= context and 0 < output <= case['max_tokens'],
                    mode + '/' + regime + ': token budget mismatch')
            cached = 0 if regime == 'cold' else count - 1
            require(timing['cache_n'] == cached == usage['prompt_tokens_details']['cached_tokens']
                    and timing['prompt_n'] + cached == count
                    and timing['predicted_n'] == output
                    and usage['total_tokens'] == count + output,
                    mode + '/' + regime + ': inconsistent cache/token telemetry')
            require(result['seconds'] > 0, 'Nonpositive elapsed time')
            responses.append((result['text'], result['reasoning'], count, output))
            observations[mode + '_' + regime] = {
                'independent_grade': True,
                **{key: result[key] for key in ('seconds', 'first_visible_answer_seconds',
                                               'first_reasoning_seconds', 'usage', 'server_timings')},
            }
    require(all(response == responses[0] for response in responses),
            'Full body, reasoning or token counts differ between requests')
    return {
        'bounded_code_screen_passed': True,
        'actual_input_tokens': responses[0][2], 'output_tokens': responses[0][3],
        'all_four_body_and_reasoning_equal': True, 'candidate_executed': True,
        'observations': observations,
        'observed_warm_total_speedup': observations['baseline_warm']['seconds'] / observations['candidate_warm']['seconds'],
        'scope': 'One full fixture; identical model/backend/budgets/sampling, fresh server per mode. '
                 'Baseline precedes candidate, one cold/warm pair each; no repeated speed or broad quality acceptance.',
        'acceptance_complete': False, 'source_sha256': sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--minimum-input-tokens', type=int, default=204800)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = review(args.prefix, args.minimum_input_tokens)
    with Path(args.output).open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({key: result[key] for key in ('bounded_code_screen_passed', 'actual_input_tokens',
                                                  'observed_warm_total_speedup', 'acceptance_complete')}))


if __name__ == '__main__':
    main()
