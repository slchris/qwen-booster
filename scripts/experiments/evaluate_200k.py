"""Large-input retrieval, ten real turns and a final identical cold replay.

Quality failures are retained without aborting subsequent diagnostic questions.
Protocol failures stop the conversation; no fabricated assistant history is added.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from evaluate_long_conversation import record_questions
from evaluate_sessions import reset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['native', 'vlm'], required=True)
    parser.add_argument('--turns', type=int, default=10)
    parser.add_argument('--stop-on-cache-loss', action='store_true',
                        help='Stop after a VLM response if its APC snapshot was not retained')
    args = parser.parse_args()
    raw = Path(args.suite).read_bytes()
    fixtures = [json.loads(line) for line in raw.splitlines() if line.strip()]
    fixture = next(c for c in fixtures if c['fact_position_fraction'] == 0.95)
    if fixture['prompt_tokens'] < 204800 or fixture['headroom_tokens'] < 4096 or args.turns < 4:
        parser.error('need >=204800 input tokens, >=4096 headroom, and >=4 turns')
    questions = record_questions(fixture['messages'][0]['content'])
    messages = list(fixture['messages'])
    cache_key = 'cache_n' if args.backend == 'vlm' else 'cached_tokens'
    with Path(args.output).open('x') as output:
        def save(row):
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()

        save({'type': 'metadata', 'suite_sha256': hashlib.sha256(raw).hexdigest(),
              'fixture_id': fixture['id'], 'input_tokens': fixture['prompt_tokens'],
              'backend': args.backend, 'model': args.model, 'turns': args.turns,
              'scope': 'synthetic retrieval; final turn only has paired cold replay; no P95'})
        reset(args.endpoint)
        for turn in range(args.turns):
            question = questions[(turn - 1) % len(questions)] if turn else None
            expected = question['expected'] if question else fixture['expected_substrings'][0]
            case = {'messages': list(messages), 'max_tokens': 64, 'enable_thinking': False,
                    'temperature': 0, 'grading': {'kind': 'exact_text', 'expected': expected}}
            result = evaluate(args.endpoint, case, 7200, model=args.model,
                              include_usage=args.backend == 'vlm')
            cached = (result.get('server_timings') or {}).get(cache_key)
            cache_stats = None
            if args.stop_on_cache_loss and args.backend == 'vlm':
                with urllib.request.urlopen(args.endpoint.rstrip('/') + '/cache/stats', timeout=30) as response:
                    cache_stats = json.load(response)
            row = {'type': 'measurement', 'turn': turn, 'question': question,
                   'expected': expected, 'result': result,
                   'cache_stats_after_response': cache_stats,
                   'reuse_passed': cached is not None and (turn == 0 or cached >= fixture['prompt_tokens'] - 1)}
            save(row)
            print(json.dumps({'turn': turn, 'quality_passed': result['task_check_passed'],
                              'reuse_passed': row['reuse_passed'], 'cached_tokens': cached,
                              'first_answer_seconds': result['first_visible_answer_seconds']}), flush=True)
            if not result['complete'] or result['finish_reason'] != 'stop':
                raise RuntimeError('incomplete response; inspect saved result')
            if (cache_stats is not None and turn < args.turns - 1
                    and cache_stats.get('exact_resident_bytes', 0) == 0):
                raise RuntimeError('completed response retained no exact APC snapshot; inspect memory before retrying')
            if turn == args.turns - 1:
                reset(args.endpoint)
                cold = evaluate(args.endpoint, case, 7200, model=args.model,
                                include_usage=args.backend == 'vlm')
                save({'type': 'cold_replay', 'turn': turn, 'result': cold,
                      'parity_passed': cold['complete'] and cold['finish_reason'] == 'stop'
                      and cold['text'] == result['text'] and cold['reasoning'] == result['reasoning']})
            else:
                messages.extend([{'role': 'assistant', 'content': result['text']},
                                 {'role': 'user', 'content': questions[turn % len(questions)]['question']}])


if __name__ == '__main__':
    main()
