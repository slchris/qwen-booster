import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/experiments'))
from evaluate_switching_long_conversation import screen


def simulate(fault=None):
    cases = [{'messages': [{'role': 'user', 'content': key}], 'temperature': 0,
              'grading': {'kind': 'exact_text', 'expected': key}, 'max_tokens': 256}
             for key in ('A', 'B')]
    latest, rows, requests = {}, [], []
    resets = 0

    def clear():
        nonlocal resets
        resets += 1
        latest.clear()

    def count(case):
        return 204800 + 50 * (len(case['messages']) - 1)

    def request(case):
        requests.append(copy.deepcopy(case['messages']))
        key = case['messages'][0]['content']
        messages = case['messages']
        # Every appended answer must be the actual response for its prior history.
        for index in range(1, len(messages), 2):
            assert messages[index] == {'role': 'assistant', 'content': key,
                                      'reasoning_content': 'history-' + str(index)}
        n = count(case)
        cached = latest.get(key, 1) - 1
        if fault == 'old_prefix' and cached:
            cached = 204799  # >99% reuse still fails the exact-history gate.
        if fault == 'lost_history' and len(messages) > 1:
            n = 204800
        if fault == 'cold_reuse' and len(requests) == 27:
            cached = n - 1
        latest[key] = n
        return {'complete': True, 'finish_reason': 'stop', 'server_errors': [],
                'text': key, 'reasoning': 'history-' + str(len(messages)),
                'usage': {'prompt_tokens': n, 'completion_tokens': 1,
                          'prompt_tokens_details': {'cached_tokens': cached}},
                'server_timings': {'cache_n': cached, 'prompt_n': n - cached}}

    result = screen(cases, request, clear, lambda: {'disk_hits': 0}, rows.append,
                    audit_input=lambda c: {'actual_prompt_tokens': count(c),
                         'fits_native_context': True, 'meets_minimum_input': True})
    return result, requests, resets


def test_actual_histories_grow_and_final_cold_prompts_match():
    result, requests, resets = simulate()
    assert result['bounded_switching_passed']
    assert len(requests) == 28 and resets == 3
    assert len(result['reuse_checks']) == 24
    assert requests[24] == requests[26] and requests[25] == requests[27]
    assert len(requests[26]) == len(requests[27]) == 21


def test_old_prefix_reuse_is_not_full_history_retention():
    result, _, _ = simulate('old_prefix')
    assert result['all_answers_correct']
    assert not result['bounded_switching_passed']


@pytest.mark.parametrize('fault,match', [('lost_history', 'full-history token audit'),
                                       ('cold_reuse', 'Cold control reused cache')])
def test_invalid_controls_and_dropped_history_are_rejected(fault, match):
    with pytest.raises(RuntimeError, match=match):
        simulate(fault)
