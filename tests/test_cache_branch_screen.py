"""Fault injection for the HTTP branch validator; no model or GPU is loaded."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
from evaluate_cache_branches import require_memory_only_cache, screen


def test_disk_cache_rejected_even_before_first_disk_hit():
    require_memory_only_cache({'disk_hits': 0})
    with pytest.raises(ValueError, match='memory-only reset'):
        require_memory_only_cache({'disk_hits': 0, 'disk_max_bytes': 12000000000,
                                   'disk_files': 0})
    with pytest.raises(ValueError, match='cannot verify'):
        require_memory_only_cache({})
    require_memory_only_cache({'entries': 1, 'bytes': 100, 'budget_bytes': 200}, 'native')
    with pytest.raises(ValueError, match='expected native'):
        require_memory_only_cache({'disk_hits': 0}, 'native')


@pytest.mark.parametrize('fault', [None, 'mutated_snapshot', 'cache_miss', 'length', 'quality',
                                  'prompt_mismatch', 'missing_completion'])
@pytest.mark.parametrize('mode', ['edit_last_user', 'followup'])
@pytest.mark.parametrize('backend', ['native', 'vlm'])
def test_branch_screen_rejects_cache_and_completion_faults(fault, mode, backend):
    case = {'messages': [{'role': 'user', 'content': 'Read the complete source.'}],
            'temperature': 0, 'grading': {'kind': 'placeholder'}}
    original = copy.deepcopy(case)
    requests, records, resets = [], [], []
    is_cold = True

    def clear():
        nonlocal is_cold
        is_cold = True
        resets.append(True)

    def request(fixture):
        nonlocal is_cold
        number = len(requests)
        requests.append(copy.deepcopy(fixture))
        result = {'complete': True, 'finish_reason': 'stop',
                  'text': fixture['messages'][-1]['content'], 'reasoning': '',
                  'usage': {'prompt_tokens': 8192, 'completion_tokens': 20},
                  'server_timings': {'cache_n': 0 if is_cold else 8190},
                  'task_check_passed': True}
        is_cold = False
        if number == 3:
            if fault == 'mutated_snapshot':
                result['text'] = 'Corrupted base snapshot after sibling branch.'
            elif fault == 'cache_miss':
                result['server_timings']['cache_n'] = 1
            elif fault == 'length':
                result['finish_reason'] = 'length'
            elif fault == 'quality':
                result['task_check_passed'] = False
            elif fault == 'prompt_mismatch':
                result['usage']['prompt_tokens'] -= 10
            elif fault == 'missing_completion':
                result['usage']['completion_tokens'] = None
        if backend == 'native':
            result['server_timings'] = {
                'prompt_tokens': result['usage']['prompt_tokens'],
                'generation_tokens': result['usage']['completion_tokens'],
                'cached_tokens': result['server_timings']['cache_n']}
            result['usage'] = None
        return result

    if fault == 'length':
        with pytest.raises(RuntimeError, match='incomplete task'):
            screen(case, request, clear, records.append, branch_mode=mode, backend=backend)
        assert records[-1]['result']['finish_reason'] == 'length'
    else:
        summary = screen(case, request, clear, records.append, branch_mode=mode, backend=backend)
        assert summary['cache_parity_passed'] == (fault in (None, 'quality'))
        assert summary['quality_passed'] == (fault != 'quality')
        assert len(resets) == 3
        assert requests[2] == requests[5]
        assert requests[4] == requests[6]
        assert requests[0] == requests[1] == requests[3]
        if mode == 'followup':
            assert requests[2]['messages'][:-2] == original['messages']
            assert requests[2]['messages'][-2] == {
                'role': 'assistant', 'content': original['messages'][-1]['content']}
    assert case == original


def test_followup_preserves_actual_reasoning_and_rejects_empty_base_answer():
    case = {'messages': [{'role': 'user', 'content': 'Original task'}]}
    requests, records = [], []

    def request(fixture):
        requests.append(copy.deepcopy(fixture))
        return {'complete': True, 'finish_reason': 'stop', 'text': 'answer',
                'reasoning': 'actual emitted reasoning', 'task_check_passed': True,
                'usage': {'prompt_tokens': 8192, 'completion_tokens': 20},
                'server_timings': {'cache_n': 8191}}

    screen(case, request, lambda: None, records.append, branch_mode='followup')
    assert requests[2]['messages'][-2]['reasoning_content'] == 'actual emitted reasoning'
    with pytest.raises(RuntimeError, match='incomplete task'):
        screen(case, lambda _: {'complete': True, 'finish_reason': 'stop', 'text': ' '},
               lambda: None, records.append, branch_mode='followup')
