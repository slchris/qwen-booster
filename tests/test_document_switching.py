import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/experiments'))
from evaluate_document_switching import screen


def simulate(*, evict=False, wrong=False, short=False, shared=False, count_mismatch=False,
             oversized=False, lost_history=False, disk=False, cold_disk=False,
             json_output=False, fenced=False):
    cases = [{'messages': [{'role': 'user', 'content': key}], 'temperature': 0,
              'grading': {'kind': 'exact_text', 'expected': key},
              'prompt_tokens': 204807 if count_mismatch else 204800,
              'max_tokens': 60000 if oversized else 64}
             for key in ('A', 'B')]
    if json_output:
        for case in cases:
            case['grading'] = {'kind': 'json', 'expected': {'value': case['messages'][0]['content']}}
    seen = set()
    rows = []
    calls = 0
    disk_hits = 0

    def clear():
        nonlocal disk_hits
        seen.clear()
        disk_hits = 0

    def request(case):
        nonlocal calls, disk_hits
        calls += 1
        key = case['messages'][0]['content']
        followup = len(case['messages']) > 1
        n = (204800 if not short else 8192) + (20 if followup and not lost_history else 0)
        cached = 204799 if key in seen and not evict else 0
        if shared and calls == 2:
            cached = 204799
        if disk and cached:
            disk_hits += 1
        if cold_disk and calls == 7:
            disk_hits += 1
            cached = 204799
        seen.add(key)
        text = json.dumps({'value': key}) if json_output else key
        if fenced:
            text = '```json\n' + text + '\n```'
        return {'complete': True, 'finish_reason': 'stop', 'server_errors': [],
                'text': 'wrong' if wrong and calls == 3 else text, 'reasoning': '',
                'usage': {'prompt_tokens': n, 'completion_tokens': 1,
                          'prompt_tokens_details': {'cached_tokens': cached}},
                'server_timings': {'cache_n': cached, 'prompt_n': n - cached}}

    def audit(case):
        return {'actual_prompt_tokens': 204800 + (20 if len(case['messages']) > 1 else 0),
                'fits_native_context': True, 'meets_minimum_input': True}
    result = screen(cases, request, clear, lambda: {'disk_hits': disk_hits}, rows.append,
                    audit_input=audit, allow_disk=disk or cold_disk)
    return result, rows


def test_two_documents_followups_and_independent_cold_controls():
    result, rows = simulate()
    assert result['bounded_switching_passed']
    measurements = [r for r in rows if r['type'] == 'measurement']
    assert [r['name'] for r in measurements] == [
        'a_initial', 'b_initial', 'a_revisit', 'b_revisit',
        'a_followup', 'b_followup', 'a_followup_cold', 'b_followup_cold']
    assert all(r['result']['server_timings']['cache_n'] == 0 for r in measurements[-2:])
    assert len(result['comparisons']) == 4


@pytest.mark.parametrize('fault', ['evict', 'wrong', 'shared'])
def test_correct_recomputation_wrong_answer_and_shared_document_do_not_pass(fault):
    result, _ = simulate(**{fault: True})
    assert not result['bounded_switching_passed']


def test_short_inputs_do_not_masquerade_as_200k():
    with pytest.raises(RuntimeError, match='full-input'):
        simulate(short=True)


def test_above_floor_but_truncated_initial_input_is_rejected():
    with pytest.raises(RuntimeError, match='independently audited'):
        simulate(count_mismatch=True)


def test_native_context_reserves_output_budget():
    with pytest.raises(RuntimeError, match='reserved output'):
        simulate(oversized=True)


def test_followup_history_cannot_be_dropped_even_when_initial_200k_remains():
    with pytest.raises(RuntimeError, match='full-history token audit'):
        simulate(lost_history=True)


def test_disk_hits_are_measured_per_request_across_cold_resets():
    result, rows = simulate(disk=True)
    assert result['bounded_switching_passed'] and result['disk_recovery_observed']
    assert [r['cache_tier'] for r in rows[:-1]] == [
        'cold', 'cold', 'disk', 'disk', 'disk', 'disk', 'cold', 'cold']
    assert [r['disk_hits_during_request'] for r in rows[:-1]] == [0, 0, 1, 1, 1, 1, 0, 0]


def test_disk_restoration_cannot_masquerade_as_cold_control():
    with pytest.raises(RuntimeError, match='Cold control reused cache'):
        simulate(cold_disk=True)


def test_correct_json_in_fences_fails_required_bare_json_format():
    good, _ = simulate(json_output=True)
    bad, _ = simulate(json_output=True, fenced=True)
    assert good['bounded_switching_passed']
    assert bad['all_answers_correct']
    assert not bad['all_required_formats_correct']
    assert not bad['bounded_switching_passed']
