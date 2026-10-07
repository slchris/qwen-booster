import copy
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('summary_200k', Path(__file__).parents[1] / 'scripts/summarize_200k.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    rows = [{'type': 'metadata', 'turns': 10, 'backend': 'vlm', 'input_tokens': 204801}]
    for turn in range(10):
        rows.append({'type': 'measurement', 'turn': turn, 'expected': '12345678',
                     'cache_stats_after_response': {'exact_resident_bytes': 6000000000},
                     'result': {'complete': True, 'finish_reason': 'stop', 'text': '12345678',
                                'reasoning': '', 'usage': {'prompt_tokens': 204801 + turn * 40},
                                'server_timings': {'cache_n': 0 if turn == 0 else 204800},
                                'first_visible_answer_seconds': 500 if turn == 0 else 2}})
    cold = copy.deepcopy(rows[-1]['result'])
    cold['server_timings']['cache_n'] = 0
    rows.append({'type': 'cold_replay', 'turn': 9, 'result': cold, 'parity_passed': True})
    return rows


def test_complete_bounded_evidence_is_not_general_quality_acceptance():
    report = module.summarize(evidence())
    assert report['bounded_ten_turn_check_passed']
    assert report['observed_warm_first_answer_median_seconds'] == 2
    assert not report['daily_use_quality_validated']


def test_saved_pass_flag_cannot_hide_cached_or_changed_cold_replay():
    for mutate in (
        lambda r: r['server_timings'].update(cache_n=204800),
        lambda r: r.update(reasoning='changed'),
        lambda r: r.update(finish_reason='length'),
        lambda r: r['usage'].update(prompt_tokens=204800),
    ):
        rows = evidence()
        mutate(rows[-1]['result'])
        assert not module.summarize(rows)['bounded_ten_turn_check_passed']


def test_partial_duplicate_wrong_answer_and_lost_snapshot_fail():
    rows = evidence()
    assert not module.summarize(rows[:-1])['bounded_ten_turn_check_passed']
    assert not module.summarize(rows + [rows[1]])['bounded_ten_turn_check_passed']
    rows[5]['result']['text'] = 'wrong'
    rows[5]['result']['task_check_passed'] = True
    assert not module.summarize(rows)['bounded_ten_turn_check_passed']
    rows = evidence()
    rows[5]['cache_stats_after_response']['exact_resident_bytes'] = 0
    assert not module.summarize(rows)['bounded_ten_turn_check_passed']
