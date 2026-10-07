import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'build_opensource_suite', Path(__file__).parents[1] / 'scripts/build_opensource_suite.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_gsm8k_uses_marker_value_and_recommended_thinking_sampling():
    cases, skipped = module.convert('gsm8k', [
        {'id': 'train-1', 'question': 'What is 6*7?', 'answer': 'six sevens #### 42'}],
        thinking=True)
    case = cases[0]
    assert skipped == 0
    assert case['id'] == 'gsm8k-train-1'
    assert case['grading'] == {'kind': 'numeric', 'expected': 42.0}
    assert case['enable_thinking'] and case['max_tokens'] == 2048
    assert case['temperature'] == 1.0 and case['reasoning_effort'] == 'medium'


def test_gsm8k_records_without_numbers_are_skipped():
    cases, skipped = module.convert('gsm8k', [{'id': 1, 'question': '?', 'answer': 'no number'}])
    assert cases == [] and skipped == 1


def test_mmlu_letter_and_index_answers_agree():
    records = [
        {'id': 'a', 'question': 'Q?', 'choices': ['x', 'y', 'z', 'w'], 'answer': 2},
        {'id': 'b', 'question': 'Q?', 'choices': ['x', 'y', 'z', 'w'], 'answer': 'C'}]
    cases, _ = module.convert('mmlu', records)
    assert [c['grading']['expected'] for c in cases] == ['C', 'C']
    assert 'C. z' in cases[0]['messages'][0]['content']


def test_mmlu_out_of_range_is_rejected():
    try:
        module.convert('mmlu', [{'id': 'a', 'question': 'Q?', 'choices': ['x'], 'answer': 3}])
    except ValueError as error:
        assert 'out of range' in str(error)
    else:
        raise AssertionError("out-of-range answer must fail")


def test_niah_and_longbench_use_contains_any():
    niah, _ = module.convert('niah', [
        {'id': 0, 'input': 'haystack then question', 'outputs': ['97804441']}])
    assert niah[0]['grading'] == {'kind': 'contains_any', 'expected': ['97804441']}
    longbench, _ = module.convert('longbench', [
        {'id': 1, 'context': 'doc', 'input': 'q', 'answers': ['a1', 'a2']}])
    content = longbench[0]['messages'][0]['content']
    assert content.startswith('doc') and 'q' in content
    assert longbench[0]['grading']['expected'] == ['a1', 'a2']


def test_limit_sampling_is_deterministic_and_unique():
    records = [{'id': i, 'input': f'q{i}', 'outputs': [str(i)]} for i in range(20)]
    first, _ = module.convert('niah', records, limit=5, seed=7)
    second, _ = module.convert('niah', records, limit=5, seed=7)
    assert [c['id'] for c in first] == [c['id'] for c in second]
    assert len({c['id'] for c in first}) == 5


def test_generic_mapping_reports_missing_fields():
    try:
        module.convert('generic', [{'id': 'x', 'problem': '?'}],
                       generic={'question_field': 'problem', 'context_field': 'context',
                                'answer_field': 'gold', 'kind': 'exact_text'})
    except ValueError as error:
        assert 'gold' in str(error)
    else:
        raise AssertionError("missing answer field must fail")


def test_duplicate_ids_are_rejected():
    try:
        module.convert('niah', [{'id': 1, 'input': 'a', 'outputs': ['x']},
                                {'id': 1, 'input': 'b', 'outputs': ['y']}])
    except ValueError as error:
        assert 'duplicate' in str(error)
    else:
        raise AssertionError("duplicate ids must fail")
