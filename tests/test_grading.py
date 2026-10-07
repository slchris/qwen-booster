from qwen_booster.grading import grade_answer, grade_json_format


def test_exact_answer_rejects_distractor_substrings():
    assert grade_answer("K-123", {"kind": "exact_text", "expected": "K-123"})
    assert not grade_answer("K-123 or K-456", {"kind": "exact_text", "expected": "K-123"})


def test_json_grading_checks_structure_and_values():
    rule = {"kind": "json", "expected": {"answer": [0, 2]}}
    assert grade_answer('```json\n{"answer": [0, 2]}\n```', rule)
    assert not grade_answer('{"answer": [0, 3]}', rule)
    assert not grade_answer('The answer is 0, 2', rule)
    assert grade_answer("anything", None) is None


def test_json_booleans_are_not_numeric_answers():
    assert not grade_answer('{"index": true}', {"kind": "json", "expected": {"index": 1}})
    assert not grade_answer('{"flags": [1]}', {"kind": "json", "expected": {"flags": [True]}})
    assert grade_answer('{"flags": [true]}', {"kind": "json", "expected": {"flags": [True]}})


def test_json_format_separates_fences_from_correct_content():
    rule = {"kind": "json", "expected": {"answer": 3}}
    fenced = '```json\n{"answer": 3}\n```'
    assert grade_answer(fenced, rule)
    assert grade_json_format(fenced, rule) is False
    assert grade_json_format(' \n{"answer": 4}\t', rule) is True
    assert grade_answer('{"answer": 4}', rule) is False


def test_json_format_rejects_non_json_and_multiple_documents():
    rule = {"kind": "json", "expected": {}}
    for text in ('', '{} explanation', '{} {}', '{"x": NaN}',
                 '{"x": Infinity}', '{"x": -Infinity}', '<think>x</think>{}'):
        assert grade_json_format(text, rule) is False
    assert grade_json_format('{"x": "NaN"}', rule) is True
    assert grade_json_format('anything', None) is None
    assert grade_json_format('anything', {"kind": "exact_text"}) is None


def test_choice_grading_accepts_common_formats():
    rule = {"kind": "choice", "expected": "B"}
    for text in ("B", "b", "(B)", "**B**", "The answer is B.", "答案是 B。"):
        assert grade_answer(text, rule), text
    assert not grade_answer("A is wrong, so maybe not.", rule)
    assert not grade_answer("The letter H appears", rule)


def test_numeric_grading_prefers_the_marker_and_tolerates_formatting():
    rule = {"kind": "numeric", "expected": 42}
    assert grade_answer("reasoning... #### 42", rule)
    assert grade_answer("#### 42.0", rule)
    assert grade_answer("total: 42", rule)
    assert grade_answer("I computed 41 then corrected to 42.", rule)
    assert not grade_answer("#### 41 (I wrote 42 earlier)", rule)
    assert grade_answer("no digits here", rule) is False


def test_numeric_grading_tolerance_is_explicit():
    rule = {"kind": "numeric", "expected": 0.5, "tolerance": 0.01}
    assert grade_answer("0.505", rule)
    assert not grade_answer("0.52", rule)


def test_contains_grading_handles_case_and_lists():
    rule = {"kind": "contains_all", "expected": ["Paris", "France"]}
    assert grade_answer("paris is in france", rule)
    assert not grade_answer("Paris only", rule)
    other = {"kind": "contains_any", "expected": ["97804441", "30158402"]}
    assert grade_answer("The identifier is 30158402.", other)


def test_regex_grading_search_and_full():
    assert grade_answer("answer: 1234", {"kind": "regex", "pattern": r"\d{4}"})
    assert not grade_answer("answer: 12345", {"kind": "regex", "pattern": r"\d{4}", "full": True})
