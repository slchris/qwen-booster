"""Deterministic answer checks for controlled tasks, separate from broad quality claims."""

import json


def grade_json_format(text, rule):
    """Check raw JSON syntax independently of answer content or fence tolerance."""
    if rule is None or rule.get("kind") != "json":
        return None

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")

    try:
        json.loads(text, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return False
    return True


def _json_equal(actual, expected):
    # Python considers True == 1 and False == 0; JSON keeps these types distinct.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_json_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_json_equal(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def grade_answer(text, rule):
    if rule is None:
        return None
    kind = rule["kind"]
    actual = text.strip()
    if kind == "exact_text":
        return actual == rule["expected"].strip()
    if kind == "json":
        if actual.startswith("```json\n") and actual.endswith("```"):
            actual = actual[8:-3].strip()
        try:
            decoded = json.loads(actual)
        except (ValueError, TypeError):
            return False
        return _json_equal(decoded, rule["expected"])
    raise ValueError(f"unknown grading rule: {kind}")
