"""Deterministic answer checks for controlled tasks, separate from broad quality claims."""

import json
import re

CHOICE_PATTERN = re.compile(r"\b([A-J])\b")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_choice(text):
    """First standalone option letter, the convention used by MCQ benchmarks."""
    match = CHOICE_PATTERN.search(text.upper())
    return match.group(1) if match else None


def extract_number(text):
    """Marker value after the last '####' when present, else the final number."""
    tail = text.rsplit("####", 1)[-1]
    if tail != text:
        candidates = NUMBER_PATTERN.findall(tail)
        if candidates:
            return _to_float(candidates[0])
    candidates = NUMBER_PATTERN.findall(text)
    if not candidates:
        return None
    return _to_float(candidates[-1])


def _to_float(value):
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


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
    if kind == "choice":
        return extract_choice(text) == str(rule["expected"]).strip().upper()
    if kind == "numeric":
        value = extract_number(text)
        if value is None:
            return False
        tolerance = float(rule.get("tolerance", 1e-6))
        return abs(value - float(rule["expected"])) <= tolerance
    if kind in ("contains_all", "contains_any"):
        haystack = text if rule.get("case_sensitive") else text.lower()
        needles = [str(item) for item in rule["expected"]]
        if not rule.get("case_sensitive"):
            needles = [item.lower() for item in needles]
        check = all if kind == "contains_all" else any
        return check(needle in haystack for needle in needles)
    if kind == "regex":
        flags = re.IGNORECASE if rule.get("ignore_case") else 0
        match = re.fullmatch if rule.get("full") else re.search
        return match(rule["pattern"], text, flags) is not None
    raise ValueError(f"unknown grading rule: {kind}")
