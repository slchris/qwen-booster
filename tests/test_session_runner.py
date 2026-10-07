import importlib.util
import json
import sys
from pathlib import Path
import pytest


@pytest.mark.parametrize("backend,key", [("native", "cached_tokens"), ("vlm", "cache_n")])
def test_sampled_cold_checks_are_not_reported_as_full_parity(tmp_path, monkeypatch, backend, key):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("session_runner", scripts / "evaluate_sessions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    cache_empty = True
    def reset(endpoint):
        nonlocal cache_empty
        cache_empty = True
    def evaluate(endpoint, case, timeout, **kwargs):
        nonlocal cache_empty
        assert kwargs["include_usage"] == (backend == "vlm")
        calls.append(case)
        cached = 0 if cache_empty else 100
        cache_empty = False
        return {"complete": True, "finish_reason": "stop", "text": "OK", "first_content_seconds": 0.1,
                "server_timings": {key: cached}}
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module, "reset", reset)
    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(sys, "argv", ["evaluate_sessions", "--output", str(output),
                                    "--turns", "3", "--cold-every", "3", "--backend", backend])
    module.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 4
    assert [r["parity_passed"] for r in rows] == [None, None, True]
    assert [r["cold_checked"] for r in rows] == [False, False, True]
    assert all(r["reuse_check_passed"] for r in rows)


@pytest.mark.parametrize("fault", ["length", "empty", "reasoning", "cold_cache"])
def test_session_gate_rejects_false_parity(tmp_path, monkeypatch, fault):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("session_runner", scripts / "evaluate_sessions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = 0
    def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"complete": True, "finish_reason": "length" if fault == "length" else "stop",
                "text": "" if fault == "empty" else "OK", "first_content_seconds": 0.1,
                "reasoning": str(calls) if fault == "reasoning" else "",
                "server_timings": {"cache_n": 100 if fault == "cold_cache" else 0}}
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module, "reset", lambda endpoint: None)
    output = tmp_path / "failed.jsonl"
    monkeypatch.setattr(sys, "argv", ["evaluate_sessions", "--output", str(output),
                                    "--turns", "2", "--backend", "vlm"])
    with pytest.raises(RuntimeError, match="parity or reuse failed"):
        module.main()
    row = json.loads(output.read_text())
    assert row["parity_passed"] is False
    assert row["reuse_check_passed"] is False
