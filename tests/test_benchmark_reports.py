import importlib.util
from pathlib import Path


def script(name):
    root = Path(__file__).parents[1] / "scripts"
    path = root / "experiments" / f"{name}.py"
    if not path.is_file():
        path = root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mtp_speedups_require_matching_outputs_per_case_and_block():
    rows = [{"case": 0, "block": 2, "token_parity": True, "end_to_end_speedup": 1.5},
            {"case": 0, "block": 2, "token_parity": False, "end_to_end_speedup": 3},
            {"case": 1, "block": 3, "token_parity": True, "end_to_end_speedup": 1.2}]
    summary = script("benchmark_mtp").summarize_pairs(rows)
    assert summary[0]["median_paired_speedup"] is None
    assert summary[1]["median_paired_speedup"] == 1.2


def test_incomplete_output_is_not_a_quality_pass():
    rows = [{"type": "measurement", "case_id": "context-8192-position-0.5",
             "complete": False, "task_check_passed": True, "retrieval_check_passed": True,
             "server_timings": None}]
    report = script("summarize_context").summarize(rows)
    assert report["tiers"][0]["strict_passes"] == 0
    assert report["tiers"][0]["retrieval_passes"] == 0
    assert report["recommendation"] is None


def test_conversation_report_rejects_missing_duplicate_or_incomplete_turns():
    summarize = script("summarize_conversations").summarize
    rows = [{"type": "measurement", "context_tier": 8192, "turn": turn,
             "regime": "cold" if turn == 0 else "warm_append", "passed": True,
             "result": {"complete": True, "first_visible_answer_seconds": 20 if turn == 0 else 0.5}}
            for turn in range(4)]
    report = summarize(rows)
    tier = report["tiers"][0]
    assert tier["all_checks_passed"]
    assert tier["regimes"]["cold"]["first_answer_median_seconds"] == 20
    assert tier["regimes"]["warm_append"]["first_answer_median_seconds"] == 0.5
    assert not summarize(rows[:-1])["tiers"][0]["all_checks_passed"]
    assert not summarize(rows + [rows[-1]])["tiers"][0]["all_checks_passed"]
    rows[-1]["result"]["complete"] = False
    tier = summarize(rows)["tiers"][0]
    assert not tier["all_checks_passed"]
    assert tier["regimes"]["warm_append"]["passes"] == 2
    assert not report["daily_use_quality_validated"]
