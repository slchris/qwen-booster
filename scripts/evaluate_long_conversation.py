"""Cold load a long fixture, then verify three distant records through real follow-up turns."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from evaluate_api import evaluate
from evaluate_sessions import reset


def record_questions(content):
    records = list(re.finditer(r"记录 (\d{6}): 项目编号 (\d{8});", content))
    if len(records) < 20:
        raise ValueError("fixture must contain the generated numbered records")
    questions = []
    for fraction in (0.05, 0.5, 0.95):
        match = records[int((len(records)-1) * fraction)]
        questions.append({"record_position_fraction": fraction, "record_id": match[1],
            "question": f"请查找上下文中的记录 {match[1]}，仅输出其项目编号的八位数字，不要解释。",
            "expected": match[2]})
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="build_context_suite output with >=1024 token headroom")
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080/v1")
    args = parser.parse_args()
    fixtures = [json.loads(line) for line in Path(args.suite).read_text().splitlines() if line.strip()]
    fixtures = [c for c in fixtures if c["fact_position_fraction"] == 0.95]
    if not fixtures or any(c.get("headroom_tokens", 0) < 1024 for c in fixtures):
        parser.error("need position-0.95 fixtures with >=1024 tokens reserved for conversation growth")
    with Path(args.output).open("x") as output:
        with urllib.request.urlopen(args.endpoint.rstrip("/").removesuffix("/v1") + "/health", timeout=30) as response:
            health = json.load(response)
        output.write(json.dumps({"type": "metadata", "server_health": health,
            "suite_sha256": hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),
            "scope": "one cold load per tier, then three warm record-location queries; no paired cold speedup claim"}) + "\n")
        output.flush()
        for fixture in fixtures:
            reset(args.endpoint)
            messages = list(fixture["messages"])
            questions = record_questions(messages[0]["content"])
            for turn in range(4):
                expected = fixture["expected_substrings"][0] if turn == 0 else questions[turn-1]["expected"]
                case = {"messages": list(messages), "max_tokens": 32,
                        "grading": {"kind": "exact_text", "expected": expected}}
                result = evaluate(args.endpoint, case, 7200)
                timings = result.get("server_timings") or {}
                passed = result["complete"] and result["task_check_passed"]
                if turn:
                    passed = passed and timings.get("cached_tokens", 0) >= fixture["prompt_tokens"] - 1
                row = {"type": "measurement", "context_tier": fixture["context_tier"], "turn": turn,
                       "regime": "cold" if turn == 0 else "warm_append",
                       "question": questions[turn-1] if turn else None,
                       "passed": passed, "result": result}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps({"tier": fixture["context_tier"], "turn": turn, "passed": passed,
                    "first_answer_seconds": result["first_visible_answer_seconds"],
                    "cached_tokens": timings.get("cached_tokens")}), flush=True)
                if not passed:
                    raise RuntimeError("long-conversation retrieval or prefix reuse failed; inspect saved output")
                if turn < 3:
                    messages.extend([{"role": "assistant", "content": result["text"]},
                                     {"role": "user", "content": questions[turn]["question"]}])


if __name__ == "__main__":
    main()
