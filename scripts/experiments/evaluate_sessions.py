"""Verify real HTTP multi-turn reuse against cache-cleared responses.

Run against the instrumented qwen-booster server with no other requests.
This mutates only that server's disposable prefix cache.
"""

import argparse
import json
import urllib.request
from pathlib import Path

from evaluate_api import evaluate


def completed_answer(result):
    return (result.get("complete") is True and result.get("finish_reason") == "stop"
            and bool(result.get("text", "").strip()))


def reset(endpoint):
    request = urllib.request.Request(endpoint.rstrip("/") + "/cache/reset", data=b"",
                                     method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--backend", choices=["native", "vlm"], default="native")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--cold-every", type=int, default=1,
                        help="Compare every Nth turn plus the final turn; default checks every turn")
    parser.add_argument("--prefix-id", help="Choose a case ID from --prefix-case")
    parser.add_argument("--prefix-case", help="JSONL suite: use the first case's user content as history")
    args = parser.parse_args()
    if args.turns < 2 or args.cold_every < 1:
        parser.error("turns must be at least two and cold-every must be positive")
    prefix = "请记住项目约束：项目名青鹭，语言 Python，只使用标准库，索引从零开始。"
    if args.prefix_case:
        cases = [json.loads(line) for line in Path(args.prefix_case).read_text().splitlines() if line.strip()]
        matches = [case for case in cases if case["id"] == args.prefix_id] if args.prefix_id else cases
        if not matches:
            parser.error("prefix-id was not found")
        first = matches[0]
        prefix = first["messages"][0]["content"] + "\n" + prefix
    messages = [{"role": "system", "content": "请遵循项目约束，简短准确地回答。"},
                {"role": "user", "content": prefix + "\n请确认项目名。"}]
    questions = ["项目名是什么？只回答名字。", "我们约定使用哪种编程语言？",
                 "可以使用第三方依赖吗？", "索引从几开始？"]
    with Path(args.output).open("x") as output:
        reset(args.endpoint)
        for turn in range(args.turns):
            case = {"messages": list(messages), "max_tokens": 96}
            warm = evaluate(args.endpoint, case, 7200, model=args.model,
                            include_usage=args.backend == "vlm")
            stats = warm.get("server_timings")
            if stats is None:
                raise RuntimeError("server lacks cache telemetry; restart with current code")
            cold = None
            if (turn + 1) % args.cold_every == 0 or turn == args.turns - 1:
                reset(args.endpoint)
                cold = evaluate(args.endpoint, case, 7200, model=args.model,
                                include_usage=args.backend == "vlm")
            # Cold execution leaves a complete snapshot for the next real turn.
            cache_key = "cache_n" if args.backend == "vlm" else "cached_tokens"
            parity = (completed_answer(cold) and completed_answer(warm)
                      and warm["text"] == cold["text"]
                      and warm.get("reasoning", "") == cold.get("reasoning", "")
                      and (cold.get("server_timings") or {}).get(cache_key) == 0
                      if cold is not None else None)
            passed = completed_answer(warm) and parity is not False
            cached_tokens = stats.get(cache_key)
            if cached_tokens is None:
                raise RuntimeError("server response is missing the selected backend's cache telemetry")
            if turn:
                passed = passed and cached_tokens > 0
            row = {"turn": turn, "parity_passed": parity, "reuse_check_passed": passed,
                   "backend": args.backend, "model": args.model,
                   "cold_checked": cold is not None, "warm": warm, "cold": cold}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({"turn": turn, "passed": passed,
                "cached_tokens": cached_tokens,
                "warm_first_content_seconds": warm["first_content_seconds"],
                "cold_first_content_seconds": cold["first_content_seconds"] if cold else None}), flush=True)
            if not passed:
                raise RuntimeError("multi-turn cache parity or reuse failed; inspect saved result")
            messages.extend([{"role": "assistant", "content": warm["text"]},
                             {"role": "user", "content": questions[turn % len(questions)]}])


if __name__ == "__main__":
    main()
