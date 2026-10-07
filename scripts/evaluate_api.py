"""Replay a JSONL quality suite against a local OpenAI-compatible streaming API.

Each case: id, messages, optional max_tokens, enable_thinking, expected_substrings.
Substring checks are retrieval diagnostics, never a general intelligence score.
Run each backend with the same suite and settings; cache state is managed externally.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from qwen_booster.grading import grade_answer, grade_json_format


def evaluate(endpoint, case, timeout, model="qwen3.8-27b", include_usage=False):
    body = {"model": model, "messages": case["messages"], "stream": True,
            "temperature": case.get("temperature", 0), "max_tokens": case.get("max_tokens", 256),
            "enable_thinking": case.get("enable_thinking", False)}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    for key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
                "seed", "preserve_thinking", "reasoning_effort"):
        if key in case:
            body[key] = case[key]
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_content = first_reasoning = None
    text, reasoning, finish, done = "", "", None, False
    arrivals = []
    timings = None
    usage = None
    server_errors = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                server_errors.append(event["error"])
            timings = event.get("timings", timings)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                now = time.perf_counter() - started
                if content:
                    if first_content is None and content.strip():
                        first_content = now
                    text += content
                    arrivals.append(now)
                if thought:
                    first_reasoning = now if first_reasoning is None else first_reasoning
                    reasoning += thought
                finish = choice.get("finish_reason") or finish
    expected = case.get("expected_substrings", [])
    # Some backends expose inline <think> as content. Do not mislabel it as正文 latency.
    inline_thinking = "<think>" in text
    return {"seconds": time.perf_counter() - started,
            "first_content_seconds": first_content,
            "first_visible_answer_seconds": None if inline_thinking else first_content,
            "first_reasoning_seconds": first_reasoning,
            "inline_thinking_requires_parsing": inline_thinking,
            "content_chunk_arrivals": arrivals, "text": text, "reasoning": reasoning,
            "finish_reason": finish, "complete": done and finish is not None and not server_errors,
            "server_errors": server_errors,
            "retrieval_check_passed": all(s in text for s in expected) if expected else None,
            "task_check_passed": grade_answer(text, case.get("grading")),
            "bare_json_format_passed": grade_json_format(text, case.get("grading")),
            "quality_review_required": True, "server_timings": timings, "usage": usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--label", required=True, help="Backend, revision, quantization and cache regime")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--include-usage", action="store_true",
                        help="Request optional SSE usage statistics from supporting backends")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    raw = Path(args.suite).read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("suite needs nonempty, unique case IDs")
    try:
        health_url = args.endpoint.rstrip("/").removesuffix("/v1") + "/health"
        with urllib.request.urlopen(health_url, timeout=10) as response:
            health = json.load(response)
    except Exception as error:
        health = {"unavailable": str(error)}
    with Path(args.output).open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "label": args.label,
            "suite_sha256": hashlib.sha256(raw).hexdigest(), "endpoint": args.endpoint,
            "scope": "HTTP wall time; SSE chunks are not token counts; cache state managed externally",
            "repeats": args.repeats, "server_health": health, "model": args.model,
            "include_usage": args.include_usage}) + "\n")
        handle.flush()
        for repeat in range(args.repeats):
            for case in cases:
                try:
                    result = evaluate(args.endpoint, case, args.timeout, model=args.model,
                                      include_usage=args.include_usage)
                except Exception as error:
                    result = {"complete": False, "error": str(error)}
                row = {"type": "measurement", "case_id": case["id"], "repeat": repeat,
                       "context_tier": case.get("context_tier"),
                       "prompt_tokens": case.get("prompt_tokens"), **result}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ("text", "reasoning", "content_chunk_arrivals")}), flush=True)


if __name__ == "__main__":
    main()
