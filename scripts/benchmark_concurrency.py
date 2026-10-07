"""Measure concurrent greedy streaming requests against an OpenAI-compatible API.

For each concurrency level the script starts N requests at the same time over
one shared prompt and reports per-request time-to-first-token and decode
tokens/s plus the level's makespan and aggregate output throughput. SSE
chunks are not tokens: throughput uses stream usage when the backend returns
it and falls back to counting non-empty deltas. This measures one local
server; it is not a capacity plan for other hardware.
"""

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = ("你是一名运维诊断助手。请阅读以下告警并给出排查顺序：\n"
                  "服务 api-gateway 在 10:03 开始 5xx 比例从 0.1% 升到 7%，"
                  "同时下游 payment-service P99 从 120ms 升到 2.4s，"
                  "order-service 线程池排队 340，数据库连接池使用率 98%。"
                  "请用 5 条以内的编号步骤回答。")


def stream_request(endpoint, body, timeout, api_key):
    started = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    first = last = None
    text = ""
    deltas = 0
    usage = None
    errors = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                errors.append(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    deltas += 1
                    text += piece
    ended = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens") or deltas
    decode_seconds = (last - first) if first and last and last > first else None
    return {"started": started, "ended": ended, "first_delta": first,
            "ttft_seconds": (first - started) if first else None,
            "decode_seconds": decode_seconds,
            "decode_tps": (tokens - 1) / decode_seconds if decode_seconds and tokens > 1 else None,
            "tokens": tokens, "delta_chunks": deltas, "usage": usage,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_preview": text[:120], "errors": errors}


def run_level(endpoint, body, level, timeout, api_key):
    barrier = threading.Barrier(level)
    results = [None] * level

    def worker(index):
        try:
            barrier.wait(timeout=30)
            results[index] = stream_request(endpoint, body, timeout, api_key)
        except Exception as error:
            results[index] = {"error": str(error)}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    finished = [result for result in results if result and "error" not in result]
    failed = [result for result in results if result and "error" in result]
    makespan = None
    aggregate_tps = None
    if finished:
        makespan = max(r["ended"] for r in finished) - min(r["started"] for r in finished)
        tokens = sum(r["tokens"] for r in finished)
        aggregate_tps = tokens / makespan if makespan else None
    return results, {"level": level, "completed": len(finished), "failed": len(failed),
                     "makespan_seconds": makespan, "aggregate_tps": aggregate_tps}


def level_summary(rows):
    levels = sorted({row["level"] for row in rows})
    summary = {}
    for level in levels:
        subset = [row for row in rows if row["level"] == level and "error" not in row]
        if not subset:
            summary[str(level)] = {"completed": 0}
            continue
        ttft = [row["ttft_seconds"] for row in subset if row["ttft_seconds"] is not None]
        decode = [row["decode_tps"] for row in subset if row["decode_tps"] is not None]
        summary[str(level)] = {
            "requests": len(subset),
            "ttft_median_seconds": statistics.median(ttft) if ttft else None,
            "decode_tps_median": statistics.median(decode) if decode else None,
            "tokens_total": sum(row["tokens"] for row in subset),
            "texts_identical": len({row["text_sha256"] for row in subset}) == 1}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--api-key")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(level < 1 for level in args.levels) or args.rounds < 1 or args.tokens < 2:
        parser.error("levels, rounds and tokens must be positive (tokens >= 2)")
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.tokens, "temperature": 0, "enable_thinking": False,
            "stream": True, "stream_options": {"include_usage": True}}
    rows, levels = [], {}
    for round_index in range(args.rounds):
        for level in args.levels:
            results, stats = run_level(args.endpoint, body, level, args.timeout, args.api_key)
            stats["round"] = round_index
            levels.setdefault(str(level), []).append(stats)
            for index, result in enumerate(results):
                rows.append({"type": "measurement", "round": round_index,
                             "level": level, "request": index, **result})
            print(f"round={round_index} level={level} completed={stats['completed']} "
                  f"failed={stats['failed']} makespan={stats['makespan_seconds']} "
                  f"aggregate={stats['aggregate_tps']}", flush=True)
    summary = {"levels": level_summary([row for row in rows if "error" not in row]),
               "rounds": levels, "model": args.model, "tokens": args.tokens,
               "release_validated": False,
               "scope": "one local single-model server; not a multi-host capacity plan"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "endpoint": args.endpoint,
                                 "model": args.model, "tokens": args.tokens,
                                 "rounds": args.rounds, "levels": args.levels,
                                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "scope": summary["scope"]}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["levels"], indent=2))


if __name__ == "__main__":
    main()
