"""Decode-throughput curve across long contexts on a running OpenAI-compatible server.

For each target context length the script sends one short prime request that
fills and stores the prefix cache, then repeated greedy decode requests with a
fixed output budget. It reports time to first token, decode tokens/s and
cached prompt tokens from stream usage. SSE chunks are not tokens: decode
tokens/s uses streaming ``usage.completion_tokens``. Draft statistics are
recorded when the backend emits them.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

CORPUS = """# 运维诊断手册（示例章节）

## 服务拓扑
网关 api-gateway 承接公网流量，转发到 order-service、payment-service 与 user-service。
所有服务通过 PostgreSQL 主库（一主两从）和 Redis 集群共享状态，事件总线使用 Kafka 三副本。

## 常见告警与处置顺序
1. 5xx 比例上升：先看网关错误分布是否集中在单个下游，再查对应服务实例的 GC 与线程池。
2. P99 延迟恶化：检查数据库连接池使用率、慢查询日志与锁等待，其次检查 DNS 与 sidecar。
3. 队列积压：确认消费者实例数与分区数匹配，检查是否存在毒消息导致的重试风暴。
4. 内存持续上涨：对比 RSS 与堆内指标，排查缓存键膨胀和连接泄漏，必要时限流自保。

## 变更与回滚
任何变更必须先灰度单实例，观察十五分钟核心指标，再分批扩大；回滚窗口不少于两小时。
"""


def tile_tokens(unit, target):
    if not unit:
        raise ValueError("tokenizer produced no tokens for the corpus")
    if target <= 0:
        raise ValueError("target must be positive")
    repeats = math.ceil(target / len(unit))
    return (list(unit) * repeats)[:target]


def build_prompt(tokenizer_dir, target_tokens, corpus=CORPUS):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))
    tokens = tile_tokens(tokenizer.encode(corpus).ids, target_tokens)
    return tokenizer.decode(tokens)


def stream_request(endpoint, body, timeout, api_key):
    started = time.perf_counter()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    first = last = None
    text, timings, usage, errors = "", None, None, []
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
            timings = event.get("timings", timings)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    text += piece
    ended = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens")
    decode_seconds = (last - first) if first and last and last > first else None
    return {"seconds": ended - started,
            "ttft_seconds": (first - started) if first else None,
            "decode_seconds": decode_seconds,
            "decode_tps": ((tokens - 1) / decode_seconds) if decode_seconds and tokens and tokens > 1 else None,
            "completion_tokens": tokens, "usage": usage, "timings": timings,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "errors": errors}


def summarize(rows):
    targets = []
    for row in rows:
        if row.get("target") not in targets:
            targets.append(row["target"])
    summary = {}
    for target in targets:
        prime = [row for row in rows if row.get("target") == target and row.get("phase") == "prime"]
        decode = [row for row in rows if row.get("target") == target and row.get("phase") == "decode"]
        cached = [row["usage"]["prompt_tokens_details"].get("cached_tokens")
                  for row in decode if row.get("usage") and row["usage"].get("prompt_tokens_details")]
        tps = [row["decode_tps"] for row in decode if row.get("decode_tps")]
        ttft = [row["ttft_seconds"] for row in decode if row.get("ttft_seconds")]
        summary[str(target)] = {
            "prime_seconds": prime[0]["seconds"] if prime else None,
            "prompt_tokens": (prime[0].get("usage") or {}).get("prompt_tokens") if prime else None,
            "cached_tokens": cached[0] if cached else None,
            "decode_tps_median": statistics.median(tps) if tps else None,
            "ttft_median_seconds": statistics.median(ttft) if ttft else None,
            "texts_identical": len({row["text_sha256"] for row in decode}) <= 1}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--api-key")
    parser.add_argument("--skip-prime", action="store_true",
                        help="Restore-only: skip cold primes and expect cache hits")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(length < 1024 for length in args.lengths) or args.tokens < 2 or args.repeats < 1:
        parser.error("lengths must be >= 1024; tokens >= 2 and repeats >= 1")
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    rows = []
    for target in args.lengths:
        prompt = build_prompt(args.tokenizer_dir, target)
        base = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "enable_thinking": False, "stream": True,
                "stream_options": {"include_usage": True}}
        if not args.skip_prime:
            prime = stream_request(args.endpoint, {**base, "max_tokens": 8}, args.timeout, args.api_key)
            rows.append({"type": "measurement", "phase": "prime", "target": target,
                         "repeat": 0, **prime})
            print(f"target~{target} prime: {prime['seconds']:.1f}s "
                  f"prompt={((prime.get('usage') or {}).get('prompt_tokens'))}", flush=True)
        for repeat in range(args.repeats):
            result = stream_request(args.endpoint, {**base, "max_tokens": args.tokens},
                                    args.timeout, args.api_key)
            rows.append({"type": "measurement", "phase": "decode", "target": target,
                         "repeat": repeat, **result})
            print(f"target~{target} decode#{repeat}: ttft={result['ttft_seconds']}s "
                  f"decode={result['decode_tps']}tps "
                  f"cached={((result.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')}",
                  flush=True)
    summary = {"targets": summarize(rows), "model": args.model, "tokens": args.tokens,
               "repeats": args.repeats, "release_validated": False,
               "scope": "single local server, one synthetic ops document per length; "
                        "not sustained throughput or quality acceptance"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps({"type": "metadata", "endpoint": args.endpoint,
                                 "model": args.model, "lengths": args.lengths,
                                 "tokens": args.tokens, "scope": summary["scope"]},
                                ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["targets"], indent=2))


if __name__ == "__main__":
    main()
