"""Pure reporting helpers. No MLX import, suitable for non-Mac CI."""

import statistics


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["prompt_tokens"], row["prefill_step_size"])
        groups.setdefault(key, []).append(row)
    result = []
    for (length, step), samples in sorted(groups.items()):
        result.append({
            "prompt_tokens": length,
            "prefill_step_size": step,
            "runs": len(samples),
            **{key: statistics.median(s[key] for s in samples) for key in (
                "ttft_seconds", "prefill_tps", "decode_tps", "peak_memory_gb",
            )},
            "ttft_min_seconds": min(s["ttft_seconds"] for s in samples),
            "ttft_max_seconds": max(s["ttft_seconds"] for s in samples),
        })
    return result


def recommend(summary, baseline_step=2048):
    """Recommendations require a measured baseline at the same prompt length."""
    lengths = sorted({s["prompt_tokens"] for s in summary})
    result = []
    for length in lengths:
        candidates = [s for s in summary if s["prompt_tokens"] == length]
        baseline = next((s for s in candidates if s["prefill_step_size"] == baseline_step), None)
        if baseline is None:
            continue
        # Steps >= length-1 execute the same single prefill chunk. Choosing one
        # over another would attribute measurement noise to an optimization.
        distinct = [s for s in candidates
                    if min(s["prefill_step_size"], length - 1)
                    != min(baseline_step, length - 1)]
        best = min([baseline, *distinct], key=lambda s: s["ttft_seconds"])
        result.append({
            "prompt_tokens": length,
            "baseline_step": baseline_step,
            "recommended_step": best["prefill_step_size"],
            "ttft_speedup": baseline["ttft_seconds"] / best["ttft_seconds"],
            "provisional": True,
            "baseline_relative_range": (baseline["ttft_max_seconds"] - baseline["ttft_min_seconds"]) / baseline["ttft_seconds"],
        })
    return result
