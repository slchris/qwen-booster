"""Report cold and appended-turn context diagnostics as separate workloads."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    def passed(row):
        return row.get("passed") is True and row["result"].get("complete") is True

    groups = defaultdict(list)
    for row in rows:
        if row.get("type") == "measurement":
            groups[row["context_tier"]].append(row)
    tiers = []
    for tier, observations in sorted(groups.items()):
        regimes = {}
        for regime in ("cold", "warm_append"):
            cases = [r for r in observations if r["regime"] == regime]
            timings = [r["result"].get("server_timings") or {} for r in cases]
            latency = [r["result"]["first_visible_answer_seconds"] for r in cases
                       if r["result"].get("complete") and
                       r["result"].get("first_visible_answer_seconds") is not None]
            peaks = [t["peak_mlx_bytes"] for t in timings if t.get("peak_mlx_bytes") is not None]
            regimes[regime] = {
                "requests": len(cases), "passes": sum(passed(r) for r in cases),
                "first_answer_median_seconds": statistics.median(latency) if latency else None,
                "first_answer_range_seconds": [min(latency), max(latency)] if latency else None,
                "prompt_tokens": [t.get("prompt_tokens") for t in timings],
                "cached_tokens": [t.get("cached_tokens") for t in timings],
                "max_observed_mlx_bytes": max(peaks) if peaks else None,
            }
        # Detect missing or duplicate turns rather than certifying a partial file.
        complete = sorted(r["turn"] for r in observations) == [0, 1, 2, 3]
        tiers.append({"context_tier": tier, "four_turn_protocol_complete": complete,
                      "all_checks_passed": complete and all(passed(r) for r in observations),
                      "regimes": regimes})
    return {"tiers": tiers, "daily_use_quality_validated": False,
            "limitations": ["synthetic exact-answer retrieval only",
                            "cold and warm queries differ; no paired speedup ratio",
                            "no P95 estimate from one conversation per tier",
                            "actual token counts leave space for follow-up turns and outputs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    report = summarize(rows)
    report["source_metadata"] = [r for r in rows if r.get("type") == "metadata"]
    with Path(args.output).open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["tiers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
