"""Summarize completed diagnostics without treating them as a quality certification."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("type") != "measurement":
            continue
        tier = row.get("context_tier")
        if tier is None:
            tier = int(row["case_id"].split("-")[1])
        groups[tier].append(row)
    result = []
    for tier, cases in sorted(groups.items()):
        latencies = [c["first_visible_answer_seconds"] for c in cases
                     if c.get("complete") and c.get("first_visible_answer_seconds") is not None]
        peaks = [c["server_timings"]["peak_mlx_bytes"] for c in cases
                 if (c.get("server_timings") or {}).get("peak_mlx_bytes") is not None]
        result.append({"context_tier": tier, "observed_requests": len(cases),
            "unique_cases": len({c["case_id"] for c in cases}),
            "complete_requests": sum(bool(c.get("complete")) for c in cases),
            "retrieval_passes": sum(c.get("complete", False) and
                                    c.get("retrieval_check_passed") is True for c in cases),
            "strict_checks": sum(c.get("task_check_passed") is not None for c in cases),
            "strict_passes": sum(c.get("complete", False) and
                                 c.get("task_check_passed") is True for c in cases),
            "max_observed_mlx_bytes": max(peaks) if peaks else None,
            "first_answer_median_seconds": statistics.median(latencies) if latencies else None,
            "first_answer_range_seconds": [min(latencies), max(latencies)] if latencies else None,
            "daily_use_quality_validated": False})
    return {"tiers": result, "recommendation": None,
            "limitations": ["synthetic diagnostics only", "no P95 estimate from small samples",
                            "real task quality, sampling and warm multi-turn evaluation remain required"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    report = summarize(rows)
    report["source_metadata"] = [r for r in rows if r.get("type") == "metadata"]
    with Path(args.output).open("x") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["tiers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
