"""Check the isolated VLM API contract and prove speculative decoding is active."""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from evaluate_api import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--expect-mtp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("refusing to overwrite results")
    case = {"messages": [{"role": "user", "content": "只回答这个数字：314159"}],
            "max_tokens": 32, "enable_thinking": False, "temperature": 0,
            "expected_substrings": ["314159"]}
    rows = []
    stream = evaluate(args.endpoint, case, 300, model=args.model, include_usage=True)
    timings = stream.get("server_timings") or {}
    mtp_active = (timings.get("draft_kind") == "mtp"
                  and (timings.get("draft_rounds") or 0) > 0
                  and (timings.get("draft_n") or 0) > 0)
    rows.append({"check": "stream", "passed": bool(stream["complete"]
                 and stream["retrieval_check_passed"] and stream.get("usage")
                 and (mtp_active if args.expect_mtp else not mtp_active)),
                 "mtp_active": mtp_active, **stream})

    def post(body):
        request = urllib.request.Request(args.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    body = {"model": args.model, "messages": case["messages"], "max_tokens": 32,
            "temperature": 0, "enable_thinking": False, "stream": False}
    result = post(body)
    choice = result["choices"][0]
    rows.append({"check": "nonstream", "passed": choice["finish_reason"] == "stop"
                 and choice["message"]["content"].strip() == stream["text"].strip(),
                 "response": result})
    body["max_tokens"] = args.context
    # Even this tiny prompt must exceed input + reserved output capacity.
    for streaming in (False, True):
        body["stream"] = streaming
        try:
            result = post(body)
            rows.append({"check": "overflow", "stream": streaming,
                         "passed": False, "response": result})
        except urllib.error.HTTPError as error:
            detail = error.read().decode()
            rows.append({"check": "overflow", "stream": streaming,
                         "passed": error.code == 400 and "context" in detail.lower(),
                         "status": error.code, "detail": detail})
    output.write_text(json.dumps({"endpoint": args.endpoint, "model": args.model,
        "expect_mtp": args.expect_mtp, "checks": rows,
        "all_passed": all(row["passed"] for row in rows)}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: row[k] for k in ("check", "passed")} for row in rows]))
    if not all(row["passed"] for row in rows):
        raise SystemExit("API contract check failed; inspect the report")


if __name__ == "__main__":
    main()
