"""Convert open benchmark exports (JSONL/JSON on disk) into quality suites.

Supported sources: RULER-style need-in-a-haystack, LongBench-style long
documents, GSM8K math, MMLU multiple choice, and a generic field mapping.
The converter is offline by design: point it at exports you already have, it
never downloads, truncates or rewrites the source tasks. Every case records
its grading rule and the fact that a converted open set is still not by itself
daily-use acceptance.

Examples:
  python scripts/build_opensource_suite.py --dataset niah \
      --input exports/ruler_niah_32k.jsonl --output results/suite-niah.jsonl --limit 32
  python scripts/build_opensource_suite.py --dataset gsm8k --thinking \
      --input exports/gsm8k_test.jsonl --output results/suite-gsm8k.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from qwen_booster.grading import extract_number

DATASETS = ("niah", "longbench", "gsm8k", "mmlu", "generic")
SCOPE = ("open-source {dataset} from a local export; the converter does not verify "
         "dataset licensing or contamination, and converted cases are not daily-use acceptance")


def load_records(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a record list")
    return records


def _joined(record, question_field, context_field):
    parts = []
    context = record.get(context_field)
    if context:
        parts.append(str(context))
    question = record.get(question_field)
    if question:
        parts.append(str(question))
    if not parts:
        raise ValueError(f"record {record.get('id', '?')} has no question text")
    return "\n\n".join(parts)


def _case(case_id, prompt, grading, *, thinking, max_tokens):
    case = {"id": case_id, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "enable_thinking": thinking,
            "grading": grading}
    if thinking:
        case.update(reasoning_effort="medium", temperature=1.0, top_p=0.95, top_k=20)
    return case


def convert(dataset, records, *, limit=0, seed=42, max_tokens=0, thinking=False,
            generic=None):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    budget = max_tokens or (2048 if thinking else 512)
    cases, seen, skipped = [], set(), 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        source_id = record.get("id", index)
        case_id = f"{dataset}-{source_id}"
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id}")
        try:
            case = _build(dataset, record, case_id, budget, thinking, generic or {})
        except KeyError as error:
            raise ValueError(f"record {source_id} is missing field {error}") from error
        if case is None:
            skipped += 1
            continue
        seen.add(case_id)
        case["scope"] = SCOPE.format(dataset=dataset)
        cases.append(case)
    if limit and limit < len(cases):
        cases = random.Random(seed).sample(cases, limit)
    return cases, skipped


def _build(dataset, record, case_id, budget, thinking, generic):
    if dataset == "niah":
        prompt = _joined(record, "question" if "question" in record else "input", "context")
        answers = record.get("outputs") or [record["answer"]]
        return _case(case_id, prompt,
                     {"kind": "contains_any", "expected": [str(a) for a in answers]},
                     thinking=thinking, max_tokens=budget)
    if dataset == "longbench":
        prompt = _joined(record, "input", "context")
        answers = record.get("answers") or [record["answer"]]
        return _case(case_id, prompt,
                     {"kind": "contains_any", "expected": [str(a) for a in answers]},
                     thinking=thinking, max_tokens=budget)
    if dataset == "gsm8k":
        expected = extract_number(str(record["answer"]))
        if expected is None:
            return None
        return _case(case_id, str(record["question"]),
                     {"kind": "numeric", "expected": expected},
                     thinking=thinking, max_tokens=budget)
    if dataset == "mmlu":
        choices = record["choices"]
        answer = record["answer"]
        if isinstance(answer, str) and answer.strip().upper().isalpha():
            letter = answer.strip().upper()
        else:
            index = int(answer)
            if not 0 <= index < len(choices):
                raise ValueError(f"record {case_id} answer index out of range")
            letter = chr(ord("A") + index)
        prompt = str(record["question"]) + "\n" + "\n".join(
            f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))
        prompt += "\n\nAnswer with a single letter (A, B, C or D)."
        return _case(case_id, prompt, {"kind": "choice", "expected": letter},
                     thinking=thinking, max_tokens=budget)
    prompt = _joined(record, generic["question_field"], generic.get("context_field"))
    expected = record[generic["answer_field"]]
    kind = generic["kind"]
    if kind == "choice":
        grading = {"kind": "choice", "expected": expected}
    elif kind == "numeric":
        grading = {"kind": "numeric", "expected": expected}
    elif kind in ("contains_all", "contains_any"):
        values = expected if isinstance(expected, list) else [expected]
        grading = {"kind": kind, "expected": [str(v) for v in values]}
    else:
        grading = {"kind": kind, "expected": expected}
    return _case(case_id, prompt, grading, thinking=thinking, max_tokens=budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Local JSONL/JSON exports")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 keeps every converted case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Output budget per case; 0 uses 2048 thinking / 512 non-thinking")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking with the model's recommended sampling")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--kind", default="exact_text",
                        choices=("exact_text", "choice", "numeric", "contains_all",
                                 "contains_any", "regex"))
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens < 0:
        parser.error("--limit and --max-tokens must be nonnegative")
    generic = {"question_field": args.question_field, "context_field": args.context_field,
               "answer_field": args.answer_field, "kind": args.kind}
    records = []
    for value in args.input:
        path = Path(value)
        if not path.is_file():
            parser.error(f"input not found: {path}")
        records.extend(load_records(path))
    try:
        cases, skipped = convert(args.dataset, records, limit=args.limit, seed=args.seed,
                                 max_tokens=args.max_tokens, thinking=args.thinking,
                                 generic=generic)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases ({skipped} skipped) to {output}")


if __name__ == "__main__":
    main()
