"""Small controlled reasoning/code-semantics suite; not a comprehensive coding benchmark."""

import argparse
import json
from pathlib import Path


TASKS = [
    ("lower-bound-fix", "下面函数应返回第一个 >= x 的下标。修正唯一的循环推进错误。\n"
     "def lower_bound(a,x):\n    lo,hi=0,len(a)\n    while lo<hi:\n"
     "        mid=(lo+hi)//2\n        if a[mid]<x: lo=mid\n        else: hi=mid\n    return lo\n"
     "仅返回 JSON，键 replacement 是修复后的赋值语句（格式 lo = ...），"
     "键 index 是 a=[1,2,2,5],x=2 的正确返回值。",
     {"replacement": "lo = mid + 1", "index": 1}),
    ("cross-file-semantics", "config.py: FACTOR=3\n"
     "normalize.py: def normalize(x): return max(0, x-2)\n"
     "billing.py: def bill(xs): return sum(normalize(x)*FACTOR for x in xs if x%2==0)\n"
     "调用方先把输入去重再调用 bill。输入 [2,4,4,5,8,-2]。"
     "仅返回 JSON：total 是结果，contributing_inputs 是产生正贡献的去重输入升序列表。",
     {"total": 24, "contributing_inputs": [4, 8]}),
    ("constraint-update", "项目初始约定：编号从1开始；排序按分数降序，同分按名字升序。"
     "随后仅修正编号从0开始，其他约定保持。数据：李=8，王=10，陈=8。"
     "同分名字按 Unicode 码点排序。只返回 JSON：order 是排好序的名字数组，li_index 是李的编号。",
     {"order": ["王", "李", "陈"], "li_index": 1}),
    ("transaction-reasoning", "余额初始100。事务A读取余额后计划减30；事务B随后读取余额后计划加20。"
     "A提交写入70，B按自己读取的旧值提交写入120。没有锁和版本检测。"
     "只返回 JSON：actual 是最终余额；serial 是任意串行执行两笔交易后的余额；"
     "lost_update 是是否发生丢失更新的布尔值。",
     {"actual": 120, "serial": 90, "lost_update": True}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("x") as handle:
        for thinking in (False, True):
            for name, prompt, expected in TASKS:
                case = {"id": f"{name}-thinking-{thinking}",
                    "messages": [{"role": "user", "content": prompt}],
                    "enable_thinking": thinking, "reasoning_effort": "medium",
                    "max_tokens": 2048 if thinking else 512,
                    "temperature": 1.0 if thinking else 0.7,
                    "top_p": 0.95 if thinking else 0.8, "top_k": 20,
                    "presence_penalty": 0.0 if thinking else 1.5, "seed": 1729,
                    "grading": {"kind": "json", "expected": expected},
                    "scope": "controlled code semantics and reasoning; limited screening suite"}
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
