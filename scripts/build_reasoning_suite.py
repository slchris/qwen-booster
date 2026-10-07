"""Three distinct executable cross-file tasks, embedded without cutting source files.

Synthetic semantics tests, not a real-repository benchmark or broad quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path


TASKS = [
    {
        'id': 'config-precedence',
        'files': [
            ('app/defaults.py', 'DEFAULT = {"region": "us", "workers": 8, "timeout": 30, "features": ["audit"]}\n'),
            ('app/resolve.py', 'from .defaults import DEFAULT\ndef resolve(file_cfg, env):\n    cfg = {**DEFAULT, **file_cfg}\n    for key in ("workers", "timeout"):\n        if key.upper() in env:\n            cfg[key] = int(env[key.upper()])\n    cfg["workers"] = min(cfg["workers"], 3)\n    cfg["features"] = sorted(set(cfg["features"]))\n    return cfg\n'),
            ('app/main.py', 'from .resolve import resolve\ndef run():\n    return resolve({"region": "eu", "workers": 6, "features": ["trace", "trace"]}, {"WORKERS": "5", "TIMEOUT": "12", "REGION": "ap"})\n'),
        ],
        'expected': {'region': 'eu', 'workers': 3, 'timeout': 12, 'features': ['trace']},
    },
    {
        'id': 'retry-accounting',
        'files': [
            ('app/policy.py', 'COST = {"timeout": 2, "busy": 1, "ok": 4}\nRETRYABLE = {"timeout", "busy"}\nMAX_ATTEMPTS = 3\n'),
            ('app/worker.py', 'from .policy import COST, RETRYABLE, MAX_ATTEMPTS\ndef attempt(events):\n    cost = 0\n    for n, event in enumerate(events[:MAX_ATTEMPTS], 1):\n        cost += COST.get(event, 0)\n        if event == "ok":\n            return {"ok": True, "cost": cost, "attempts": n}\n        if event not in RETRYABLE:\n            break\n    return {"ok": False, "cost": cost, "attempts": n}\n'),
            ('app/main.py', 'from .worker import attempt\ndef run():\n    jobs = [["timeout", "busy", "ok"], ["busy", "timeout", "busy", "ok"], ["denied", "ok"]]\n    results = [attempt(events) for events in jobs]\n    return {"successful": sum(r["ok"] for r in results), "total_cost": sum(r["cost"] for r in results), "attempts": [r["attempts"] for r in results]}\n'),
        ],
        'expected': {'successful': 1, 'total_cost': 11, 'attempts': [3, 3, 1]},
    },
    {
        'id': 'dependency-scheduler',
        'files': [
            ('app/graph.py', 'DEPS = {"fetch": [], "parse": ["fetch"], "index": ["parse"], "audit": ["fetch"], "publish": ["index", "audit"]}\nDISABLED = {"audit"}\n'),
            ('app/scheduler.py', 'from .graph import DEPS, DISABLED\ndef schedule():\n    done = set()\n    remaining = set(DEPS) - DISABLED\n    order = []\n    while remaining:\n        ready = sorted(k for k in remaining if set(DEPS[k]) <= done)\n        if not ready:\n            break\n        task = ready[0]\n        done.add(task)\n        remaining.remove(task)\n        order.append(task)\n    return {"order": order, "blocked": sorted(remaining)}\n'),
            ('app/main.py', 'from .scheduler import schedule\ndef run():\n    return schedule()\n'),
        ],
        'expected': {'order': ['fetch', 'parse', 'index'], 'blocked': ['publish']},
    },
]


def oracle(task):
    # Execute only the fixed source literals above. Strip local imports because
    # all definitions are evaluated in dependency order in one isolated namespace.
    namespace = {}
    for name, source in task['files']:
        code = '\n'.join(line for line in source.splitlines() if not line.startswith('from .'))
        exec(compile(code, name, 'exec'), namespace)
    answer = namespace['run']()
    if answer != task['expected']:
        raise ValueError('Executable oracle disagrees with independently specified answer')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-input', type=int, default=204800)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model) / 'config.json').read_text())
    native = config.get('text_config', config)['max_position_embeddings']
    reserve = 4096
    if not 4096 <= args.minimum_input <= native - reserve - 1024:
        parser.error('input target must leave output and fixture overhead within native context')
    blocks = [f'\n# file: auxiliary/item_{i:05d}.py\nVALUE_{i} = {(i * 37) % 997}\ndef extra_{i}(x):\n    return (x + VALUE_{i}) % {11 + i % 83}\n' for i in range(args.minimum_input // 20)]
    cases = []
    for task in TASKS:
        expected = oracle(task)
        def make(n):
            points = [n // 20, n // 2, n * 19 // 20]
            parts = ['下面是完整的 Python 源文件。只运行 app 包；auxiliary 文件均未被导入。\n']
            previous = 0
            for point, (name, source) in zip(points, task['files']):
                parts.extend(blocks[previous:point])
                parts.append(f'\n# file: {name}\n{source}\n')
                previous = point
            parts.extend(blocks[previous:n])
            parts.append('\n请沿导入关系推导 app.main.run() 的返回值，仅输出对应 JSON，不添加解释。')
            messages = [{'role': 'user', 'content': ''.join(parts)}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort='medium')
            return messages, ids
        low, high = 0, len(blocks)
        if len(make(high)[1]) < args.minimum_input:
            raise ValueError('Insufficient complete filler files')
        while low < high:
            mid = (low + high) // 2
            if len(make(mid)[1]) < args.minimum_input:
                low = mid + 1
            else:
                high = mid
        messages, ids = make(low)
        assert args.minimum_input <= len(ids) and len(ids) + reserve <= native
        case = {'id': f'{task["id"]}-{args.minimum_input}', 'messages': messages, 'prompt_tokens': len(ids), 'context_tier': native, 'max_tokens': reserve, 'enable_thinking': True, 'reasoning_effort': 'medium', 'seed': 1729, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0, 'grading': {'kind': 'json', 'expected': expected}, 'prompt_token_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'oracle_files': dict(task['files']), 'scope': __doc__}
        cases.append(case)
        print(json.dumps({'id': case['id'], 'actual_input_tokens': len(ids), 'oracle': expected}), flush=True)
    with Path(args.output).open('x') as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
