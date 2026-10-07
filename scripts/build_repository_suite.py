"""Build a >=200K real-source configuration trace fixture from the pinned backend.

Complete source files are retained; this tests repository reading, not code editing.
The expected answer is independently derived from the launcher and APC defaults.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    repo = Path(args.repository).resolve()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '66e68ce36816cb5134acba3b0d72c6271f35ba24':
        parser.error('expected the pinned backend revision for the reviewed answer')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only'], text=True).strip():
        parser.error('backend tracked sources must be unmodified')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    critical = [Path('src/qwen_booster/vlm_server.py'),
                repo / 'mlx_vlm/server/runtime_config.py', repo / 'mlx_vlm/apc.py']
    intro = ('下面是一份真实项目源码快照。按给出的源码分析配置传递；无需执行代码。'
             '所有文件内容完整，文件排列不是调用顺序。\n')
    question = (
        '\n任务：用户通过 qwen_booster.vlm_server 启动，传入有效 --model、'
        '--context 262144 --cache-mb 24000 --memory-mb 80000；不传其他可选参数。'
        '启动前没有相关环境变量覆盖，运行后没有管理端配置修改。'
        '请从 launcher、RuntimeConfig.from_env 与 APCManager 的配置路径推导最终值。'
        '只返回 JSON，恰好包含以下字段：apc_enabled、disk_enabled（布尔），'
        'memory_max_bytes（APC 驻留预算，整数 byte，不是模型内存预算），'
        'checkpoint_entries、checkpoint_interval_tokens、checkpoint_guard_tokens、'
        'max_kv_size（launcher 传给后端 CLI 的值）、prefill_step_size。')
    expected = {'apc_enabled': True, 'disk_enabled': False, 'memory_max_bytes': 24000000000,
                'checkpoint_entries': 2, 'checkpoint_interval_tokens': 2048,
                'checkpoint_guard_tokens': 1, 'max_kv_size': 262144, 'prefill_step_size': 2048}

    def name(path):
        return str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)

    def section(path):
        return '\n# BEGIN FILE ' + name(path) + '\n' + path.read_text() + '\n# END FILE\n'

    selected = []
    total = sum(len(tokenizer.encode(section(p), add_special_tokens=False)) for p in critical)
    # Other model implementations are genuine repository context, not repeated padding.
    for path in sorted((repo / 'mlx_vlm/models').rglob('*.py')):
        text = section(path)
        size = len(tokenizer.encode(text, add_special_tokens=False))
        if total + size > 225000:
            continue
        selected.append(path)
        total += size
        if total >= 206000:
            break
    midpoint = len(selected) // 2
    ordered = [critical[0], *selected[:midpoint], critical[1], *selected[midpoint:], critical[2]]
    content = intro + ''.join(section(p) for p in ordered) + question
    messages = [{'role': 'user', 'content': content}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=True, reasoning_effort='medium')
    if not 204800 <= len(tokens) <= 258048:
        raise ValueError(f'fixture outside 200K input / native output budget: {len(tokens)}')
    case = {'id': 'repository-200k-apc-trace-thinking', 'messages': messages,
            'prompt_tokens': len(tokens), 'context_tier': 262144, 'max_tokens': 4096,
            'enable_thinking': True, 'reasoning_effort': 'medium', 'temperature': 0,
            'top_p': 1, 'top_k': 0, 'presence_penalty': 0, 'seed': 1729,
            'grading': {'kind': 'json', 'expected': expected}, 'scope': __doc__,
            'source_revision': revision,
            'source_repository': 'https://github.com/Blaizzy/mlx-vlm',
            'source_license': (repo / 'LICENSE').read_text(),
            'source_files': {name(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ordered},
            'prompt_token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
    with Path(args.output).open('x') as output:
        output.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(json.dumps({'id': case['id'], 'prompt_tokens': len(tokens), 'complete_files': len(ordered)}))


if __name__ == '__main__':
    main()
