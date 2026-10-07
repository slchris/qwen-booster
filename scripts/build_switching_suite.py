"""Build two independent full-record documents for long-context cache switching."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-input', type=int, default=204800)
    args = parser.parse_args()
    if Path(args.output).exists() or args.minimum_input < 1:
        parser.error('new output path and positive minimum input required')
    from transformers import AutoTokenizer
    model = Path(args.model)
    config = json.loads((model / 'config.json').read_text())
    native = config.get('text_config', config)['max_position_embeddings']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = []
    for label, seed in [('A', 1729), ('B', 8191)]:
        rng = random.Random(seed)
        values = rng.sample(range(10000000, 99999999), 12000)
        records = [f'{label}-记录-{i:06d}: 项目编号 {value}; '
                   f'状态 {rng.choice(["待复核", "已归档", "处理中"])}; '
                   f'校验码 {rng.getrandbits(48):012x}。'
                   for i, value in enumerate(values)]

        def make(count):
            indices = [int((count - 1) * position) for position in (.05, .5, .95)]
            keys = [f'{label}-记录-{i:06d}' for i in indices]
            expected = dict(zip(keys, (values[i] for i in indices)))
            content = (f'独立资料 {label}。记录仅对本资料有效。\n' +
                       '\n'.join(records[:count]) + '\n请查询以下记录的项目编号：' +
                       '、'.join(keys) + '。仅输出 JSON 对象，键为完整记录编号，值为整数项目编号。不要代码围栏或解释。')
            messages = [{'role': 'user', 'content': content}]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            return messages, tokens, expected, indices

        low, high = 3, len(records)
        assert len(make(high)[1]) >= args.minimum_input
        while low < high:
            middle = (low + high) // 2
            if len(make(middle)[1]) >= args.minimum_input:
                high = middle
            else:
                low = middle + 1
        messages, tokens, expected, indices = make(low)
        assert args.minimum_input <= len(tokens) and len(tokens) + 256 <= native
        assert all(messages[0]['content'].count(records[i]) == 1 for i in indices)
        case = {'id': f'document-{label}-{args.minimum_input}', 'messages': messages,
                'max_tokens': 256, 'enable_thinking': False, 'temperature': 0,
                'context_tier': native, 'prompt_tokens': len(tokens),
                'grading': {'kind': 'json', 'expected': expected},
                'record_count': low, 'target_record_indices': indices,
                'document_seed': seed, 'document_label': label,
                'prompt_token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                'scope': 'Synthetic three-position retrieval in distinct full-record documents; '
                         'cache-switching diagnostic, not real-repository quality acceptance.'}
        cases.append(case)
        print(json.dumps({'case': case['id'], 'input_tokens': len(tokens),
                          'records': low, 'expected': expected}), flush=True)
    with Path(args.output).open('x') as out:
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
