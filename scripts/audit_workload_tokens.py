"""Count identical workload messages with each local model's own chat template.

Produces a sidecar manifest; never pads, truncates or rewrites the source suite.
This is a tokenizer/context audit, not an inference or quality result.
"""

import argparse
import hashlib
import json
from pathlib import Path


def count_case(tokenizer, case, native_context, minimum_input):
    options = {key: case[key] for key in
               ('enable_thinking', 'preserve_thinking', 'reasoning_effort') if key in case}
    options.setdefault('enable_thinking', False)
    tokens = tokenizer.apply_chat_template(case['messages'], tokenize=True,
        add_generation_prompt=True, return_dict=False, **options)
    count = len(tokens)
    output = case.get('max_tokens', 256)
    return {
        'case_id': case['id'], 'actual_prompt_tokens': count,
        'source_prompt_tokens': case.get('prompt_tokens'),
        'output_budget_tokens': output, 'native_context_tokens': native_context,
        'remaining_history_tokens': native_context - count - output,
        'fits_native_context': count + output <= native_context,
        'meets_minimum_input': count >= minimum_input,
        'prompt_token_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        'messages_sha256': hashlib.sha256(json.dumps(case['messages'], ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'template_options': options,
    }


def main():
    from transformers import AutoTokenizer
    from qwen_booster.models import resolve_local_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--suites', nargs='+', required=True)
    parser.add_argument('--minimum-input', type=int, default=204800)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.minimum_input < 1:
        parser.error('minimum input must be positive')
    suites = []
    for name in args.suites:
        raw = Path(name).read_bytes()
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not cases or len({case['id'] for case in cases}) != len(cases):
            parser.error(f'Empty suite or duplicate case IDs: {name}')
        suites.append((name, hashlib.sha256(raw).hexdigest(), cases))
    results = []
    for value in args.models:
        path, config, profile = resolve_local_model(value, require_weights=False)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        native = config.get('text_config', config)['max_position_embeddings']
        hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                  for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                               'chat_template.jinja') if (path / name).is_file()}
        for name, digest, cases in suites:
            results.append({'model_id': profile.id, 'model_path': str(path),
                'model_file_sha256': hashes, 'suite': name, 'suite_sha256': digest,
                'cases': [count_case(tokenizer, case, native, args.minimum_input) for case in cases]})
    with Path(args.output).open('x') as out:
        json.dump({'scope': 'tokenizer audit only; no inference or quality acceptance',
                   'minimum_input_tokens': args.minimum_input, 'audits': results},
                  out, ensure_ascii=False, indent=2)
    print(f'Wrote {len(results)} model/suite audits to {args.output}')


if __name__ == '__main__':
    main()
