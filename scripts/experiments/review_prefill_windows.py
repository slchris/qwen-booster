"""Validate full-prefix diagnostic records without treating them as acceptance."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    source = Path(args.input)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    meta, final = rows[0], rows[-1]
    assert meta['type'] == 'metadata' and final['type'] == 'complete'
    assert meta['prompt_tokens'] == final['processed_input_tokens'] == 204800
    assert meta['step'] == 2048 and meta['windows'] == [0, 51200, 102400, 196608]
    assert meta['native_context_limit'] == 262144
    for name, digest in meta['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    chunks = [r for r in rows if r['type'] == 'chunk']
    blocks = [r for r in rows if r['type'] == 'block']
    assert len(chunks) == 100 and len(blocks) == 512 and len(rows) == 614
    end = 0
    for row in chunks:
        assert row['start'] == end
        end = min(end + 2048, 204799)
        assert row['end'] == end
        assert row['instrumented'] == (row['start'] in meta['windows'])
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
    windows = []
    for start in meta['windows']:
        selected = [r for r in blocks if r['chunk_start'] == start]
        expected = {(i, 'mlp') for i in range(64)} | {
            (i, 'self_attn' if i % 4 == 3 else 'linear_attn') for i in range(64)}
        assert len(selected) == 128 and {(r['layer'], r['block']) for r in selected} == expected
        sums = collections.defaultdict(float)
        for row in selected:
            assert row['chunk_tokens'] == 2048
            assert math.isfinite(row['seconds']) and row['seconds'] > 0
            if row['block'] == 'self_attn':
                assert row['kv_before'] == start and row['kv_after'] == start + 2048
            sums[row['block']] += row['seconds']
        windows.append({'start': start, 'block_seconds': dict(sums),
                        'fraction_of_instrumented_blocks': {k: v / sum(sums.values()) for k, v in sums.items()}})
    baseline_path = Path(args.baseline)
    baselines = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    baselines = [r for r in baselines if r.get('type') == 'measurement' and r['mode'] == 'default']
    assert len(baselines) == 3
    assert all(r['prompt_sha256'] == meta['prompt_sha256'] for r in baselines)
    parity = all(r['output_token_ids'][0] == final['first_token_id'] for r in baselines)
    report = {'status': 'completed_diagnostic_review', 'acceptance_complete': False,
              'input_tokens': 204800, 'contiguous_prefix_verified': True,
              'first_token_matches_three_baselines': parity, 'windows': windows,
              'chunks': chunks, 'completion': final,
              'limitations': ['Synchronized blocks perturb execution; no speedup claim.',
                             'One synthetic prompt, one pass, no discarded warmup.',
                             'Block shares exclude embedding, normalization and output head overhead.',
                             'First-token parity does not establish semantic quality or full output parity.'],
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source, baseline_path, Path(__file__)]}}
    with Path(args.output).open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'windows': windows, 'first_token_parity': parity}))


if __name__ == '__main__':
    main()
