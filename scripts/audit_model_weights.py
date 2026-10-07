"""Audit local safetensors indexes without loading model tensors onto the GPU.

Checks file sizes against an optional pinned download manifest and indexed tensor
membership with the safetensors parser. This is structural verification, not a
content checksum, quantization quality check or successful inference claim.
"""

import argparse
import hashlib
import json
from pathlib import Path


def audit_weights(path, manifest=None):
    from safetensors import safe_open

    path = Path(path).resolve()
    index_bytes = (path / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)
    weight_map = index.get('weight_map', {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('Index requires a nonempty weight map')
    expected_sizes = {f['name']: f['size'] for f in (manifest or {}).get('weight_files', [])}
    files = []
    for name in sorted(set(weight_map.values())):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Index shard escapes model directory: {name}')
        # Hugging Face snapshots normally link to blobs outside the snapshot.
        file = path / name
        expected_keys = {k for k, v in weight_map.items() if v == name}
        row = {'file': name, 'resolved_path': str(file.resolve()), 'indexed_tensors': len(expected_keys),
               'exists': file.is_file(), 'expected_bytes': expected_sizes.get(name)}
        if file.is_file():
            row['actual_bytes'] = file.stat().st_size
            row['size_matches_manifest'] = (row['actual_bytes'] == expected_sizes[name]
                                            if name in expected_sizes else None)
            try:
                with safe_open(file, framework='numpy') as handle:
                    keys = set(handle.keys())
                    row['missing_indexed_tensors'] = sorted(expected_keys - keys)
                    row['extra_unindexed_tensors'] = len(keys - expected_keys)
                    # Header-only access validates shape metadata without reading tensor payloads.
                    row['indexed_elements'] = sum(
                        _elements(handle.get_slice(key).get_shape()) for key in expected_keys & keys)
                row['format_valid'] = True
            except Exception as error:
                row['format_valid'] = False
                row['error'] = str(error)
        row['passed'] = (row['exists'] and row.get('format_valid', False)
                         and not row.get('missing_indexed_tensors')
                         and row.get('size_matches_manifest') is not False)
        files.append(row)
    manifest_matches_index = (set(expected_sizes) == set(weight_map.values()) if manifest else None)
    return {'model_path': str(path), 'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
            'manifest_repository': (manifest or {}).get('repo'),
            'manifest_revision': (manifest or {}).get('revision'),
            'manifest_matches_index_shards': manifest_matches_index,
            'indexed_tensors': len(weight_map), 'files': files,
            'complete': all(row['passed'] for row in files) and manifest_matches_index is not False,
            'payload_hash_verified': False, 'inference_verified': False}


def _elements(shape):
    import math
    return math.prod(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('refusing to overwrite an audit')
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    report = audit_weights(args.model, manifest)
    with Path(args.output).open('x') as out:
        json.dump(report, out, indent=2)
    print(json.dumps({'complete': report['complete'], 'shards': len(report['files']),
        'passed_shards': sum(row['passed'] for row in report['files'])}))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
