"""Summarize allocator observations; never infer ownership from byte deltas."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(log):
    checkpoints, pending, errors = [], None, []
    for line in log.splitlines():
        if 'APC_SNAPSHOT_DIAGNOSTIC ' in line:
            event = json.loads(line.split('APC_SNAPSHOT_DIAGNOSTIC ', 1)[1])
            phase = event['phase']
            if phase == 'checkpoint.before':
                if pending is not None:
                    errors.append('new checkpoint before previous checkpoint ended')
                pending = {'events': [], 'admission': []}
                checkpoints.append(pending)
            if pending is None:
                errors.append('snapshot event outside checkpoint: ' + phase)
                continue
            pending['events'].append(event)
            if phase == 'checkpoint.after':
                pending['allocation_deltas'] = {}
                for stage in ('checkpoint', 'extract', 'clone'):
                    before = [e for e in pending['events'] if e['phase'] == stage + '.before']
                    after = [e for e in pending['events'] if e['phase'] == stage + '.after']
                    if len(before) == len(after) == 1:
                        pending['allocation_deltas'][stage] = {
                            key: after[0][key] - before[0][key]
                            for key in ('active_bytes', 'allocator_cache_bytes', 'resident_bytes')}
                    elif before or after:
                        errors.append('missing or repeated phase: ' + stage)
                pending = None
        elif 'APC_MEMORY_DIAGNOSTIC ' in line and pending is not None:
            pending['admission'].append(json.loads(line.split('APC_MEMORY_DIAGNOSTIC ', 1)[1]))
    if pending is not None:
        errors.append('final checkpoint incomplete')
    return {'checkpoints': checkpoints, 'parse_errors': errors,
            'scope': 'Diagnostic allocator deltas, not physical ownership proof or throughput benchmark.',
            'acceptance_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = Path(args.log).read_bytes()
    report = summarize(raw.decode())
    report['source_sha256'] = hashlib.sha256(raw).hexdigest()
    with Path(args.output).open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'checkpoints': len(report['checkpoints']), 'parse_errors': report['parse_errors']}))


if __name__ == '__main__':
    main()
