"""Serial, dedicated-server two-document 200K screen; no automatic GPU overlap."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/document-switching-200k-suite-v1.jsonl'
AUDIT = ROOT / 'results/document-switching-200k-token-audit-v1.json'
OVERRIDES = {'PYTHONPATH': str(ROOT / 'src'), 'MLX_GDN_PACKED': '1',
             'APC_CHECKPOINT_ENTRIES': '4', 'APC_DISK_ENABLED': '0',
             'APC_MEMORY_RESERVE_GB': str(4_000_000_000 / (1 << 30))}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    audits = json.loads(AUDIT.read_text())['audits']
    cases = [json.loads(line) for line in SUITE.read_text().splitlines()]
    ids = [case['id'] for case in cases]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise RuntimeError('Need exactly two distinct fixture IDs')
    settings = {
        'qwen3.8-27b': ('27b-combined', '48000', ['--draft', str(ROOT / '.cache/qwen3.8-mtp-4bit'), '--release-finished-mtp', '--fused-prefill', '--borrow-27b-checkpoint-row', '--credit-27b-live-warm-cache']),
        'qwen3.8-flash-next': ('flash-cached', '32000',
                             ['--credit-live-warm-cache', '--borrow-single-checkpoint-row'])}
    runs = []
    for audit in audits:
        if audit['suite_sha256'] != digest(SUITE):
            raise RuntimeError('Suite changed since token audit')
        by_id = {c['case_id']: c for c in audit['cases']}
        for case in cases:
            item = by_id[case['id']]
            if not (item['actual_prompt_tokens'] >= 204800 and item['fits_native_context']
                    and item['actual_prompt_tokens'] + case['max_tokens'] <= 262144):
                raise RuntimeError('Fixture does not fit required context')
        model = Path(audit['model_path'])
        for name, expected in audit['model_file_sha256'].items():
            if digest(model / name) != expected:
                raise RuntimeError('Audited model/tokenizer file changed')
        mode, cache_mb, flags = settings[audit['model_id']]
        command = [str(ROOT / '.cache/vlm-env/bin/python'), '-m', 'qwen_booster.vlm_server',
                   '--model', str(model), '--context', '262144', '--port', '18083',
                   '--cache-mb', cache_mb, '--memory-mb', '110000',
                   '--allocator-cache-mb', '2000', '--token-timeout', '7200', *flags]
        runs.append({'mode': mode, 'model': str(model), 'command': command})
    if len(runs) != 2 or {a['model_id'] for a in audits} != set(settings):
        raise RuntimeError('Need one audited run per model')
    return runs, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', default='results/credit-27b-switching-200k-v1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['27b-combined'],
                        default=['27b-combined'])
    parser.add_argument('--checkpoint-interval', type=int, default=0)
    parser.add_argument('--trace-cache', action='store_true')
    parser.add_argument('--trace-admission', action='store_true',
                        help='Observe memory admission/evictions; diagnostic, not timing benchmark')
    args = parser.parse_args()
    if args.checkpoint_interval < 0 or len(set(args.modes)) != len(args.modes):
        parser.error('Need nonnegative checkpoint interval and unique modes')
    OVERRIDES['APC_CHECKPOINT_INTERVAL_TOKENS'] = str(args.checkpoint_interval)
    OVERRIDES['APC_TRACE'] = '1' if args.trace_cache else '0'
    os.chdir(ROOT)
    gate = json.loads((ROOT / 'results/credit-27b-conversation-200k-v1-review.json').read_text())
    assert gate['content_correct'] == 10 and gate['warm_reuse_turns'] == 9
    assert gate['final_cold_replay']['content_correct'] and gate['final_cold_replay']['output_parity']
    prior = json.loads((ROOT / 'results/credit-27b-conversation-200k-v1-config.json').read_text())
    assert all(digest(p) == h for p, h in prior['source_sha256'].items())
    runs, ids = prepare()
    runs = [next(run for run in runs if run['mode'] == mode) for mode in args.modes]
    if args.trace_admission:
        for run in runs:
            run['command'][1:3] = [str(ROOT / 'scripts/trace_apc_server.py')]
    config = {'runs': runs, 'case_ids': ids, 'environment_overrides': OVERRIDES,
              'admission_diagnostic': args.trace_admission,
              'acceptance_complete': False,
              'scope': 'Eight serial requests per model: A/B initial, A/B revisit, '
                       'A/B repeated-question follow-up, independent cold follow-up controls. '
                       'Combined 27B fused/MTP/APC with borrowed checkpoint normalization and live warm-cache credit; 48GB RAM cache, final-only checkpoints, disk disabled. Not an isolated speedup comparison.'}
    if args.prepare_only:
        print(json.dumps(config, indent=2))
        return
    # A live attention benchmark must finish before a model server may start.
    check = subprocess.run(['pgrep', '-f',
        '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|'
        '[b]enchmark_fused_prefill.py|[p]rofile_long_attention.py|[s]creen_head_group_attention.py'],
        capture_output=True, text=True)
    if check.returncode != 1:
        raise RuntimeError('Existing model process or failed process audit: ' + check.stdout)
    prefix = Path(args.output_prefix)
    if list(prefix.parent.glob(prefix.name + '-*')):
        raise RuntimeError('Output prefix already has artifacts; choose a new prefix')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = list((ROOT / 'src/qwen_booster').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    hashes = {str(p): digest(p) for p in files + [SUITE, AUDIT]}
    config['source_sha256'] = hashes
    Path(str(prefix) + '-config.json').write_text(json.dumps(config, indent=2) + '\n')
    env = {**os.environ, **OVERRIDES}

    def status(stage, **extra):
        row = {'stage': stage, 'unix_time': time.time(), 'acceptance_complete': False, **extra}
        Path(str(prefix) + '-status.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    def verify_sources():
        if any(digest(p) != expected for p, expected in hashes.items()):
            raise RuntimeError('Experiment sources changed')

    outcomes = []
    try:
        for run in runs:
            verify_sources()
            stem = str(prefix) + '-' + run['mode']
            monitor = None
            with Path(stem + '-server.log').open('x') as log:
                server = subprocess.Popen(run['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    monitor = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/monitor_memory.py',
                        '--pid', str(server.pid), '--output', stem + '-memory.jsonl'])
                    status('loading', mode=run['mode'], pid=server.pid)
                    deadline = time.monotonic() + 1200
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError('Model server exited')
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:18083/health', timeout=5) as response:
                                if json.load(response).get('loaded_model') == run['model']:
                                    break
                        except (OSError, TimeoutError):
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError('Model readiness timeout')
                        time.sleep(5)
                    output = stem + '.jsonl'
                    status('evaluating', mode=run['mode'], pid=server.pid, output=output)
                    result = subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/evaluate_document_switching.py',
                        '--suite', str(SUITE), '--case-ids', *ids, '--model', run['model'],
                        '--endpoint', 'http://127.0.0.1:18083/v1', '--output', output], env=env)
                    rows = [json.loads(line) for line in Path(output).read_text().splitlines()]
                    if (len(rows) != 10 or rows[-1].get('type') != 'summary'
                            or result.returncode not in (0, 1)):
                        raise RuntimeError('Incomplete screen; preserve partial evidence')
                    verify_sources()
                    assert all(marker in Path(stem + '-server.log').read_text() for marker in ['FUSED_PREFILL_ACTIVE', 'BORROW_27B_CHECKPOINT_ROW', 'WARM_27B_RESERVE'])
                    outcomes.append({'mode': run['mode'], 'summary': rows[-1]})
                    status('screen_recorded', mode=run['mode'], summary=rows[-1])
                finally:
                    if server.poll() is None:
                        server.terminate()
                    server.wait(timeout=120)
                    if monitor is not None:
                        monitor.wait(timeout=30)
        status('observations_finished_pending_review', outcomes=outcomes)
    except (Exception, KeyboardInterrupt) as error:
        status('failed_pending_review', error=str(error))
        raise


if __name__ == '__main__':
    main()
