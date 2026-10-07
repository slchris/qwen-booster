"""Bounded RAM+SSD two-document screen with disk-disabled fresh cold controls.

Full fixtures and history are preserved. Server loading is logged separately;
request latency includes cache restoration but excludes process/model startup.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from evaluate_api import evaluate
from evaluate_document_switching import screen
from evaluate_sessions import reset
from run_document_switching import ROOT, SUITE, AUDIT, OVERRIDES, digest, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['flash-cached', '27b-normalized'], required=True)
    parser.add_argument('--disk-dir', required=True)
    parser.add_argument('--disk-mb', type=int, required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.disk_mb <= 0:
        parser.error('Need positive bounded disk capacity')
    os.chdir(ROOT)
    runs, ids = prepare()
    run = next(r for r in runs if r['mode'] == args.mode)
    disk = Path(args.disk_dir).resolve()
    prefix = Path(args.output_prefix)
    if disk.exists() or list(prefix.parent.glob(prefix.name + '-*')):
        parser.error('Disk directory and output prefix must be fresh')
    if not disk.is_relative_to(ROOT / '.cache'):
        parser.error('Use a fresh test directory under the workspace .cache')
    env_options = {**OVERRIDES, 'APC_CHECKPOINT_INTERVAL_TOKENS': '0',
                   'APC_TRACE': '1', 'APC_DISK_TRACE': '1'}
    config = {'scope': __doc__, 'run': run, 'case_ids': ids,
              'disk_directory': str(disk), 'disk_budget_mb': args.disk_mb,
              'environment_overrides': env_options, 'acceptance_complete': False,
              'cold_control_policy': 'Initial fresh empty SSD; each final cold control uses a new disk-disabled server.'}
    if args.prepare_only:
        print(json.dumps(config, indent=2))
        return
    check = subprocess.run(['pgrep', '-f',
        '[q]wen_booster.(vlm_server|cli)|[t]race_(apc|warm_reserve|snapshot)_server.py|'
        '[b]enchmark_fused_prefill.py|[p]rofile_long_attention.py|[s]creen_head_group_attention.py'],
        capture_output=True, text=True)
    if check.returncode != 1:
        raise RuntimeError('Existing model process or failed process audit: ' + check.stdout)
    if shutil.disk_usage(ROOT).free < args.disk_mb * 1_000_000 + 4_000_000_000:
        raise RuntimeError('Insufficient free disk for declared budget and reserve')
    files = list((ROOT / 'src/qwen_booster').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    hashes = {str(p): digest(p) for p in files + [SUITE, AUDIT]}
    config['source_sha256'] = hashes
    prefix.parent.mkdir(parents=True, exist_ok=True)
    disk.mkdir(parents=True, exist_ok=False)
    Path(str(prefix) + '-config.json').write_text(json.dumps(config, indent=2) + '\n')
    endpoint = 'http://127.0.0.1:18083'
    env = {**os.environ, **env_options}
    server = monitor = log = None
    generation = 0
    clears = 0

    def verify():
        if any(digest(p) != expected for p, expected in hashes.items()):
            raise RuntimeError('Frozen experiment sources changed')

    def status(stage, **extra):
        row = {'stage': stage, 'unix_time': time.time(), 'acceptance_complete': False, **extra}
        Path(str(prefix) + '-status.json').write_text(json.dumps(row, indent=2) + '\n')
        with Path(str(prefix) + '-lifecycle.jsonl').open('a') as events:
            events.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def stop():
        nonlocal server, monitor, log
        if server is not None:
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=120)
            server = None
        if monitor is not None:
            monitor.wait(timeout=30)
            monitor = None
        if log is not None:
            log.close()
            log = None

    def start(use_disk):
        nonlocal server, monitor, log, generation
        stop()
        verify()
        generation += 1
        stem = str(prefix) + f'-server-{generation}'
        command = list(run['command'])
        if use_disk:
            command.extend(['--disk-cache-dir', str(disk), '--disk-cache-mb', str(args.disk_mb)])
        log = Path(stem + '.log').open('x')
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/monitor_memory.py',
            '--pid', str(server.pid), '--output', stem + '-memory.jsonl'])
        launched = time.monotonic()
        status('loading', generation=generation, pid=server.pid, disk_enabled=use_disk)
        while time.monotonic() - launched < 1200:
            if server.poll() is not None:
                raise RuntimeError('Server exited during loading')
            try:
                with urllib.request.urlopen(endpoint + '/health', timeout=5) as response:
                    if json.load(response).get('loaded_model') == run['model']:
                        status('ready', generation=generation, pid=server.pid,
                               disk_enabled=use_disk, startup_seconds=time.monotonic() - launched)
                        return
            except (OSError, TimeoutError):
                pass
            time.sleep(5)
        raise RuntimeError('Server loading timed out')

    def clear():
        nonlocal clears
        clears += 1
        if clears > 1:
            start(False)
        reset(endpoint + '/v1')

    def stats():
        with urllib.request.urlopen(endpoint + '/v1/cache/stats', timeout=30) as response:
            value = json.load(response)
        value['observed_server_pid'] = server.pid
        value['server_generation'] = generation
        if clears > 1 and ('disk_max_bytes' in value or value.get('disk_hits')):
            raise RuntimeError('Cold control must use a disk-disabled fresh server')
        return value

    from transformers import AutoTokenizer
    from audit_workload_tokens import count_case
    tokenizer = AutoTokenizer.from_pretrained(run['model'], local_files_only=True)
    model_config = json.loads((Path(run['model']) / 'config.json').read_text())
    native = model_config.get('text_config', model_config)['max_position_embeddings']
    cases = [json.loads(line) for line in SUITE.read_text().splitlines()]
    try:
        start(True)
        with Path(str(prefix) + '-responses.jsonl').open('x') as output:
            def save(row):
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
            save({'type': 'metadata', 'suite_sha256': digest(SUITE), 'model': run['model'],
                  'scope': __doc__, 'cold_control_policy': config['cold_control_policy']})
            summary = screen(cases, lambda c: evaluate(endpoint + '/v1', c, 7200,
                model=run['model'], include_usage=True), clear, stats, save,
                audit_input=lambda c: count_case(tokenizer, c, native, 204800), allow_disk=True)
        verify()
        status('observations_finished_pending_review', summary=summary,
               bounded_disk_switching_passed=(summary['bounded_switching_passed']
                                             and summary['disk_recovery_observed']))
    except (Exception, KeyboardInterrupt) as error:
        status('failed_pending_review', error=str(error))
        raise
    finally:
        stop()


if __name__ == '__main__':
    main()
