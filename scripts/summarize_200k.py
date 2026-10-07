"""Review ten-turn 200K evidence without trusting saved pass flags.

This bounded retrieval check does not certify general quality or daily usability.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(rows):
    metadata = [r for r in rows if r.get('type') == 'metadata']
    turns = [r for r in rows if r.get('type') == 'measurement']
    replays = [r for r in rows if r.get('type') == 'cold_replay']
    failures = []
    if len(metadata) != 1:
        failures.append('need exactly one metadata record')
    meta = metadata[0] if metadata else {}
    if meta.get('backend') != 'vlm':
        failures.append('this reviewer requires VLM token telemetry')
    if meta.get('turns') != 10 or [r.get('turn') for r in turns] != list(range(10)):
        failures.append('need exactly ten consecutive ordered turns')

    def completed(result):
        return (result.get('complete') is True and result.get('finish_reason') == 'stop'
                and bool(result.get('text', '').strip()) and not result.get('server_errors'))

    def cached(result):
        return (result.get('server_timings') or {}).get('cache_n')

    observations = []
    for row in turns:
        result = row.get('result', {})
        usage = result.get('usage') or {}
        count = usage.get('prompt_tokens')
        expected = row.get('expected')
        quality = bool(expected) and result.get('text', '').strip() == expected
        cache = cached(result)
        is_first = row.get('turn') == 0
        reuse = (cache == 0 if is_first else
                 isinstance(cache, int) and cache >= max(204800, meta.get('input_tokens', 0) - 1))
        context = isinstance(count, int) and count >= 204800
        if not (completed(result) and quality and reuse and context):
            failures.append(f"turn {row.get('turn')}: completion, answer, cache or context gate failed")
        stats = row.get('cache_stats_after_response')
        if stats is not None and stats.get('exact_resident_bytes', 0) <= 0:
            failures.append(f"turn {row.get('turn')}: no resident exact snapshot")
        observations.append({'turn': row.get('turn'), 'correct': quality,
                             'input_tokens': count, 'cached_tokens': cache,
                             'seconds': result.get('seconds'),
                             'first_answer_seconds': result.get('first_visible_answer_seconds')})
    if len(replays) != 1 or replays[0].get('turn') != 9 or not turns:
        failures.append('need one final-turn cold replay')
    else:
        cold = replays[0].get('result', {})
        warm = turns[-1].get('result', {})
        cold_count = (cold.get('usage') or {}).get('prompt_tokens')
        warm_count = (warm.get('usage') or {}).get('prompt_tokens')
        if not (completed(cold) and cached(cold) == 0
                and isinstance(cold_count, int) and cold_count >= 204800
                and cold_count == warm_count
                and cold.get('text') == warm.get('text')
                and cold.get('reasoning') == warm.get('reasoning')):
            failures.append('final cold replay completion, zero-cache, input count or output parity failed')
    latencies = [r['first_answer_seconds'] for r in observations[1:]
                 if isinstance(r['first_answer_seconds'], (int, float))]
    return {'bounded_ten_turn_check_passed': not failures, 'failures': failures,
            'model': meta.get('model'), 'suite_sha256': meta.get('suite_sha256'),
            'observed_turns': len(turns), 'turns': observations,
            'observed_warm_first_answer_median_seconds': statistics.median(latencies) if latencies else None,
            'daily_use_quality_validated': False,
            'scope': 'Synthetic retrieval, four repeated follow-up questions; one final cold replay. No P95, broad quality or controlled speedup claim. Replay message identity follows runner implementation; no message hash in legacy records.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw = Path(args.input).read_bytes()
    result = summarize([json.loads(line) for line in raw.splitlines() if line.strip()])
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    with Path(args.output).open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({k: result[k] for k in ('bounded_ten_turn_check_passed', 'observed_turns', 'failures')}))


if __name__ == '__main__':
    main()
