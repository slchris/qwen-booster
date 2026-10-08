"""Record a cold response and only replay when a complete RAM snapshot was retained."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from evaluate_api import evaluate
from evaluate_sessions import reset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['suite','output','endpoint','model']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--warm-repeats', type=int, default=1)
    args=p.parse_args()
    if args.warm_repeats < 1:
        p.error('--warm-repeats must be positive')
    raw=Path(args.suite).read_bytes();cases=[json.loads(x) for x in raw.splitlines()]
    assert len(cases)==1
    case=cases[0]
    with Path(args.output).open('x') as out:
        def save(row):
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
        save({'type':'metadata','suite_sha256':hashlib.sha256(raw).hexdigest(),'case_id':case['id'],
              'scope':'Admission diagnostic; retain incorrect answers. Skip warm request if no RAM snapshot exists.'})
        reset(args.endpoint)
        for regime in ['cold'] + ['warm' if i == 0 else f'warm_{i+1}' for i in range(args.warm_repeats)]:
            r=evaluate(args.endpoint,case,7200,model=args.model,include_usage=True)
            with urllib.request.urlopen(args.endpoint.rstrip('/')+'/cache/stats',timeout=30) as response:
                stats=json.load(response)
            save({'type':'measurement','regime':regime,'result':r,'cache_stats':stats})
            print(json.dumps({'regime':regime,'correct':r['task_check_passed'],'cache_n':r['server_timings']['cache_n'],'stats':stats}),flush=True)
            if not r['complete'] or r['finish_reason']!='stop':
                raise RuntimeError('Incomplete response')
            if stats.get('exact_resident_bytes',0)==0:
                save({'type':'complete','warm_skipped':regime=='cold','remaining_replays_skipped':True,'after_regime':regime,'reason':'no_retained_RAM_snapshot','acceptance_complete':False})
                return
        save({'type':'complete','warm_skipped':False,'acceptance_complete':False})

if __name__=='__main__':
    main()
