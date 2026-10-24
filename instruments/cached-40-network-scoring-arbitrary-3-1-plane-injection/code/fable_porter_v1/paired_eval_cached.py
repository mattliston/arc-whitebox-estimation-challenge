#!/usr/bin/env python3
"""TAG fable-porter (PT-3): paired evaluation of a candidate against CACHED bank rows (make_bank_cache.py). Runs only the candidate side
(fable_refit_v1/score_variant.py, nice 10, up to <jobs> at a time), then reports exactly the fields of fable_refit_v1/paired_eval.py
(adjusted = MSE*max(.1, util), own bills). Optional [ranks] = comma subset of the cache ranks for a quick screen.
usage: paired_eval_cached.py <candidate.py> <cache.json> <jobs> <out.json> [ranks]   (refuses to overwrite <out.json>)"""
import sys, json, subprocess, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
cand, cache_path, jobs, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3]), Path(sys.argv[4])
cache = json.loads(cache_path.read_text())
scorer = P2 / 'fable_refit_v1/score_variant.py'
assert hashlib.sha256(scorer.read_bytes()).hexdigest() == cache['scorer_sha256'], 'scorer changed since the cache was made'
assert hashlib.sha256(Path(cache['bank']).read_bytes()).hexdigest() == cache['bank_sha256'], 'bank changed since the cache was made'
ranks = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else cache['ranks']
assert all(str(r) in cache['rows'] for r in ranks)
out.open('x').close()
PY = str(P2.parent.parent.parent.parent / '.venv/bin/python')
tasks = list(ranks); results = {f'parent:{r}': dict(cache['rows'][str(r)], cached=True) for r in ranks}; running = []
t0 = time.time()
while tasks or running:
    while tasks and len(running) < jobs:
        r = tasks.pop(0)
        running.append((r, subprocess.Popen(['nice', '-n', '10', PY, str(scorer), str(cand), str(r)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)))
    for item in list(running):
        r, p = item
        if p.poll() is not None:
            running.remove(item)
            lines = [l for l in p.stdout.read().splitlines() if l.startswith('{')]
            rec = json.loads(lines[-1]) if lines else dict(error='no output')
            results[f'cand:{r}'] = rec
            print(json.dumps(dict(role='cand', rank=r, util=rec.get('util'), mse=rec.get('mse'), adjusted=rec.get('adjusted'), err=rec.get('error'),
                                  bank_mse=cache['rows'][str(r)]['mse'], elapsed=round(time.time() - t0))), flush=True)
    time.sleep(2)
pairs = [(results[f'cand:{r}'], results[f'parent:{r}']) for r in ranks if 'mse' in results.get(f'cand:{r}', {})]
raw_c = sum(c['mse'] for c, p in pairs) / len(pairs); raw_p = sum(p['mse'] for c, p in pairs) / len(pairs)
adj_c = sum(c['adjusted'] for c, p in pairs) / len(pairs); adj_p = sum(p['adjusted'] for c, p in pairs) / len(pairs)
summary = dict(n=len(pairs), raw_cand=raw_c, raw_parent=raw_p, raw_ratio=raw_c / raw_p, adj_cand=adj_c, adj_parent=adj_p, adj_ratio=adj_c / adj_p,
               raw_wins=sum(c['mse'] < p['mse'] for c, p in pairs), adj_wins=sum(c['adjusted'] < p['adjusted'] for c, p in pairs),
               util_cand=sum(c['util'] for c, p in pairs) / len(pairs), util_parent=sum(p['util'] for c, p in pairs) / len(pairs),
               wall=round(time.time() - t0), parent_rows='cached')
print(json.dumps(dict(summary=summary)), flush=True)
json.dump(dict(candidate=str(cand), parent=cache['bank'], parent_cache=str(cache_path), ranks=ranks, results=results, summary=summary), open(out, 'w'), indent=1)
