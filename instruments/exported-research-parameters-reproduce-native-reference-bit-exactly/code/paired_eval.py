#!/usr/bin/env python3
"""TAG fable-refit (RF-6): native PAIRED evaluation of a candidate vs a parent on a rank list (own bills, adjusted = MSE*max(.1,util)),
running up to --jobs score_variant.py processes at a time. Reports per-net and pooled raw/adjusted ratios and win counts.
usage: paired_eval.py <candidate.py> <parent.py> <ranks comma list | valid40> <jobs> <out.json>"""
import sys, json, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
cand, parent = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); jobs = int(sys.argv[4]); out = Path(sys.argv[5])
if sys.argv[3] == 'valid40':
    ranks = json.loads((P2 / "codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json").read_text())['valid']
else:
    ranks = [int(x) for x in sys.argv[3].split(',')]
PY = str(P2.parent.parent.parent.parent / '.venv/bin/python')
tasks = [(role, f, r) for r in ranks for role, f in (('parent', parent), ('cand', cand))]
results = {}; running = []
def launch(task):
    role, f, r = task
    p = subprocess.Popen(['nice', '-n', '10', PY, str(HERE / 'score_variant.py'), str(f), str(r)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return (task, p)
t0 = time.time()
while tasks or running:
    while tasks and len(running) < jobs:
        running.append(launch(tasks.pop(0)))
    for item in list(running):
        (role, f, r), p = item
        if p.poll() is not None:
            running.remove(item)
            lines = [l for l in p.stdout.read().splitlines() if l.startswith('{')]
            rec = json.loads(lines[-1]) if lines else dict(error='no output')
            results[f'{role}:{r}'] = rec
            print(json.dumps(dict(role=role, rank=r, util=rec.get('util'), mse=rec.get('mse'), adjusted=rec.get('adjusted'), err=rec.get('error'), elapsed=round(time.time() - t0))), flush=True)
    time.sleep(2)
import math
pairs = [(results[f'cand:{r}'], results[f'parent:{r}']) for r in ranks if 'mse' in results.get(f'cand:{r}', {}) and 'mse' in results.get(f'parent:{r}', {})]
raw_c = sum(c['mse'] for c, p in pairs) / len(pairs); raw_p = sum(p['mse'] for c, p in pairs) / len(pairs)
adj_c = sum(c['adjusted'] for c, p in pairs) / len(pairs); adj_p = sum(p['adjusted'] for c, p in pairs) / len(pairs)
summary = dict(n=len(pairs), raw_cand=raw_c, raw_parent=raw_p, raw_ratio=raw_c / raw_p, adj_cand=adj_c, adj_parent=adj_p, adj_ratio=adj_c / adj_p,
               raw_wins=sum(c['mse'] < p['mse'] for c, p in pairs), adj_wins=sum(c['adjusted'] < p['adjusted'] for c, p in pairs),
               util_cand=sum(c['util'] for c, p in pairs) / len(pairs), util_parent=sum(p['util'] for c, p in pairs) / len(pairs), wall=round(time.time() - t0))
print(json.dumps(dict(summary=summary)), flush=True)
json.dump(dict(candidate=str(cand), parent=str(parent), ranks=ranks, results=results, summary=summary), open(out, 'w'), indent=1)
