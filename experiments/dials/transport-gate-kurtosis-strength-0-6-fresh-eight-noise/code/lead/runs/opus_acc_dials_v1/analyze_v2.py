"""[opus-acc] paired changes vs base: plain final MSE (vs pooled truth) AND unbiased cross-half U = mean((y-A)(y-B)) at L15."""
import json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path(__file__).resolve().parent; R = W/'res'; F = W.parents[2]/'analyst/fresh'
base = sys.argv[1] if len(sys.argv) > 1 else 'base'
halves = {}
def U(seed, y15):
    if seed not in halves:
        with np.load(F/f'truth_seed{seed}.npz') as z: halves[seed] = (z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64))
    A, B = halves[seed]; return float(np.mean((y15 - A)*(y15 - B)))
rows = defaultdict(dict)
for p in R.glob('*.json'):
    if p.name.endswith('_guard.json'): continue
    d = json.loads(p.read_text()); tag = p.stem.rsplit('_', 1)[0]
    with np.load(str(p).replace('.json', '_pred.npz')) as z: y15 = z['y'][-1].astype(np.float64)
    d['U'] = U(d['seed'], y15); rows[tag][d['seed']] = d
B = rows.get(base, {})
print(f"{'arm':<12}{'n':>3}{'dMSE%':>8}{'dU%':>8}{'  U per-seed':<40}{'dbill%':>8} lower")
out = {}
for tag in sorted(rows):
    if tag == base: continue
    seeds = sorted(s for s in rows[tag] if s in B)
    if not seeds: continue
    dm = [100*math.log(rows[tag][s]['mse_final']/B[s]['mse_final']) for s in seeds]
    du = [100*math.log(rows[tag][s]['U']/B[s]['U']) for s in seeds]
    db = [100*math.log(rows[tag][s]['flops']/B[s]['flops']) for s in seeds]
    m = lambda a: sum(a)/len(a)
    out[tag] = dict(n=len(seeds), dMSE=m(dm), dU=m(du), dbill=m(db), U_per_seed=du)
    print(f"{tag:<12}{len(seeds):>3}{m(dm):>8.3f}{m(du):>8.3f}  {' '.join(f'{x:+.2f}' for x in du):<38}{m(db):>8.3f} {sum(x<0 for x in du)}/{len(du)}")
(W/'analysis_latest.json').write_text(json.dumps(out, indent=1))
