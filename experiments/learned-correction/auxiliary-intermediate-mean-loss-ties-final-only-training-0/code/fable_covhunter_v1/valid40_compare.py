#!/usr/bin/env python3
"""TAG fable-covhunter (CH-46): grade a VALID40 eval-only file (refit6b eval_n40.json) against baselines, per net (paired).
Pooled = sum(cand)/sum(base) - 1 with a 95% paired-bootstrap CI over nets (20k resamples, seed 0); per-net mean of cand/base - 1
with a t-interval (df = n-1); wins = nets where cand < base. Baselines: 'parity:<jsonl>' (analyst rows, key torch_s6b) or an
eval_n40.json. usage: valid40_compare.py <cand eval_n40.json> <baseline> [<baseline> ...]"""
import sys, json, numpy as np
def load(spec):
    if spec.startswith('parity:'):
        return {str(json.loads(l)['rank']): json.loads(l)['torch_s6b'] for l in open(spec[7:]) if l.strip()}
    return {str(k): v for k, v in json.load(open(spec))['valid'].items()}
cand = load(sys.argv[1]); rng = np.random.default_rng(0)
for spec in sys.argv[2:]:
    base = load(spec); nets = [r for r in cand if r in base]; c = np.array([cand[r] for r in nets]); b = np.array([base[r] for r in nets])
    pooled = c.sum() / b.sum() - 1
    idx = rng.integers(0, len(nets), size=(20000, len(nets))); boot = c[idx].sum(1) / b[idx].sum(1) - 1
    lo, hi = np.percentile(boot, [2.5, 97.5]); rel = c / b - 1; se = rel.std(ddof=1) / np.sqrt(len(rel))
    t = {39: 2.0227, 15: 2.1314}.get(len(rel) - 1, 1.96)
    print(f'vs {spec}: n {len(nets)}  cand mean {c.mean():.5e}  base mean {b.mean():.5e}')
    print(f'   pooled {pooled:+.3%} [95% boot CI {lo:+.3%}, {hi:+.3%}]  per-net mean {rel.mean():+.3%} [t CI {rel.mean()-t*se:+.3%}, '
          f'{rel.mean()+t*se:+.3%}]  wins {int((rel < 0).sum())}/{len(rel)}  median {np.median(rel):+.3%}')
