#!/usr/bin/env python3
"""TAG analyst S53 per-network dial gradients x kink features (lead 10-13 00:5xZ): reads O/trainer/cert/RAD/S53/pernet_grads_v1.json (the trainer's
per-network dial gradients on the 64 A36 networks; its layout is read generically: every numeric leaf under a family / band becomes one series keyed by
seed) and O/analyst/hard_nets_featA36_v1.json (kink and dead fractions at L8-11 / L12-15 and, where already computed, lambda_2 / lambda_1 at L15),
correlates every gradient series with each feature over the shared seeds (Pearson r, Spearman rho) and prints ONE TEAM_LOG line with the lead's rule:
|r| >= 0.4 = a kink-gated dial candidate; nothing >= 0.3 = the lead closes. With n = 64 one test's SE is about 0.125, and many tests are made, so a
single |r| near 0.3 is expected by chance; the line says how many series cross each bar.
usage: grads_kink_corr_v1.py   -> the line on stdout (appended to TEAM_LOG by the waiter), grads_kink_corr_v1.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json, re, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent
G = json.loads((O / 'trainer/cert/RAD/S53/pernet_grads_v1.json').read_text()); F = {int(k): v for k, v in json.loads((HERE / 'hard_nets_featA36_v1.json').read_text()).items()}
isseed = lambda k: bool(re.fullmatch(r'7\d{6}', str(k)))
series = {}


def leaves(x, name):
    if isinstance(x, bool): return
    if isinstance(x, (int, float)): yield name, float(x)
    elif isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f'{name}.{k}' if name else str(k))
    elif isinstance(x, list) and all(isinstance(v, (int, float)) for v in x):
        for i, v in enumerate(x): yield f'{name}[{i}]', float(v)


def add(seed, name, val): series.setdefault(name, {})[int(seed)] = val


def walk(d, prefix=''):
    if isinstance(d, dict):
        if d and all(isseed(k) for k in d):                                  # {seed: {...}}
            for s, v in d.items():
                for n, val in leaves(v, prefix): add(s, n, val)
            return
        if 'seeds' in d:                                                     # {'seeds': [...], fam: [...]}
            ss = d['seeds']
            for k, v in d.items():
                if k != 'seeds' and isinstance(v, list) and len(v) == len(ss):
                    for s, x in zip(ss, v):
                        for n, val in leaves(x, f'{prefix}{k}'): add(s, n, val)
                elif k != 'seeds': walk(v, f'{prefix}{k}.')
            return
        for k, v in d.items():
            if isinstance(v, dict) and v and all(isseed(s) for s in v):      # {fam: {seed: ...}}
                for s, x in v.items():
                    for n, val in leaves(x, f'{prefix}{k}'): add(s, n, val)
            else: walk(v, f'{prefix}{k}.')
    elif isinstance(d, list) and d and all(isinstance(r, dict) and 'seed' in r for r in d):
        for r in d:
            for n, val in leaves({k: v for k, v in r.items() if k != 'seed'}, prefix.rstrip('.')): add(r['seed'], n, val)


walk(G)
feats = ['kink 12-15', 'kink 8-11', 'dead 12-15', 'dead 8-11', 'lambda_2 / lambda_1 (pre) L15']
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
res = []
for name, d in series.items():
    for f in feats:
        ss = [s for s in d if s in F and f in F[s]]
        if len(ss) < 20: continue
        x = np.array([F[s][f] for s in ss]); y = np.array([d[s] for s in ss])
        if y.std() == 0 or x.std() == 0: continue
        res.append((name, f, len(ss), float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rank(x), rank(y))[0, 1])))
res.sort(key=lambda z: -abs(z[3]))
t = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
n4, n3 = sum(abs(r[3]) >= 0.4 for r in res), sum(abs(r[3]) >= 0.3 for r in res)
top = '; '.join(f"{n} x {f}: r {r_:+.2f} rho {rho:+.2f} (n {k})" for n, f, k, r_, rho in res[:8])
verdict = (f'{n4} series x feature at |r| >= 0.4: kink-gated dial CANDIDATE(S)' if n4 else (f'none >= 0.4, {n3} at >= 0.3 (about {0.017 * len(res):.0f} expected by chance)' if n3 else 'nothing >= 0.3: the kink lead CLOSES by the rule'))
print(f"[analyst] {t} S53 PER-NETWORK DIAL GRADIENTS x KINK FEATURES (lead 00:5xZ; O/analyst/grads_kink_corr_v1.py, .json; trainer/cert/RAD/S53/pernet_grads_v1.json x O/analyst/hard_nets_featA36_v1.json): {len(series)} gradient series x {len(feats)} features, {len(res)} tests; strongest: {top}. VERDICT: {verdict}.")
(HERE / 'grads_kink_corr_v1.json').write_text(json.dumps(dict(n_series=len(series), tests=[dict(series=n, feature=f, n=k, r=r_, rho=rho) for n, f, k, r_, rho in res]), indent=1))
