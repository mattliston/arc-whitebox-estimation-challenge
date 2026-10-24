"""TAG opus-release loss model v3 (2026-10-11 11:40 UTC; v2 + #335340 and #335349 folded in; v2 kept). No session.
P(a private run of N networks loses a network) for the designated pair #335349 (S47e) + #335340 (S47s), each file and both.
RESIDUAL: network factor x = residual / run median over our 25 graded leader-sized runs (O/board_evals, #335329, #335340, #335349:
2,500 networks), generalized-Pareto tail above u = 1.10 (central) and 1.12 (fatter); a file loses when its median residual x x > 0.4 s;
the median is the file's own graded one (its hour) or 1.5% slower. WALL: 2 of 359 public runs with mean wall 85-90 s lost a network on the
120 s limit (analyst a16), per 100 networks; both S47 files ran 86.6-87.6 s mean. Runs of N networks: 1 - (1 - p)^N; the two
designated files run separately, so both lose with the product of their probabilities.
usage: loss_model_v3.py"""
import glob
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy.stats import genpareto

O = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(O/'board_evals'/'eval_*.json'))) + [str(O/'fire'/p) for p in
        ('s46x_twin_20261011/eval_335329.json', 's47s_20261011/eval_335340.json', 's47e_20261011/eval_335349.json')]
xs = []
for f in files:
    res = (lambda z: (z.get('evaluation') or z)['results'])(json.loads(Path(f).read_text()))
    r = [m['telemetry']['residual_wall_time_s'] for m in res.get('per_mlp', []) if (m.get('telemetry') or {}).get('residual_wall_time_s') is not None]
    if len(r) >= 90 and st.median(r) >= 0.15:
        xs += [v/st.median(r) for v in r]
x = np.array(xs)
fits = {}
for u in (1.10, 1.12):
    exc = x[x > u] - u
    c, _, s = genpareto.fit(exc, floc=0)
    fits[u] = (len(exc)/len(x), c, s)
S = lambda t, u: fits[u][0]*genpareto.sf(t - u, fits[u][1], 0, fits[u][2])
WALL100 = 2/359
print(f"{len(x)} networks; above 1.20: {(x > 1.2).sum()}; GPD shape {fits[1.10][1]:+.2f} (u 1.10), {fits[1.12][1]:+.2f} (u 1.12)")
P = {}
for name, med in (('#335349 S47e', 0.3075), ('#335340 S47s', 0.3107)):
    for hour, f in (('own hour', 1.0), ('1.5% slower', 1.015)):
        t = 0.4/(med*f)
        P[name, hour] = {n: {u: 1 - (1 - S(t, u))**n*(1 - WALL100)**(n/100) for u in fits} for n in (100, 300)}
        print(f"{name} {hour}: cap/median {t:.3f}: N=100 {100*P[name, hour][100][1.10]:.1f}% ({100*P[name, hour][100][1.12]:.1f}%), "
              f"N=300 {100*P[name, hour][300][1.10]:.1f}% ({100*P[name, hour][300][1.12]:.1f}%)")
for hour in ('own hour', '1.5% slower'):
    a, b = P['#335349 S47e', hour], P['#335340 S47s', hour]
    print(f"both lose, {hour}: N=100 {100*a[100][1.10]*b[100][1.10]:.2f}% ({100*a[100][1.12]*b[100][1.12]:.2f}%), "
          f"N=300 {100*a[300][1.10]*b[300][1.10]:.2f}% ({100*a[300][1.12]*b[300][1.12]:.2f}%)")
