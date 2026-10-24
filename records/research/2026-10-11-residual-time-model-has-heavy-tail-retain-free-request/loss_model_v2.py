"""TAG opus-release loss model v2 (2026-10-11 06:40 UTC, lead's refit after #335329 lost network 14 to the residual cap). No session.
Two readings of P(a private run of N networks loses a network on the 0.4 s residual cap):
  NETWORK model (new): a network's residual = the run's median x its factor x; the factors of OUR 23 graded leader-sized runs
    (O/board_evals + #335329: 2,300 networks; x = residual / run median) get a generalized-Pareto tail above u (scipy MLE);
    P_N = 1 - (1 - S(0.4 / median))^N. Readings u = 1.10 (central) and u = 1.12 (fatter: the one 1.313 event weighs more).
    The 105 public pages of other teams (O/analyst/board_pages) are NOT used for the tail: their factors include per-network
    work that differs by design (p99 1.28, max 1.69), ours vary by +-2% in requests.
  RUN model (loss_risk.py's): the run's worst = mean x our usual worst / mean x the run's excursion (729 leader runs, a19);
    usual = median over our seven S44-S46x graded runs (#335329's 1.288 included).
Price of a request: median 7.161 us (our S44-S46 graded runs), slow 7.267 us (the slowest of them); median / mean residual 0.99.
usage: loss_model_v2.py   (prints the lead's scenario table)"""
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy.stats import genpareto

R = Path(__file__).resolve().parent; O = R.parent
x = np.array(json.loads((R/'runs'/'net_factors_20261011.json').read_text())['ours'])
fits = {}
for u in (1.10, 1.12):
    exc = x[x > u] - u
    c, _, scale = genpareto.fit(exc, floc=0)
    fits[u] = (len(exc)/len(x), c, scale)
S = lambda t, u: fits[u][0]*genpareto.sf(t - u, fits[u][1], 0, fits[u][2])
PL = json.loads((O/'analyst'/'a19_page_list.json').read_text())
E = {'local': [v[1] for v in PL['excess'].values()], 'week': [v[0] for v in PL['excess'].values()]}
USUAL = st.median([1.098, 1.183, 1.142, 1.134, 1.123, 1.139, 1.288])     # S44 s/t, S45 s/t, S46 s/t, S46x twin #335329
run_p = lambda mean, rd: sum(1 for e in E[rd] if e > 0.4/(mean*USUAL))/len(E[rd])
print(f"our network factors: {len(x)} networks of 23 runs; GPD u=1.10: rate {fits[1.10][0]:.4f}, shape {fits[1.10][1]:+.2f}, scale "
      f"{fits[1.10][2]:.4f}; u=1.12: shape {fits[1.12][1]:+.2f}; empirical: 1 network of {len(x)} above 1.30 (#335329 n14, 1.313); usual {USUAL:.3f}")
SC = [('S46 safe #335259', 42970, 0.0), ('S47s at S46 counts', 42970, 0.0), ('S47s - 1,000 requests', 41970, 0.0),
      ('S47s - 2,000 requests', 40970, 0.0), ('S47s - 30 ms own Python', 42970, 0.030)]
print(f"{'file':26s} {'hour':6s} {'median s':>8s} {'cap/med':>7s}   network model u1.10 (u1.12): N=100 / N=300        run model local-week: N=100 / N=300")
for name, req, cut in SC:
    for hour, price in (('median', 7.161e-6), ('slow', 7.267e-6)):
        mean = req*price - cut; med = 0.99*mean; t = 0.4/med
        pn = {u: [1 - (1 - S(t, u))**n for n in (100, 300)] for u in (1.10, 1.12)}
        pr = {rd: run_p(mean, rd) for rd in ('local', 'week')}
        p300 = {rd: 1 - (1 - pr[rd])**3 for rd in pr}
        print(f"{name:26s} {hour:6s} {med:8.4f} {t:7.3f}   {100*pn[1.10][0]:4.1f}% ({100*pn[1.12][0]:4.1f}%) / {100*pn[1.10][1]:4.1f}% ({100*pn[1.12][1]:4.1f}%)"
              f"          {100*pr['local']:4.1f}-{100*pr['week']:4.1f}% / {100*p300['local']:4.1f}-{100*p300['week']:4.1f}%")
