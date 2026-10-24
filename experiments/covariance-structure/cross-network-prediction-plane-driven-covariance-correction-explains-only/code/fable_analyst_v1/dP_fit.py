#!/usr/bin/env python3
"""TAG fable-analyst (AN-23): predictability of dP (plane-driven, mean-orthogonal covariance correction; dP_data.py) from the covariance
reads of cheap candidate planes. Per layer and pooled over layers: ridge LSQ on one net, R2 on the other net (cross-net), both directions;
single-feature cross-net R2; plus dP vs the carried error P dC P at L4/7/10/13 (norm ratio, cosine with -P dC P, SVD energy of dP).
usage: dP_fit.py"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
D = {r: np.load(HERE / f'dP_data_r{r}.npz') for r in (35, 100)}
names = list(D[35]['names']); p = len(names); L = list(range(2, 15))
def st(r, l): d = D[r]; return d[f'G_{l}'], d[f'b_{l}'], float(d[f'yy_{l}'])
def fit(G, b, lam=1e-6):
    s = np.sqrt(np.diag(G)) + 1e-300; Gn = G / np.outer(s, s); bn = b / s
    beta_n = np.linalg.solve(Gn + lam * np.eye(len(b)), bn); return beta_n / s
def r2(beta, G, b, yy): return 1 - (yy - 2 * beta @ b + beta @ G @ beta) / yy
print('== per-layer ridge fit, in-sample and cross-net R2 (fit net -> test net) ==')
rows = []
for l in L:
    G35, b35, y35 = st(35, l); G100, b100, y100 = st(100, l)
    B35 = fit(G35, b35); B100 = fit(G100, b100)
    rows.append(dict(layer=l, in35=round(r2(B35, G35, b35, y35), 3), in100=round(r2(B100, G100, b100, y100), 3),
                     x35to100=round(r2(B35, G100, b100, y100), 3), x100to35=round(r2(B100, G35, b35, y35), 3)))
    print(json.dumps(rows[-1]))
print('== pooled over layers 2-14 (one coefficient set) ==')
for (tr, te) in ((35, 100), (100, 35)):
    G = sum(st(tr, l)[0] for l in L); b = sum(st(tr, l)[1] for l in L); B = fit(G, b)
    tot = sum(st(te, l)[2] for l in L); sse = sum(st(te, l)[2] - 2 * B @ st(te, l)[1] + B @ st(te, l)[0] @ B for l in L)
    per = {l: round(r2(B, *st(te, l)), 3) for l in L}
    print(json.dumps(dict(fit=tr, test=te, pooled_R2=round(1 - sse / tot, 3), per_layer=per)))
    print('   coef x feature-rms:', {n_: float('%.3g' % (B[i] * np.sqrt(st(tr, 8)[0][i, i]))) for i, n_ in enumerate(names)})
print('== single-feature cross-net R2 (fit r35 -> r100, pooled layers) ==')
Gp = {r: sum(st(r, l)[0] for l in L) for r in (35, 100)}; bp = {r: sum(st(r, l)[1] for l in L) for r in (35, 100)}; yp = {r: sum(st(r, l)[2] for l in L) for r in (35, 100)}
single = {}
for i, n_ in enumerate(names):
    bi = bp[35][i] / Gp[35][i, i]
    single[n_] = dict(x35to100=round(1 - (yp[100] - 2 * bi * bp[100][i] + bi * bi * Gp[100][i, i]) / yp[100], 4),
                      in35=round(bp[35][i] ** 2 / Gp[35][i, i] / yp[35], 4))
print(json.dumps(single))
# greedy forward selection on r35, scored cross-net on r100 (pooled)
sel = []; rem = list(range(p))
for step in range(8):
    best = None
    for j in rem:
        idx = sel + [j]; G = Gp[35][np.ix_(idx, idx)]; b = bp[35][idx]; B = fit(G, b)
        sc = 1 - (yp[100] - 2 * B @ bp[100][idx] + B @ Gp[100][np.ix_(idx, idx)] @ B) / yp[100]
        if best is None or sc > best[0]: best = (sc, j)
    sel.append(best[1]); rem.remove(best[1]); print(f'greedy step {step+1}: +{names[best[1]]:10s} cross-net pooled R2 {best[0]:.4f}')
