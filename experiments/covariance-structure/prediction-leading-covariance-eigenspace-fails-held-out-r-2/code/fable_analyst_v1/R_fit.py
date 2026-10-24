#!/usr/bin/env python3
"""TAG fable-analyst (AN-28): cross-net predictability of the one-sided subspace error R = P dC P U_k (R_data.py) from the card's plane-vector
products. Per layer (and per band L2-6 / L7-10 / L11-14): ridge fit on one net, R2 on the others; noise ceiling = 1 - (|noise R| / |R|)^2.
usage: R_fit.py <k 30|100>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; k = int(sys.argv[1])
NETS = (35, 100, 37, 39); D = {r: np.load(HERE / f'R_data_r{r}.npz') for r in NETS}; names = list(D[35]['names'])
def st(r, l): return D[r][f'G_{l}_{k}'], D[r][f'b_{l}_{k}'], float(D[r][f'yy_{l}_{k}'])
def fit(G, b, lam=1e-6):
    s = np.sqrt(np.diag(G)) + 1e-300; return np.linalg.solve(G / np.outer(s, s) + lam * np.eye(len(b)), b / s) / s
def r2(B, G, b, yy): return 1 - (yy - 2 * B @ b + B @ G @ B) / yy
print(f'== k = {k}: per-layer cross-net R2 (train -> mean over the other 3 nets), in-sample, noise ceiling ==')
for l in range(2, 15):
    row = dict(layer=l)
    for tr in (35, 100):
        B = fit(*st(tr, l)[:2]); row[f'in{tr}'] = round(r2(B, *st(tr, l)), 3)
        row[f'x{tr}'] = round(float(np.mean([r2(B, *st(te, l)) for te in NETS if te != tr])), 3)
    ceil = [1 - (float(D[r][f'N_norm{k}_{l}_noise']) / float(D[r][f'R_norm{k}_{l}_noise'])) ** 2 for r in NETS]
    row['signal_share'] = round(float(np.mean(ceil)), 3)
    print(json.dumps(row))
print('== bands: pooled fit on train net, R2 on the other nets per layer ==')
for band in ((2, 6), (7, 10), (11, 14), (2, 14)):
    Ls = range(band[0], band[1] + 1)
    for tr in (35, 100):
        B = fit(sum(st(tr, l)[0] for l in Ls), sum(st(tr, l)[1] for l in Ls))
        te = {r: round(1 - sum(st(r, l)[2] - 2 * B @ st(r, l)[1] + B @ st(r, l)[0] @ B for l in Ls) / sum(st(r, l)[2] for l in Ls), 3) for r in NETS if r != tr}
        print(json.dumps(dict(band=band, train=tr, test=te, coef_x_rms={n_: float('%.3g' % (B[i] * np.sqrt(st(tr, band[1])[0][i, i]))) for i, n_ in enumerate(names)})))
print('== single features (band L7-14, train 35 -> test 100) ==')
Ls = range(7, 15); G = {r: sum(st(r, l)[0] for l in Ls) for r in NETS}; b = {r: sum(st(r, l)[1] for l in Ls) for r in NETS}; yy = {r: sum(st(r, l)[2] for l in Ls) for r in NETS}
print(json.dumps({n_: dict(in35=round(b[35][i] ** 2 / G[35][i, i] / yy[35], 4), x100=round(1 - (yy[100] - 2 * (b[35][i] / G[35][i, i]) * b[100][i] + (b[35][i] / G[35][i, i]) ** 2 * G[100][i, i]) / yy[100], 4)) for i, n_ in enumerate(names)}))
