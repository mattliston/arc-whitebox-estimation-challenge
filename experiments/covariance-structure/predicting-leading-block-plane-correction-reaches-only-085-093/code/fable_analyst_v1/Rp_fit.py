#!/usr/bin/env python3
"""TAG fable-analyst (AN-31): cross-net predictability of the one-sided top-k block R_k = P D P U_k of the plane-driven correction (Rp_data.py),
per layer and per band (L2-6 / L7-10 / L11-14 / L2-14); ridge on one net, R2 on the other (35 <-> 100); noise share from the independent 8M
draw (r35, L5/9/12). usage: Rp_fit.py"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = {r: np.load(HERE / f'Rp_data_r{r}.npz') for r in (35, 100)}; names = list(D[35]['names'])
def st(r, l, k): return D[r][f'G_{l}_{k}'], D[r][f'b_{l}_{k}'], float(D[r][f'yy_{l}_{k}'])
def fit(G, b, lam=1e-4):
    s = np.sqrt(np.diag(G)) + 1e-300; return np.linalg.solve(G / np.outer(s, s) + lam * np.eye(len(b)), b / s) / s
for k in (10, 30):
    print(f'== k = {k}: per-layer R2 (in35 / in100 / x35->100 / x100->35); noise share (r35) ==')
    for l in range(2, 15):
        B35 = fit(*st(35, l, k)[:2]); B100 = fit(*st(100, l, k)[:2])
        r2 = lambda B, r: 1 - (st(r, l, k)[2] - 2 * B @ st(r, l, k)[1] + B @ st(r, l, k)[0] @ B) / st(r, l, k)[2]
        ns = D[35][f'noise_share_{l}_{k}'] if f'noise_share_{l}_{k}' in D[35].files else None
        print(json.dumps(dict(layer=l, in35=round(r2(B35, 35), 3), in100=round(r2(B100, 100), 3), x35to100=round(r2(B35, 100), 3), x100to35=round(r2(B100, 35), 3),
                              noise_share=(round(float(ns), 3) if ns is not None else None))))
    for band in ((2, 6), (7, 10), (11, 14), (2, 14)):
        Ls = range(band[0], band[1] + 1); row = dict(k=k, band=band)
        for tr, te in ((35, 100), (100, 35)):
            B = fit(sum(st(tr, l, k)[0] for l in Ls), sum(st(tr, l, k)[1] for l in Ls))
            sse = sum(st(te, l, k)[2] - 2 * B @ st(te, l, k)[1] + B @ st(te, l, k)[0] @ B for l in Ls); tot = sum(st(te, l, k)[2] for l in Ls)
            row[f'x{tr}to{te}'] = round(1 - sse / tot, 4)
            if tr == 35:
                top = sorted(((abs(B[i]) * np.sqrt(sum(st(tr, l, k)[0][i, i] for l in Ls)), n_) for i, n_ in enumerate(names)), reverse=True)[:5]
                row['top_terms_35'] = [n_ for _, n_ in top]
        print(json.dumps(row))
