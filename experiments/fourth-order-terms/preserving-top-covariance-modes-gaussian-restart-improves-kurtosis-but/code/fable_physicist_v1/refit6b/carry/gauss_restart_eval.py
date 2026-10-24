#!/usr/bin/env python3
"""TAG fable-physicist (CR-19 eval): score gauss_restart_mc.py output against the independent 32M per-unit truth (indep32M draw).
Per restart layer l0 and target layer L: relative rms error of the restart's per-unit k3/k4 vs truth, raw and noise-debiased
(restart noise from the A/B halves: ||kA-kB||^2/4; truth noise scaled from it by sample count). 'hyb' = the restart plus the isotropic
form fitted per layer (truth ~ a*restart + b*iso + c; iso = v^2 for k4, m*v = alpha*v^1.5 for k3: the mean-driven skew, 88-91% of t3 spread), i.e. a late construction that keeps an isotropic
term for everything born before l0. 'iso' = the isotropic form alone (b*v^p + c; the card's k4 shape with a free intercept).
The spec (analyst section 35): k3 within ~1.5%, k4 within ~7% relative rms at L12-14; break-even ~2.6% / ~11%.
usage: gauss_restart_eval.py <rank> <npz> [<npz> ...]   -> prints a table; writes <npz stem>_eval.json"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]
rank = int(sys.argv[1])
T = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rank:04d}_indep32M.npz'); PT = float(T['P'])
tv, tm, tk = T['mc_pre_v'], T['mc_pre_m'], {'k3': T['mc_pre_k3'], 'k4': T['mc_pre_k4']}
def fit_rel(y, X, noise2=0.0):
    X = np.column_stack([np.ones(len(y))] + X); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
    e2 = float(r @ r); return np.sqrt(e2) / np.linalg.norm(y), np.sqrt(max(e2 - noise2, 0)) / np.linalg.norm(y), c
out = {}
for f in sys.argv[2:]:
    R = np.load(f); l0s = sorted({int(k.split('_')[-1]) for k in R.files if k.startswith('N_')})
    for l0 in l0s:
        NR = float(R[f'N_{l0}'])
        for L in range(l0 + 1, 16):
            row = {}
            for key in ('k3', 'k4'):
                a, b = R[f'{key}_A_{l0}'][L], R[f'{key}_B_{l0}'][L]; kr = (a + b) / 2; kt = tk[key][L]
                nR2 = float(((a - b) ** 2).sum()) / 4; nT2 = nR2 * NR / PT; nt = np.linalg.norm(kt)
                d2 = float(((kr - kt) ** 2).sum())
                raw, deb = np.sqrt(d2) / nt, np.sqrt(max(d2 - nR2 - nT2, 0)) / nt
                vp = np.clip(tv[L], 0, None) ** 2 if key == 'k4' else tm[L] * np.clip(tv[L], 0, None)   # k4 ~ v^2; k3 ~ alpha v^1.5 = m v
                _, iso, _ = fit_rel(kt, [vp], nT2)
                _, hyb, c = fit_rel(kt, [kr, vp], nT2 + nR2)
                hyb = np.sqrt(max((hyb * nt) ** 2 - (c[1] ** 2 - 1) * nR2, 0)) / nt if c[1] > 1 else hyb   # noise enters as c1^2 nR2
                row[key] = dict(raw=raw, deb=deb, noise=np.sqrt(nR2 + nT2) / nt, iso=iso, hyb=hyb, hyb_a=float(c[1]),
                                scale=float(kr @ kt / (kt @ kt)))
            out[f'{l0}->{L}'] = row
    tag = Path(f).stem
print(f'net {rank}: relative rms error vs the 32M truth (noise-debiased; noise = combined MC noise)')
print(' l0->L |  k4: restart  hybrid   iso  noise  scale |  k3: restart  hybrid   iso  noise  scale')
for k, row in out.items():
    l0, L = map(int, k.split('->'))
    if L < 12: continue
    a, b = row['k4'], row['k3']
    print(f'{k:>6} |  {a["deb"]:.3f}  {a["hyb"]:.3f}  {a["iso"]:.3f}  {a["noise"]:.3f}  {a["scale"]:.2f} |  {b["deb"]:.3f}  {b["hyb"]:.3f}  '
          f'{b["iso"]:.3f}  {b["noise"]:.3f}  {b["scale"]:.2f}')
Path(str(Path(sys.argv[2]).with_suffix('')) + '_eval.json').write_text(json.dumps(out, indent=1))
