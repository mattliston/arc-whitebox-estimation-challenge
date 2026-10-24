"""[opus-kinkdecomp] per variant s: D_s = v14_s - v14_base on the base's kink units (|a14| < 1), dv = v_true - v14_base (relative to v14_base);
pooled over the store ranks: corr(dv, D_s), best scale beta = <dv,D>/<D,D>, share = 1 - |dv - beta D|^2 / |dv|^2 (no intercept), size of D vs dv."""
import sys, numpy as np
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
K = O / 'lead/runs/opus_kinkoracle_v1'
R = (16, 35, 150, 200, 250, 259, 277, 287)
base, truth = {}, {}
for r in R:
    d = np.load(K / f'res_dump_{r}_dump.npz'); A = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz'); B = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz')
    v0 = d['pre_v14']; a = d['mu14'] / np.sqrt(v0); k = np.abs(a) < 1.0
    base[r] = (v0, k); truth[r] = 0.5 * (A['mc_pre_v'][14] + B['mc_pre_v'][14])
tags = sys.argv[1:] or sorted({p.name.split('_')[2] for p in K.glob('res_d_*_dump.npz')})
for t in tags:
    DV, DS, per = [], [], []
    for r in R:
        p = K / f'res_d_{t}_{r}_dump.npz'
        if not p.exists(): continue
        v0, k = base[r]; vs = np.load(p)['pre_v14']
        dv = ((truth[r] - v0) / v0)[k]; ds = ((vs - v0) / v0)[k]
        DV.append(dv); DS.append(ds); per.append((r, float(np.corrcoef(dv, ds)[0, 1]) if ds.std() > 0 else float('nan')))
    if not DV: continue
    dv, ds = np.concatenate(DV), np.concatenate(DS)
    beta = float(dv @ ds / (ds @ ds)) if ds @ ds > 0 else float('nan')
    share = 1 - np.sum((dv - beta * ds) ** 2) / np.sum(dv ** 2) if ds @ ds > 0 else float('nan')
    print(f'{t:10s} ranks {len(per)} corr {np.corrcoef(dv, ds)[0,1]:+.3f} beta {beta:+.2f} share {share:+.3f} | sd(D) {100*ds.std():.4f}% vs sd(dv) {100*dv.std():.4f}% | per-rank corr ' + ' '.join(f'{r}:{c:+.2f}' for r, c in per))
