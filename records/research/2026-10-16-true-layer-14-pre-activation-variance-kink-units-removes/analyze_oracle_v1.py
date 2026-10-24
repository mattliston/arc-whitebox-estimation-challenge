"""[opus-kinkoracle] (1) oracle sizes: per mode, xMSE = 100 log(U_mode / U_ctl) per store rank; (2) if dumps exist: split the L14 kink-unit
variance error into the part propagated from L13's diagonal (sum_j W14[j,i]^2 dVar(h13_j)) and the rest."""
import json, math, statistics as st
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
K = O / 'lead/runs/opus_kinkoracle_v1'
STORE = O.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
res = {}
for p in K.glob('res_*.json'):
    d = json.loads(p.read_text()); res[(d['mode'], d['rank'])] = d
ranks = sorted({r for m, r in res if m == 'ctl'})
for mode in sorted({m for m, _ in res if m not in ('ctl', 'dump')}):
    xs = [(r, 100 * math.log(res[(mode, r)]['U'] / res[('ctl', r)]['U'])) for r in ranks if (mode, r) in res]
    if xs:
        v = [x for _, x in xs]
        print(f"{mode:5s} n {len(v)} xMSE mean {st.mean(v):+.2f}% se {st.stdev(v)/math.sqrt(len(v)) if len(v) > 1 else float('nan'):.2f}  per rank " + ' '.join(f'{r}:{x:+.1f}' for r, x in xs))
for r in ranks:
    p = K / f'res_dump_{r}_dump.npz'
    if not p.exists(): continue
    d = np.load(p); A = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz'); B = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz')
    W = np.load(STORE / f'rows/observer_fit/{r:04d}/weights.npy')[14].astype(np.float64)
    tv13 = 0.5 * (A['mc_post_var'][13] + B['mc_post_var'][13]); tv14 = 0.5 * (A['mc_pre_v'][14] + B['mc_pre_v'][14])
    v14, a = d['pre_v14'], d['mu14'] / np.sqrt(np.maximum(d['pre_v14'], 1e-10))
    k = np.abs(a) < 1.0
    dv = tv14 - v14; dv_diag = (W * W).T @ (tv13 - d['post_v13'])
    rel = dv / v14; reld = dv_diag / v14
    c = np.corrcoef(rel[k], reld[k])[0, 1]; slope = float(np.dot(rel[k], reld[k]) / np.dot(reld[k], reld[k]))
    print(f'rank {r}: kink {int(k.sum())}  dv/v kink mean {100*rel[k].mean():+.3f}% sd {100*rel[k].std():.3f}%  | diag-propagated mean {100*reld[k].mean():+.3f}% sd {100*reld[k].std():.3f}%  corr {c:+.2f} slope {slope:+.2f}  | post_v13 rel err mean {100*np.mean((tv13-d["post_v13"])/np.maximum(tv13,1e-12)):+.3f}%')
