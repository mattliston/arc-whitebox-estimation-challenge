#!/usr/bin/env python3
"""TAG opus-reexam NCAR open-loop check: the carrier's partner vectors (ncar_dump_rank<r>_<arm>.npz: U_l, Q_l) against the Monte Carlo truth
(physicist's t19_oracle_rank<r>.npz: R_L<l> = network-specific part of Ntilde at layer l). With the CARD's eigenvectors U: q_true_m = (I - U U') R u_m.
Per layer: cosine and size ratio per mode, and the share of the true one-sided block's energy the carrier holds (2 q.c - c.c) / q.q,
for all modes and for modes 2..K. usage: ncar_compare.py <rank> <arm>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; rank = int(sys.argv[1]); arm = sys.argv[2]
D = np.load(HERE / f'ncar_dump_rank{rank:04d}_{arm}.npz'); R = np.load(HERE / f'physicist/t19_oracle_rank{rank:04d}.npz')
layers = sorted(int(k[2:]) for k in D.keys() if k.startswith('Q_'))
for l in layers:
    if f'R_L{l}' not in R: continue
    U, Q = D[f'U_{l}'].astype(np.float64), D[f'Q_{l}'].astype(np.float64); Rt = R[f'R_L{l}'].astype(np.float64); Rt = (Rt + Rt.T) / 2
    qt = Rt @ U; qt = qt - U @ (U.T @ qt); K = U.shape[1]
    cos = [round(float(Q[:, m] @ qt[:, m] / max(np.linalg.norm(Q[:, m]) * np.linalg.norm(qt[:, m]), 1e-300)), 3) for m in range(K)]
    ratio = [round(float(np.linalg.norm(Q[:, m]) / max(np.linalg.norm(qt[:, m]), 1e-300)), 2) for m in range(K)]
    sh = lambda sl: round(float((2 * (Q[:, sl] * qt[:, sl]).sum() - (Q[:, sl] ** 2).sum()) / (qt[:, sl] ** 2).sum()), 3)
    best = lambda sl: round(float(((Q[:, sl] * qt[:, sl]).sum()) ** 2 / ((Q[:, sl] ** 2).sum() * (qt[:, sl] ** 2).sum())), 3)
    blk = float(2 * (qt ** 2).sum() / (Rt ** 2).sum())
    print(json.dumps(dict(layer=l, one_sided_block_share_of_R=round(blk, 3), cos=cos, size_ratio=ratio, share_all=sh(slice(0, K)), share_modes_2up=sh(slice(1, K)),
                          share_at_best_scale_all=best(slice(0, K)), share_at_best_scale_2up=best(slice(1, K)))))
