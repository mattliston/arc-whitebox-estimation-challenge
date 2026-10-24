#!/usr/bin/env python3
"""TAG opus-reexam: what is the partner vector of the MEAN-DIRECTION mode? q1 = (I - P) R u_1 from the oracle R (physicist), per layer, against
free vectors of the card orthogonalised the same way: the trace vector of the bank's third cumulant tau_i = sum_r lam_r pp_ri |pp_r|^2, the pushed b carrier
bpre, C bpre-like forms are not available here; also the carried q1 itself. usage: ncar_mode1.py <rank> <arm>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; rank = int(sys.argv[1]); arm = sys.argv[2]
D = np.load(HERE / f'ncar_dump_rank{rank:04d}_{arm}.npz'); R = np.load(HERE / f'physicist/t19_oracle_rank{rank:04d}.npz')
for l in sorted(int(k[2:]) for k in D.keys() if k.startswith('Q_')):
    U, Q = D[f'U_{l}'].astype(np.float64), D[f'Q_{l}'].astype(np.float64); Rt = R[f'R_L{l}'].astype(np.float64); Rt = (Rt + Rt.T) / 2
    pr = lambda x: x - U @ (U.T @ x); q1 = pr(Rt @ U[:, 0]); c = lambda a, b: round(float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)), 3)
    tau, bpre, mu = pr(D[f'tau_{l}']), pr(D[f'bpre_{l}']), D[f'mu_{l}']
    X = np.stack([tau, bpre], 1); co = np.linalg.lstsq(X, q1, rcond=None)[0]; fit = X @ co
    print(json.dumps(dict(layer=l, cos_u1_mu=c(U[:, 0], mu), q1_norm=float('%.3g' % np.linalg.norm(q1)), cos_q1_tau=c(q1, tau), cos_q1_bpre=c(q1, bpre), cos_tau_bpre=c(tau, bpre), cos_q1_carried=c(q1, Q[:, 0]),
                          R2_tau_bpre=round(float(1 - ((q1 - fit) ** 2).sum() / (q1 ** 2).sum()), 3), coef=[float('%.3g' % x) for x in co])))
