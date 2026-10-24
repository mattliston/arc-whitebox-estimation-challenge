#!/usr/bin/env python3
"""TAG fable-physicist (OT-4): frame truncation of the WHOLE old tier to rank r' (best weighted frame: top-r' left singular vectors of
P_old diag|lam|), with the lost content's gain-type part recovered into a Sym(C x x) term (x = LSQ fit of the lost plane incl. diagonal,
the b-vector family the card transports exactly). Static errors on the response x live block and on the diagonal, relative to the
TOTAL consumed plane / diagonal (old + young, as the card reads them) and to the old plane. numpy only.
usage: trunc_gain_recover.py <layers> <ranks>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[2] / 'fable_oldtier_economics_v1' / 'bank_r35'
def symfit(K, C):
    v = np.diag(C).copy(); n = len(v)
    N = (v @ v) * np.eye(n) + 2 * C * v[None, :] + 2 * v[:, None] * C + 4 * np.diag((C * C).sum(1))
    x = np.linalg.solve(N, K.T @ v + 2 * (K * C).sum(1))
    return np.outer(v, x) + 2 * x[:, None] * C
for l in [int(x) for x in sys.argv[1].split(',')]:
    z = np.load(D / f'layer_{l:02d}.npz'); s = np.load(D / f'state_{l:02d}.npz')
    no = int(z['old_columns']); P = z['pp'][:, :no].astype(np.float64); lam = z['lam'][:no].astype(np.float64)
    Pa = z['pp'].astype(np.float64); la = z['lam'].astype(np.float64)
    resp, live = z['response'], z['live']; C = s['Cpre'].astype(np.float64)
    plane = lambda PP, ll: (PP * PP) @ (ll[:, None] * PP.T)
    Kold = plane(P, lam); Ktot = plane(Pa, la)
    blk = lambda M: M[np.ix_(resp, live)]
    ntot, ntotd = np.linalg.norm(blk(Ktot)), np.linalg.norm(np.diag(Ktot))
    nold = np.linalg.norm(blk(Kold))
    Uw, Sw, _ = np.linalg.svd(P * np.sqrt(np.abs(lam))[None, :], full_matrices=False)
    for rr in [int(x) for x in sys.argv[2].split(',')]:
        Q = Uw[:, :rr]; Pt = Q @ (Q.T @ P)
        Kt = plane(Pt, lam); Dl = Kold - Kt
        S = symfit(Dl, C); R = Dl - S
        rec = dict(layer=l, rank=rr, old_cols=no,
                   trunc_plane_rel_total=round(float(np.linalg.norm(blk(Dl)) / ntot), 5), trunc_diag_rel_total=round(float(np.linalg.norm(np.diag(Dl)) / ntotd), 5),
                   recovered_plane_rel_total=round(float(np.linalg.norm(blk(R)) / ntot), 5), recovered_diag_rel_total=round(float(np.linalg.norm(np.diag(R)) / ntotd), 5),
                   trunc_plane_rel_old=round(float(np.linalg.norm(blk(Dl)) / nold), 5))
        print(json.dumps(rec), flush=True)
