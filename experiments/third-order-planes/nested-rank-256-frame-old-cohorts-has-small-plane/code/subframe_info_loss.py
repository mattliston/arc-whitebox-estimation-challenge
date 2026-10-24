#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-14): information loss of confining the OLDEST cohorts (age >= A) of the shared tier to a nested
rank-r sub-frame (shared across those cohorts, recomputed per layer from their own columns): relative change of the consumed (2,1)
plane (response x live) and the diagonal k3 read, per layer, on the full bank dump.  usage: subframe_info_loss.py <bank_dir> <A> <ranks e.g. 128,192,256>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); A = int(sys.argv[2]); ranks = [int(x) for x in sys.argv[3].split(",")]
out = {}
for l in range(A, 15):
    z = np.load(D / f"layer_{l:02d}.npz"); P = z["pp"].astype(np.float64); w = z["lam"].astype(np.float64)
    oc = int(z["old_columns"]); coh = z["cohorts"]; rows = z["response"]; cols = z["live"]
    sub = [b for b, (st, wd) in enumerate(coh) if l - b >= A and wd > 0 and st + wd <= oc]
    idx = np.concatenate([np.arange(st, st + wd) for b, (st, wd) in enumerate(coh) if b in sub])
    def plane(Pm): return (Pm[rows] ** 2 * w[None, :]) @ Pm[cols].T
    def diag(Pm): return (Pm ** 3 * w[None, :]).sum(1)
    full_pl, full_dg = plane(P), diag(P)
    Psub = P[:, idx]; U, S, Vt = np.linalg.svd(Psub, full_matrices=False); en = np.cumsum(S ** 2) / (S ** 2).sum()
    rec = dict(layer=l, sub_cohorts=sub, sub_columns=int(len(idx)), old_columns=oc, energy_at={r: float(en[r - 1]) for r in ranks if r <= len(en)})
    for r in ranks:
        Pr = P.copy(); Pr[:, idx] = (U[:, :r] * S[:r]) @ Vt[:r]
        pl, dg = plane(Pr), diag(Pr)
        rec[f"rank{r}"] = dict(plane_rel_err=float(np.linalg.norm(pl - full_pl) / np.linalg.norm(full_pl)), diag_rel_err=float(np.linalg.norm(dg - full_dg) / np.linalg.norm(full_dg)))
    out[l] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(D / f"subframe_info_loss_A{A}.json", "w"), indent=1)
