#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-7): PURE INFORMATION LOSS of importance-pruned births, without closure feedback. On the full
(unpruned) bank dump of one net, zero the signed column pairs of the lowest-importance births of every cohort (importance = born
tensor Frobenius norm, dumped by birth_prune.py at keep=1.0) and measure, per layer, the relative change of the consumed (2,1)
plane (response x live) and of the diagonal k3 read, for the old part and for the whole CP bank.  usage: prune_info_loss.py <bank_dir> <keeps>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); keeps = [float(x) for x in sys.argv[2].split(",")]
imp = {b: np.load(D / f"imp_{b:02d}.npz") for b in range(15) if (D / f"imp_{b:02d}.npz").exists()}
out = {}
for l in range(5, 15):
    z = np.load(D / f"layer_{l:02d}.npz"); P = z["pp"].astype(np.float64); w = z["lam"].astype(np.float64)
    oc = int(z["old_columns"]); coh = z["cohorts"]; rows = z["response"]; cols = z["live"]
    def plane(mask):
        wm = w * mask; return (P[rows] ** 2 * wm[None, :]) @ P[cols].T
    def diag(mask):
        wm = w * mask; return (P ** 3 * wm[None, :]).sum(1)
    full_pl, full_dg = plane(np.ones_like(w)), diag(np.ones_like(w))
    old_mask = np.zeros_like(w); old_mask[:oc] = 1
    old_pl, old_dg = plane(old_mask), diag(old_mask)
    rec = dict(layer=l, K=int(len(w)), old=oc, norm_plane=float(np.linalg.norm(full_pl)), norm_old_plane=float(np.linalg.norm(old_pl)))
    for keep in keeps:
        mask = np.ones_like(w)
        for b, (st, wd) in enumerate(coh):
            if wd == 0 or b not in imp: continue
            half = wd // 2; im = imp[b]["imp"]; assert len(im) == half, (l, b, len(im), half)
            k = max(32, (int(round(half * keep)) + 31) // 32 * 32); k = min(k, half)
            drop = np.argsort(-im, kind="stable")[k:]
            mask[st + drop] = 0; mask[st + half + drop] = 0
        pl, dg = plane(mask), diag(mask)
        opl, odg = plane(mask * old_mask), diag(mask * old_mask)
        rec[f"keep{keep}"] = dict(plane_rel_err=float(np.linalg.norm(pl - full_pl) / np.linalg.norm(full_pl)),
                                 diag_rel_err=float(np.linalg.norm(dg - full_dg) / np.linalg.norm(full_dg)),
                                 old_plane_rel_err=float(np.linalg.norm(opl - old_pl) / (np.linalg.norm(old_pl) + 1e-30)),
                                 old_diag_rel_err=float(np.linalg.norm(odg - old_dg) / (np.linalg.norm(old_dg) + 1e-30)),
                                 columns_kept=int(mask.sum()))
    out[l] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(D / "prune_info_loss.json", "w"), indent=1)
