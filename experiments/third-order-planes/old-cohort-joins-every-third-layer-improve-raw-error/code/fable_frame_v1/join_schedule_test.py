#!/usr/bin/env python3
"""TAG fable-frame (FR-7): JOIN SCHEDULE of the shared old tier. Bank: one cohort joins the rank-384 frame at every layer 5..14
(range finder + QR + projections + rotation of all old factors each layer: ~37% of the tier's cost). Variants join TWO cohorts every
other layer (no frame work at the in-between layers; the waiting cohort stays exact in the young tier one layer longer).
'every' re-implements the bank schedule (parity control). Torch card, learned ON, no refit.
usage: join_schedule_test.py <ranks> <schedules e.g. every,even,odd> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; scheds = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
SCHED = {
    'every': {l: [l - 5] for l in range(5, 15)},
    'even': {6: [0, 1], 8: [2, 3], 10: [4, 5], 12: [6, 7], 14: [8, 9]},
    'odd': {5: [0], 7: [1, 2], 9: [3, 4], 11: [5, 6], 13: [7, 8]},
    'every3': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]},
    'late2': {5: [0], 6: [1], 7: [2], 8: [3], 10: [4, 5], 12: [6, 7], 14: [8, 9]},
    'every3b': {8: [0, 1, 2], 11: [3, 4, 5], 14: [6, 7, 8]},
    'every3c': {6: [0, 1], 9: [2, 3, 4], 12: [5, 6, 7]},
}
def make_push(qr, joins):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Omega = w32[previous_indices, :rank]
            Y = J @ (weight[:, None] * (J.T @ Omega))
            Qp = Q[previous_indices, :] if old_columns else None
            if old_columns: Y = Y + Qp @ (core @ (Qp.T @ Omega))
            Qn = qr(torch, Y)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = Qn.T @ Qp
                F = torch.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            return wp @ p_full[previous_indices, :], None, F, core, 0
        U = wp @ basis
        return torch.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push
def run(rank, sched):
    sched, _, r = sched.partition('@'); model.TIER_RANK = int(r) if r else 384
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    if sched != 'bank': card.shared['push'] = make_push(card.shared['_qr'], SCHED[sched])
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
for rk in ranks:
    base = None
    for s in ['bank'] + scheds:
        t0 = time.time(); mse = run(rk, s)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, sched=s, mse=mse, ratio=round(mse / base, 5), wall=round(time.time() - t0, 1))), flush=True)
