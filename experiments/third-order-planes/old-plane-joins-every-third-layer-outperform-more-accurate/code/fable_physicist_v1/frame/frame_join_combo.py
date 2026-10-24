#!/usr/bin/env python3
"""TAG fable-physicist (FJ-1): old-tier JOIN SCHEDULE x FRAME QUALITY per join (torch card, learned ON, no refit).
Schedule (which cohorts join the rank-384 frame at which layer) as fable_frame_v1/join_schedule_test.py; at each join the new frame
Qn (act x 384, post space) is built from the implicit weighted Gram G X = J diag(lam^2) J^T X + Qp core Qp^T X by:
  sketch  : Qn = qr(G Omega), Omega = W[previous, :384]                      (the bank)
  power1  : Q1 = qr(G Omega); Qn = qr(G Q1)                                   (one power step)
  over<k> : B = qr(G W[previous, :k]); Rayleigh-Ritz: top-384 eigvecs of B^T G B  (oversampled sketch)
  warm    : B = qr(G [Qp, W[previous, :64]]); Rayleigh-Ritz top 384 (first join: over448)
  eig     : exact top-384 eigenvectors of the dense G (ceiling)
cfg = <schedule>/<frame> with schedule in every (bank), even, every3 (7/10/13), every3b (8/11/14), every3c (6/9/12).
usage: frame_join_combo.py <ranks> <cfgs> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE.parent; P2 = PH.parent
sys.path.insert(0, str(PH)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; cfgs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); out = Path(sys.argv[4])
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
SCHED = {'every': {l: [l - 5] for l in range(5, 15)},
         'even': {6: [0, 1], 8: [2, 3], 10: [4, 5], 12: [6, 7], 14: [8, 9]},
         'every3': {7: [0, 1, 2], 10: [3, 4, 5], 13: [6, 7, 8]},
         'every3b': {8: [0, 1, 2], 11: [3, 4, 5], 14: [6, 7, 8]},
         'every3c': {6: [0, 1], 9: [2, 3, 4], 12: [5, 6, 7]}}
def rr_top(B, GB, rank):
    M = (B.T.double() @ GB.double()); M = (M + M.T) / 2
    ev, V = torch.linalg.eigh(M)
    return (B.double() @ V[:, -rank:]).to(B.dtype)
def make_push(qr, joins, frame):
    def push(torch_, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c for L_, cs in joins.items() if L_ < layer for c in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Qp = Q[previous_indices, :] if old_columns else None
            def G(X):
                Y = J @ (weight[:, None] * (J.T @ X))
                if Qp is not None: Y = Y + Qp @ (core @ (Qp.T @ X))
                return Y
            if frame == 'sketch':
                Qn = qr(torch, G(w32[previous_indices, :rank]))
            elif frame == 'power1':
                Qn = qr(torch, G(qr(torch, G(w32[previous_indices, :rank]))))
            elif frame.startswith('over') or (frame == 'warm' and Qp is None):
                k = int(frame[4:]) if frame.startswith('over') else 448
                B = qr(torch, G(w32[previous_indices, :k])); Qn = rr_top(B, G(B), rank)
            elif frame == 'warm':
                B = qr(torch, G(torch.cat((Qp, w32[previous_indices, :64]), dim=1))); Qn = rr_top(B, G(B), rank)
            elif frame == 'eig':
                Gd = (J.double() * weight.double()[None, :]) @ J.double().T
                if Qp is not None: Gd = Gd + Qp.double() @ (core.double() @ Qp.double().T)
                ev, V = torch.linalg.eigh((Gd + Gd.T) / 2); Qn = V[:, -rank:].to(w32.dtype)
            else:
                raise ValueError(frame)
            Fj = Qn.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
            if Qp is not None:
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
p, a, c, e = model.torch_payload(torch, np, False)
with out.open('a') as fo:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        for cfg in cfgs:
            card = model.Card(torch, w, p, a, c, e)
            if cfg != 'bank':
                sched, frame = cfg.split('/')
                card.shared['push'] = make_push(card.shared['_qr'], SCHED[sched], frame)
            t0 = time.time()
            with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
            rec = dict(rank=rank, cfg=cfg, mse=float(((pred[15] - TP[15]) ** 2).mean()), wall=round(time.time() - t0, 1))
            fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps(rec), flush=True)
