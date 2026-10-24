#!/usr/bin/env python3
"""TAG fable-analyst (AN-65): ENERGY-ADAPTIVE old-tier rank at the joins (lead 02:0x). Card: the physicist's S23-structure torch card
refit6b/s6bcard_loop4j.py (joint2t dials + Nystrom join + star s1 = 0 + gn_linear_hook + pair series), loopR1 params + round-1 dials, loaded
as refit_mps_v3j2.py --eval-only does, learned ON, no refit, float64 CPU card. The join (fable_engineer_v1/join_nystrom_torch.py math) is
rebuilt with its 384 Nystrom landmark rows in ENERGY order (descending Gram diagonal d) instead of index order: the QR of the
landmark Gram columns then nests, so the first r basis vectors span exactly the rank-r Nystrom join, and the captured energy of every
truncation comes from quantities the join computes anyway: e_k = sum_c w_c (Qn^T J)_kc^2 + (Qn^T Qp core Qp^T Qn)_kk, E(r) =
sum_{k<r} e_k / trace(Gram). Arms:
  base        the card as built (index-ordered landmarks, rank 384): parity reference
  fixed<r>    energy-ordered join truncated to r at every join (fixed384 must match base to rounding)
  adapt<tau>  per join: the smallest r in {256, 320, 384} with E(r) >= tau
  sched<a>-<b>-<c>  fixed per-join ranks at L7 / L10 / L13 (no per-net adaptation)
Per net: final MSE vs the organizer truth, per join (layer, r, E(256/320/352/384), live rows, joined columns), per layer (old columns,
frame rank in force) for the bill estimate. usage: adaptive_join.py <ranks|valid16> <threads> <out.jsonl> <arms>"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for d in ('fable_engineer_v1', 'fable_covhunter_v1', 'fable_physicist_v1/refit6b', 'fable_analyst_v1'):
    sys.path.insert(0, str(P2 / d))
import numpy as np, torch
import s6bcard_loop4j as LC
import s2card
JT = LC.JT
arg, THREADS, OUT, ARMS = sys.argv[1], int(sys.argv[2]), HERE / sys.argv[3], sys.argv[4].split(',')
torch.set_num_threads(THREADS)
_dv = json.load(open(P2 / 'fable_covhunter_v1/dials_r1_ep2.json')); JT.DIALS_T = JT.make_dials(torch, {k: float(_dv[k]) for k in JT.KEYS}, trainable=())
PARAMS = P2 / 'fable_physicist_v1/refit6b/s6b_loopR1/params_epoch0.pt'
p, a, c, e = LC.payload(torch); sv = torch.load(PARAMS)
with torch.no_grad():
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
RANKS = inputs['valid'][:16] if arg == 'valid16' else [int(x) for x in arg.split(',')]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
GRID = (256, 320, 352, 384)


def make_push(qr, joins, arm, log):
    fixed = int(arm[5:]) if arm.startswith('fixed') else None; tau = float(arm[5:]) if arm.startswith('adapt') else None
    sched = dict(zip(sorted(joins), (int(x) for x in arm[5:].split('-')))) if arm.startswith('sched') else None
    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c_ for L_, cs in joins.items() if L_ < layer for c_ in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            Qp = Q[previous_indices, :] if old_columns else None
            d = (J * J) @ weight
            if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
            rows = torch.argsort(-d.to(t.float64), stable=True)[:rank]          # ENERGY order (nested truncations)
            Y = J @ (weight[:, None] * J[rows, :].T)
            if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
            Qn = qr(t, Y)
            Fj = Qn.T @ J
            ek = (Fj.to(t.float64) ** 2) @ weight.to(t.float64)
            if old_columns:
                overlap = Qn.T @ Qp
                ek = ek + ((overlap.to(t.float64) @ core.to(t.float64)) * overlap.to(t.float64)).sum(1)
            tot = float(d.to(t.float64).sum()); cum = t.cumsum(ek, 0)
            E = {r: float(cum[min(r, len(cum)) - 1]) / tot for r in GRID}
            if fixed is not None: r = fixed
            elif sched is not None: r = sched[layer]
            else: r = next((x for x in (256, 320, 384) if E[x] >= tau), 384)
            r = min(r, Qn.shape[1])
            Qn, Fj = Qn[:, :r], Fj[:r]
            joined = Fj @ (weight[:, None] * Fj.T)
            if old_columns:
                overlap = overlap[:r]
                F = t.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
            log['joins'].append(dict(layer=layer, r=int(r), E={str(k): round(v, 6) for k, v in E.items()}, live=int(len(previous_indices)),
                                     joined=int(stop - start)))
        elif old_columns:
            basis = Q[previous_indices, :]
        else:
            log['layers'].append(dict(layer=layer, old=0, rank=0))
            return wp @ p_full[previous_indices, :], None, F, core, 0
        log['layers'].append(dict(layer=layer, old=int(old_columns), rank=int(basis.shape[1])))
        U = wp @ basis
        return t.cat((U @ F, wp @ p_full[previous_indices, old_columns:]), dim=1), U, F, core, old_columns
    return push


with OUT.open('a') as fo:
    for r_ in RANKS:
        w = np.load(STORE / Path(man[r_]['path']).parent / 'weights.npy')
        tp = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r_:04d}.npz')['mc_post'][15].astype(np.float64)
        for arm in ARMS:
            t0 = time.time(); cd = LC.card(torch, w, p, a, c, e); log = dict(joins=[], layers=[])
            if arm != 'base': cd.shared['push'] = make_push(cd.shared['_qr'], s2card.EVERY3, arm, log)
            with torch.no_grad(): pred = cd.run(False)[15].to(torch.float64).numpy()
            rec = dict(rank=r_, arm=arm, mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1), **log)
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            print(json.dumps(dict(rank=r_, arm=arm, mse=rec['mse'], wall=rec['wall'], ranks=[j['r'] for j in log['joins']],
                                  E320=[j['E']['320'] for j in log['joins']])), flush=True)
