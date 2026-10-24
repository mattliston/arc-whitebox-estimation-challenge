"""TAG fable-physicist (09-25): s6bcard_loop5js = s6bcard_loop4js with S24's star-Nystrom (the porter's nysp install_star, as in
s6bcard_loop5j) in place of the s1=0 lrstar: the full S28 torch structure (star-Nystrom + per-join old-tier ranks), for CPU checks.
Original header: s6bcard_loop4j with a PER-JOIN old-tier rank schedule, for the joint refit AT the S28
schedule. The join push is the analyst's energy-ordered Nystrom join with nested truncation, copied VERBATIM from
fable_analyst_v1/r320_features.py (AN-69b, make_push; arm 'sched<L7>-<L10>-<L13>'). It is installed exactly as that script installs it:
the loop4j card is built first (join, star s1=0, gn_linear_hook, pair series), then cd.shared['push'] is replaced. So a rank cut at L10
feeds the L13 join exactly as in the analyst's torch numbers. The schedule comes from env JOIN_SCHED (default 384-320-384, the lead's
11:37 GO; L7/L10/L13). fine_ckpt=False always. Dials via s6bcard_joint2t.DIALS_T (the driver sets them)."""
import os, sys
from pathlib import Path
P2 = Path(__file__).resolve().parents[2]
for _d in ('fable_covhunter_v1', 'fable_engineer_v1', 'fable_analyst_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1'):
    if str(P2 / _d) not in sys.path: sys.path.append(str(P2 / _d))
import torch
import s6bcard_joint2t as JT
import join_nystrom_torch as jn, gn_linear_hook as gl, pair_series_torch as ps
import star_nys_test as SN
import s2card
payload, weights, model, truthpost = JT.payload, JT.weights, JT.model, JT.truthpost
SCHED = os.environ.get('JOIN_SCHED', '384-320-384')
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


def card(t, w, p, a, c, e, fine_ckpt=False, **kw):
    assert JT.DIALS_T is not None, 'set s6bcard_joint2t.DIALS_T before building the card'
    cd = JT.card(t, w, p, a, c, e, fine_ckpt=False, **kw)
    jn.install(cd, t); SN.install_star(cd, t, 'nysp', []); gl.install(cd, t); ps.install(cd, t)
    cd.shared['push'] = make_push(cd.shared['_qr'], s2card.EVERY3, 'sched' + SCHED, dict(joins=[], layers=[]))
    return cd
