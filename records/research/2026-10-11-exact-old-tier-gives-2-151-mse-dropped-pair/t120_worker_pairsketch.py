#!/usr/bin/env python3
"""TAG opus-physicist T120 worker = t116_worker_joins_v2.py with the card's join push (s6bcard_loop5js.make_push, copied VERBATIM below as mode 'copy')
re-aimed at the bank's third cumulant (env T120_MODE): the bank's newborn columns come in +- PAIRS (model_bank_bfix: per born unit a, p+- = s + x+- e_a with
weights +lam / -lam; within a cohort the first half are the + columns, the second half the - columns), so a pair's third cumulant is 2 lam (c c h + c h c + h c c
+ h h h) with c = (p+ + p-)/2, h = (p+ - p-)/2, both pushed by the same maps. Modes, all at the card's ranks and with the same products:
  copy      the card's push exactly (positive control: bit for bit = T115 base)
  pairbase  sketch on the pair-transformed columns [c, h] with weights 2 lam^2 each (the same Gram as the card's lam^2 on p+-: equal to the base to rounding)
  pairk3    the same with the first-order third-cumulant loss weights: c 24 lam^2 |c|^2 |h|^2, h 12 lam^2 (|c|^4 + |h|^4)
  colk3     the card's column sketch with weights lam^2 |p|^PEXP per column (env T120_PEXP, default 4; lead: also 3)
The old tier's coordinates F are always those of the original columns p+- (the basis only is re-aimed); its core uses the mode's Gram. Optional read-only
DUMP (env T120_DUMP = a directory): per join, per newly joined pair, lam, |c|^2, |h|^2 and their kept squared norms in the kept basis.
v2 = TAG opus-physicist T116 worker v2 = t115_worker_joins.py + each join's own log line (layer, rank kept, live units, joined columns, and the
fraction of the joined covariance's energy the energy-ordered Nystrom basis captures at ranks 256 / 320 / 352 / 384: the card's make_push log, read only)
added to the push_counts.jsonl line as 'joinlog'. Arithmetic identical to v1.
v1 = TAG opus-physicist T115 (lead 10-11 19:0xZ, option B): the trainer's generated-seed worker v4 (unchanged, through runpy) on the PINNED otr_m.py copy with the
engineer's star drop-in v2 aliased (ss -1), for the PLAIN torch card (requests without 'cutsim': the card's own QR join, not the native Nystrom join), with the
old tier's JOIN SCHEDULE of this worker process taken from the environment: T115_JOINS = '<layer>:<cohort>,<cohort>,.../...' replaces s2card.EVERY3
(fired: 7:0,1,2/10:3,4,5/13:6,7,8; the request's 'sched' gives one rank per join). Per network (tag T115_NAME_<seed>) the push's per-layer column counts
(all columns, old-tier columns) go to <run>/push_counts.jsonl for the engineer's bill price. Arithmetic otherwise = worker_generated_v4.
usage: T115_NAME=early T115_JOINS='6:0,1,2/9:3,4,5/12:6,7,8' T120_MODE=pairk3 [T120_DUMP=<dir>] python -B ../physicist/t120_worker_pairsketch.py <run dir> <worker index>   (from O/trainer)"""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER, FROZEN_M = HERE.parent / 'trainer', HERE / 'otr_frozen_bba0dea0'
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(FROZEN_M))
import otr_m as _M_FROZEN
assert Path(_M_FROZEN.__file__).resolve().parent == FROZEN_M and hashlib.sha256(Path(_M_FROZEN.__file__).read_bytes()).hexdigest().startswith('bba0dea0'), _M_FROZEN.__file__
sys.path.append(str(HERE.parent / 'engineer' / 'a33_starsum'))
import star_ss_torch_v2
sys.modules['star_ss_torch'] = star_ss_torch_v2
import otr_engine as E
import s2card
JOINS = {int(g.split(':')[0]): [int(c) for c in g.split(':')[1].split(',')] for g in os.environ['T115_JOINS'].split('/')}
s2card.EVERY3.clear(); s2card.EVERY3.update(JOINS)
NAME, CUR = os.environ.get('T115_NAME', 'joins'), {}
import numpy as np
print('T120 worker: joins', s2card.EVERY3, 'name', NAME, 'mode', os.environ['T120_MODE'], 'dump', os.environ.get('T120_DUMP'), flush=True)
run0 = E.run_network


def run_network(rank, *args, **kwargs):
    CUR['rank'] = int(rank); CUR['counts'] = {}; CUR.pop('log', None)
    rec = run0(rank, *args, **kwargs)
    with (RUN / 'push_counts.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(tag='%s_%d' % (NAME, CUR['rank']), joins={str(k): v for k, v in s2card.EVERY3.items()}, counts=CUR['counts'],
                                 joinlog=CUR.get('log', {}).get('joins'))) + '\n')
    return rec
E.run_network = run_network
MODE, DUMP = os.environ['T120_MODE'], os.environ.get('T120_DUMP')
PEXP = float(os.environ.get('T120_PEXP', '4'))   # colk3: weights lam^2 |p|^PEXP
assert MODE in ('copy', 'pairbase', 'pairk3', 'colk3'), MODE
GRID = E.LC.GRID


def pairs(t, J, lam, start, stop, cohort_stops):
    """The joined range's columns as (c, h) pair columns: per cohort inside [start, stop) the first half are the + columns, the second half the - columns."""
    cs, hs, ls = [], [], []
    a = start
    for b in cohort_stops:
        if b <= start:
            continue
        if b > stop:
            break
        n = (b - a) // 2
        assert b - a == 2 * n, (a, b)
        assert bool((lam[a + n:b] == -lam[a:a + n]).all()), 'unpaired cohort'
        P, M = J[:, a - start:a - start + n], J[:, a - start + n:b - start]
        cs.append((P + M) / 2); hs.append((P - M) / 2); ls.append(lam[a:a + n])
        a = b
    assert a == stop, (a, stop)
    return t.cat(cs, dim=1), t.cat(hs, dim=1), t.cat(ls)


def make_push(qr, joins, arm, log):
    CUR['log'] = log      # the card's own join log (one dict per join), read after the network
    fixed = int(arm[5:]) if arm.startswith('fixed') else None; tau = float(arm[5:]) if arm.startswith('adapt') else None
    sched = dict(zip(sorted(joins), (int(x) for x in arm[5:].split('-')))) if arm.startswith('sched') else None
    def push(t, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        wp = w32[previous_indices, :].T
        before = [c_ for L_, cs in joins.items() if L_ < layer for c_ in cs]
        old_columns = cohort_stops[max(before)] if before else 0
        if layer in joins:
            start, stop = old_columns, cohort_stops[max(joins[layer])]
            J = p_full[previous_indices, start:stop]; weight = lam[start:stop] ** 2
            if MODE == 'colk3':
                weight = weight * (J * J).sum(0) ** (PEXP / 2)
            if MODE in ('pairbase', 'pairk3'):
                C_, H_, lp = pairs(t, J, lam, start, stop, cohort_stops)
                if MODE == 'pairbase':
                    wc = wh = 2 * lp ** 2
                else:
                    c2, h2 = (C_ * C_).sum(0), (H_ * H_).sum(0)
                    wc, wh = 24 * lp ** 2 * c2 * h2, 12 * lp ** 2 * (c2 * c2 + h2 * h2)
                Js, ws = t.cat((C_, H_), dim=1), t.cat((wc, wh))
            else:
                Js, ws = J, weight
            Qp = Q[previous_indices, :] if old_columns else None
            d = (Js * Js) @ ws
            if old_columns: d = d + ((Qp @ core) * Qp).sum(1)
            rows = t.argsort(-d.to(t.float64), stable=True)[:rank]          # ENERGY order (nested truncations)
            Y = Js @ (ws[:, None] * Js[rows, :].T)
            if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
            Qn = qr(t, Y)
            Fj = Qn.T @ J
            Fs = Fj if Js is J else Qn.T @ Js
            ek = (Fs.to(t.float64) ** 2) @ ws.to(t.float64)
            if old_columns:
                overlap = Qn.T @ Qp
                ek = ek + ((overlap.to(t.float64) @ core.to(t.float64)) * overlap.to(t.float64)).sum(1)
            tot = float(d.to(t.float64).sum()); cum = t.cumsum(ek, 0)
            E_ = {r: float(cum[min(r, len(cum)) - 1]) / tot for r in GRID}
            if fixed is not None: r = fixed
            elif sched is not None: r = sched[layer]
            else: r = next((x for x in (256, 320, 384) if E_[x] >= tau), 384)
            r = min(r, Qn.shape[1])
            Qn, Fj, Fs = Qn[:, :r], Fj[:r], Fs[:r]
            joined = Fs @ (ws[:, None] * Fs.T)
            if DUMP:
                C_, H_, lp = pairs(t, J, lam, start, stop, cohort_stops)
                kc, kh = Qn.T @ C_, Qn.T @ H_
                np.savez(Path(DUMP) / ('%s_%d_L%d.npz' % (NAME, CUR['rank'], layer)), layer=layer, r=r,
                         lam=lp.detach().to(t.float64).numpy(), c2=(C_ * C_).sum(0).detach().to(t.float64).numpy(),
                         h2=(H_ * H_).sum(0).detach().to(t.float64).numpy(), kc2=(kc * kc).sum(0).detach().to(t.float64).numpy(),
                         kh2=(kh * kh).sum(0).detach().to(t.float64).numpy())
            if old_columns:
                overlap = overlap[:r]
                F = t.cat((overlap @ F, Fj), dim=1); core = overlap @ (core @ overlap.T) + joined
            else:
                F, core = Fj, joined
            basis, old_columns = Qn, stop
            log['joins'].append(dict(layer=layer, r=int(r), E={str(k): round(v, 6) for k, v in E_.items()}, live=int(len(previous_indices)),
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
E.LC.make_push = make_push
install0 = E.M.install


def install(cd):
    out = install0(cd)
    push0 = cd.shared['push']

    def push(tt, layer, *a, **k):
        r = push0(tt, layer, *a, **k)
        CUR.setdefault('counts', {})[str(layer)] = dict(cols=int(r[0].shape[1]), old=int(r[4]))
        return r
    cd.shared['push'] = push
    return out
E.M.install = install
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
