#!/usr/bin/env python3
"""TAG opus-physicist T123 (lead 10-12 03:0xZ, big bet: PAIR-SPLIT YOUNG READ, compress c only) = t115_worker_joins.py (plain torch card, fired joins
from T115_JOINS) + a compression of the young bank's pair COMMON parts at layers T123_LO..T123_HI (default 8-14). Young pairs p+- = c +- h (weights +-lam;
per cohort the first half are the + columns). Basis U = the top-r eigenvectors of the lam^2-weighted Gram of that layer's young c (exact eigh: the oracle basis).
  T123_MODE=off  the card exactly (control).
  T123_MODE=A    the young READ's right factor only: plane += -(L+ - L-) diag(lam) ((I - UU')c)' on the response x live block (L+- = the pair's left operands
                 pp^2 on the response rows; the law fold, the left operands and h untouched; U over the live rows); innermost read wrapper (under rso / rsy).
  T123_MODE=B    c compressed everywhere: the push's young columns become U U' c +- h (U over all rows) before every consumer (read, law, k3 diagonal, state).
T123_R = r (or 'full' = every eigenvector: must equal the card to rounding). Per network and layer, <run>/t123_log.jsonl: the lam^2-weighted c energy kept and
(mode A) the young plane's relative change |dplane|^2 / |young plane|^2.
v1 = TAG opus-physicist T115 (lead 10-11 19:0xZ, option B): the trainer's generated-seed worker v4 (unchanged, through runpy) on the PINNED otr_m.py copy with the
engineer's star drop-in v2 aliased (ss -1), for the PLAIN torch card (requests without 'cutsim': the card's own QR join, not the native Nystrom join), with the
old tier's JOIN SCHEDULE of this worker process taken from the environment: T115_JOINS = '<layer>:<cohort>,<cohort>,.../...' replaces s2card.EVERY3
(fired: 7:0,1,2/10:3,4,5/13:6,7,8; the request's 'sched' gives one rank per join). Per network (tag T115_NAME_<seed>) the push's per-layer column counts
(all columns, old-tier columns) go to <run>/push_counts.jsonl for the engineer's bill price. Arithmetic otherwise = worker_generated_v4.
usage: T115_NAME=a384 T115_JOINS='7:0,1,2/10:3,4,5/13:6,7,8' T123_MODE=A T123_R=384 python -B ../physicist/t123_worker_pairsplit.py <run dir> <worker index>   (from O/trainer)"""
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
print('T123 worker: joins', s2card.EVERY3, 'name', NAME, 'mode', os.environ.get('T123_MODE', 'off'), 'r', os.environ.get('T123_R', '0'), 'layers', os.environ.get('T123_LO', '8'), os.environ.get('T123_HI', '14'), flush=True)
run0 = E.run_network


import torch
MODE, RSPEC = os.environ.get('T123_MODE', 'off'), os.environ.get('T123_R', '0')
LO, HI = int(os.environ.get('T123_LO', '8')), int(os.environ.get('T123_HI', '14'))
assert MODE in ('off', 'A', 'B'), MODE


def pair_index(stops, old, total):
    """(+ columns, - columns) of the young cohorts in [old, total): per cohort the first half are the + columns, the second half the - columns."""
    P, M, a = [], [], old
    for b in stops:
        if b <= old:
            continue
        n = (b - a) // 2
        assert b - a == 2 * n, (a, b)
        P += range(a, a + n); M += range(a + n, b); a = b
    assert a == total, (a, total)
    return torch.tensor(P, dtype=torch.long), torch.tensor(M, dtype=torch.long)


def basis(c, lp):
    """Top-r eigenvectors of the lam^2-weighted Gram of the columns c (float64), and the kept share of that weighted energy."""
    X = c.to(torch.float64) * lp.to(torch.float64)[None, :]
    ev, V = torch.linalg.eigh(X @ X.T)
    r = V.shape[1] if RSPEC == 'full' else min(int(RSPEC), V.shape[1])
    Ub = V[:, V.shape[1] - r:]
    return Ub, float((ev[V.shape[1] - r:].sum() / ev.sum()).item())


def run_network(rank, *args, **kwargs):
    CUR['rank'] = int(rank); CUR['counts'] = {}; CUR['log'] = {}
    rec = run0(rank, *args, **kwargs)
    with (RUN / 'push_counts.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(tag='%s_%d' % (NAME, CUR['rank']), joins={str(k): v for k, v in s2card.EVERY3.items()}, counts=CUR['counts'])) + '\n')
    with (RUN / 't123_log.jsonl').open('a') as fh:
        fh.write(json.dumps(dict(tag='%s_%d' % (NAME, CUR['rank']), mode=MODE, r=RSPEC, layers=CUR['log'])) + '\n')
    return rec
E.run_network = run_network
install0 = E.M.install


def install(cd):
    if MODE == 'A':      # innermost: installed before the card module's own wrappers (rso / rsy scale lam on top of it, the law folds on top of it)
        read_inner = cd.shared['read']

        def read_a(tt, pp, lam, response_indices, indices, U, F, old_columns):
            out = read_inner(tt, pp, lam, response_indices, indices, U, F, old_columns)
            layer = CUR.get('layer')
            if layer is None or not (LO <= layer <= HI) or old_columns >= pp.shape[1]:
                return out
            with torch.no_grad():
                Pi, Mi = pair_index(CUR['stops'], old_columns, pp.shape[1])
                f64 = torch.float64
                cl = ((pp[indices][:, Pi] + pp[indices][:, Mi]) / 2).to(f64); hl = ((pp[indices][:, Pi] - pp[indices][:, Mi]) / 2).to(f64)
                lp = lam[Pi].to(f64)
                Ub, kept = basis(cl, lp)
                cperp = cl - Ub @ (Ub.T @ cl)
                Lp, Lm = pp[response_indices][:, Pi].to(f64) ** 2, pp[response_indices][:, Mi].to(f64) ** 2
                young = (Lp - Lm) @ (lp[:, None] * cl.T) + (Lp + Lm) @ (lp[:, None] * hl.T)
                delta = -(Lp - Lm) @ (lp[:, None] * cperp.T)
                yl = old_columns
                direct = (pp[response_indices][:, yl:].to(f64) ** 2) @ (pp[indices][:, yl:].to(f64) * lam[yl:].to(f64)[None, :]).T
                CUR['log'][str(layer)] = dict(c_kept=kept, dplane_rel=float(((delta ** 2).sum() / (young ** 2).sum()).item()),
                                             young_rel_total=float(((young ** 2).sum() / (out.to(f64) ** 2).sum()).item()), pairs=int(Pi.numel()), r=int(Ub.shape[1]),
                                             pair_formula_check=float(((young - direct).norm() / direct.norm()).item()))
            return out + delta.to(out.dtype)
        cd.shared['read'] = read_a
    out = install0(cd)
    push0 = cd.shared['push']

    def push(tt, layer, *a, **k):
        r = push0(tt, layer, *a, **k)
        CUR.setdefault('counts', {})[str(layer)] = dict(cols=int(r[0].shape[1]), old=int(r[4]))
        CUR['stops'], CUR['layer'] = a[7], layer
        if MODE == 'B' and LO <= layer <= HI and r[4] < r[0].shape[1]:
            with torch.no_grad():
                pp, old_columns = r[0], r[4]
                Pi, Mi = pair_index(a[7], old_columns, pp.shape[1])
                c, h = (pp[:, Pi] + pp[:, Mi]) / 2, (pp[:, Pi] - pp[:, Mi]) / 2
                Ub, kept = basis(c, a[2][Pi])
                cp = (Ub @ (Ub.T @ c.to(torch.float64))).to(pp.dtype)
                pp = pp.clone(); pp[:, Pi] = cp + h; pp[:, Mi] = cp - h
                CUR['log'][str(layer)] = dict(c_kept=kept, pairs=int(Pi.numel()), r=int(Ub.shape[1]))
            r = (pp,) + tuple(r[1:])
        return r
    cd.shared['push'] = push
    return out
E.M.install = install
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
