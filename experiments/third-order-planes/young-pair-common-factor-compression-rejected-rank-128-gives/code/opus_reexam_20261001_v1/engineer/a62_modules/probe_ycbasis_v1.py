"""TAG opus_engineer a62 (2026-10-12 03:3xZ; lead's pair-split scope, question 2): MEASUREMENT, read-only. At every layer 8-14 of one native S48b
prediction, the young pairs' common parts c = (p+ + p-)/2 (young = the bank's columns from the old tier's column count on, cohorts in birth
order, each [plus block | minus block]) and their difference parts h = (p+ - p-)/2, weighted by lam^2 (lam of the plus column), against:
  - the old tier's pushed basis of this layer (state['pre_basis'], computed every layer anyway): share of the lam^2-weighted energy of the
    c's (and of the h's) inside its span;
  - the best rank-r basis for the c's (top eigenvectors of their lam^2-weighted Gram): r = 256 / 320 / 384 / 512.
All arithmetic on local numpy copies (no billed op); predictions are the parent's.
usage: probe_ycbasis_v1.py <candidate.py> <seed> <out.json>"""
import json
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
local = lambda x: np.array(np.asarray(x).view(np.ndarray), dtype=np.float64)
STATE = {}
rows = []
parent_push = mod._OTT_ORIGINAL_PUSH


def push_wrap(layer, w32, p, previous, newborn, lam, state, rows=None):
    out = parent_push(layer, w32, p, previous, newborn, lam, state, rows=rows)
    STATE['state'] = state
    try:   # the old tier's basis pushed to ALL units of this layer (pre_basis may hold the cropped rows only)
        if state.get('Q') is not None and int(state.get('columns', 0)):
            prev = np.asarray(previous).view(np.ndarray)
            STATE['U'] = local(w32)[prev, :].T @ local(state['Q'])[prev, :]
        else:
            STATE['U'] = None
    except Exception as e:
        print('probe push error', repr(e), flush=True)
        STATE['U'] = None
    return out


mod._OTT_ORIGINAL_PUSH = push_wrap
parent_left = mod._np_left


def captured(Q, X, wt):
    e = (X * X).sum(0)
    q = Q.T @ X
    return float((wt * (q * q).sum(0)).sum() / (wt * e).sum())


def np_left(l, pp2, pp, response_indices, *args, **kwargs):
    try:
        st = STATE.get('state')
        S = mod._NP_STATE
        if 8 <= l <= 14 and st is not None and S.get('sizes') is not None and STATE.get('U') is not None:
            old = int(st['columns'])
            P = local(pp)
            lam = local(args[2]) if len(args) > 2 else None      # _np_left(l, pp2, pp, response_indices, response_mask, alive, lam, ...)
            sizes = list(S['sizes'])
            bounds, at = [], 0
            for w in sizes:
                bounds.append((at, at + w)); at += w
            cs, hs, ws = [], [], []
            for i in range(0, len(bounds) - 1, 2):
                (a0, a1), (b0, b1) = bounds[i], bounds[i + 1]
                if a0 < old:
                    continue
                cs.append((P[:, a0:a1] + P[:, b0:b1]) / 2); hs.append((P[:, a0:a1] - P[:, b0:b1]) / 2)
                ws.append(lam[a0:a1] ** 2 if lam is not None and lam.shape[0] >= a1 else np.ones(a1 - a0))
            if cs:
                C, H, wt = np.concatenate(cs, 1), np.concatenate(hs, 1), np.concatenate(ws)
                U = STATE['U']
                Q = np.linalg.qr(U)[0] if U.shape[0] == C.shape[0] else None
                Cw = C * np.sqrt(wt)[None, :]
                ev = np.linalg.eigvalsh(Cw @ Cw.T)[::-1]
                tot = ev.sum()
                row = dict(layer=int(l), rows=int(C.shape[0]), pairs=int(C.shape[1]), old_columns=old, basis_rank=int(U.shape[1]),
                           c_in_old_basis=None if Q is None else captured(Q, C, wt), h_in_old_basis=None if Q is None else captured(Q, H, wt),
                           c_best={r: float(ev[:r].sum() / tot) for r in (256, 320, 384, 512)},
                           h_share_of_energy=float((wt * (H * H).sum(0)).sum() / ((wt * (H * H).sum(0)).sum() + (wt * (C * C).sum(0)).sum())))
                rows.append(row)
                print(json.dumps(row), flush=True)
    except Exception as e:
        print('probe error', repr(e), flush=True)
    return parent_left(l, pp2, pp, response_indices, *args, **kwargs)


mod._np_left = np_left
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
out.write_text(json.dumps(dict(seed=seed, rows=rows), indent=1))
print('done', len(rows))
