"""TAG opus_engineer a35 (2026-10-02): does the old tier's join lose NORM COUPLING? In-process probe, one network, no file changed.
Background (a33_starsum): for third-order content T (post space) the pre-activation plane of the next layer holds the coherent part
m_i (W' t)_j with t_c = sum_a T_aac (the coupling of unit c to the squared norm). For the star plane, an error of 1-2% in ITS t
(the row sums the rank-32 factors missed) was what cost MSE. A join replaces the columns J of the joined cohorts (weights lam) by
their projection on a rank-r basis Qn (r = 384 / 320 / 384 at layers 7 / 10 / 13); for columns the vector is
    t = J (lam o |J_r|^2)   before,     t' = P J (lam o |P J_r|^2)   after,  P = Qn Qn'.
Here the join of each join layer is repeated in float64 numpy on the columns the estimator hands to its push (same row choice,
same weights, same rank; the old tier's own re-projection at layers 10 and 13 is NOT included, only the newly joined cohorts), and
the miss t - t' is set beside: t of the joined columns, t of all carried columns of that layer, the star plane's row sums / 3 of
the same layer (the same quantity for the content the columns do not hold) and the part of those the rank-32 factors miss.
usage: probe_join.py <candidate.py> <rank> <out.json (new)>"""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, rank, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_join_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
rows_out = []
stars = {}
push, cov_star = mod._shared_old384_push, mod._flms_cov_star
layer_of_star = [0]


def keep_star(w, C, A, retain32=False):
    layer_of_star[0] += 1
    if layer_of_star[0] in mod._JOIN_SCHEDULE:
        S = np.asarray(A, dtype=np.float64)
        cn = (S * S).sum(0); cols = np.sort(np.argsort(-cn, kind='stable')[:32])
        Q = np.linalg.qr(S[:, cols])[0]; Q = np.linalg.qr(S @ np.linalg.qr(S.T @ Q)[0])[0]
        r = S.sum(1)
        stars[layer_of_star[0]] = dict(t=r / 3, miss=(r - Q @ (Q.T @ r)) / 3)
    return cov_star(w, C, A, retain32)


def probe(layer, w32, p, previous, newborn, lam, state, rows=None):
    joins_here = mod._JOIN_SCHEDULE.get(layer)
    if joins_here is not None:
        old0 = int(state['columns'])
        start = int(state['cohorts'][joins_here[0]][0]); stop = int(state['cohorts'][joins_here[1]][0]) + int(state['cohorts'][joins_here[1]][1])
        prev = np.asarray(previous)
        P = np.asarray(p, dtype=np.float64); L = np.asarray(lam, dtype=np.float64)
        J = P[:, start - old0:stop - old0]; lj = L[start:stop]; weight = lj * lj
        r = int(mod._r320_rank(layer))
        d = (J * J) @ weight
        if old0:
            Qp = np.asarray(state['Q'], dtype=np.float64)[prev]; core = np.asarray(state['core'], dtype=np.float64)
            d = d + ((Qp @ core) * Qp).sum(1)
        pick = np.sort(np.argsort(-d, kind='stable')[:r])
        Y = J @ (weight[:, None] * J[pick].T)
        if old0:
            Y = Y + Qp @ (core @ Qp[pick].T)
        Qn = np.linalg.qr(Y)[0]
        F = Qn.T @ J
        t_before = J @ (lj * (J * J).sum(0)); t_after = Qn @ (F @ (lj * (F * F).sum(0)))
        young = P[:, stop - old0:]; ly = L[stop:stop + young.shape[1]]
        t_young = young @ (ly * (young * young).sum(0))
        full = lambda v: np.bincount(prev, weights=v, minlength=1024)
        miss = full(t_before - t_after)
        W = np.asarray(w32, dtype=np.float64)
        st = stars.get(layer, {})
        row = dict(layer=layer, rank=r, joined_columns=int(stop - start), live_rows=int(len(prev)), old_columns_before=old0,
                   energy_lost=float(((J - Qn @ F) ** 2 @ weight).sum() / ((J * J) @ weight).sum()),
                   t_joined=float(np.linalg.norm(t_before)), t_younger_columns=float(np.linalg.norm(t_young)), join_miss=float(np.linalg.norm(miss)),
                   join_miss_over_t_joined=float(np.linalg.norm(miss) / np.linalg.norm(t_before)),
                   star_t=float(np.linalg.norm(st['t'])) if st else None, star_rank32_miss=float(np.linalg.norm(st['miss'])) if st else None,
                   pushed_join_miss=float(np.linalg.norm(miss @ W)), pushed_star_rank32_miss=float(np.linalg.norm(st['miss'] @ W)) if st else None,
                   cosine_join_miss_with_star_miss=float(miss @ st['miss'] / (np.linalg.norm(miss) * np.linalg.norm(st['miss']))) if st else None)
        rows_out.append(row)
        print(json.dumps(row), flush=True)
    return push(layer, w32, p, previous, newborn, lam, state, rows=rows)


mod._shared_old384_push, mod._flms_cov_star = probe, keep_star
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
out.write_text(json.dumps(dict(candidate=str(candidate), rank=rank, finite=bool(np.isfinite(y).all()), rows=rows_out), indent=1))
print(json.dumps(dict(done=True, joins=len(rows_out))))
