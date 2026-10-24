"""TAG opus_engineer a31 (2026-10-02), third probe (the second had the mean of the squared weights wrong by the factor 2: the
weights have variance 2/n; its 'exact row sums' columns are a HALF correction).
The estimator needs s2 = (W o W)' A' W. Write (W o W)' = m 1' + R, m_i = mean over k of W[k, i]^2. Then
    s2 = m (W' A 1)' + R A' W :  a rank-one part that needs only the ROW SUMS of the star plane A, plus a rest.
With a low-rank form A ~ L (L = Q B, or the two-product form Z Q1') the error of the first part is m (W' u)', u = (A - L) 1, and one
vector through W removes it exactly. In-process, one network, planes captured as the estimator forms them, float64 numpy outside
the budget. Per layer and per rank (8, 16, 24, 32, 48): relative error of the transported plane (off-diagonal, and diagonal) for
  now  = Q B (the parent's three-product form),  alt = (A Q1) Q1' (two products),  each plain and with exact row sums.
usage: probe_star3.py <candidate.py> <rank> <out.json (new)>"""
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
mod = types.ModuleType('candidate_star_probe3'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
assert mod._LR_R is None and mod._LRN_POWER and mod._LR_Q == 1 and mod._LR_K == 32
rows = []
original = mod._flms_cov_star
RANKS = (8, 16, 24, 32, 48)


def factors(S, K):
    cn = (S * S).sum(0)
    cols = np.sort(np.argsort(-cn, kind='stable')[:K])
    Q0 = np.linalg.qr(S[:, cols])[0]
    Q1 = np.linalg.qr(S.T @ Q0)[0]
    Z = S @ Q1
    Q = np.linalg.qr(Z)[0]
    return dict(now=Q @ (Q.T @ S), alt=Z @ Q1.T)


def probe(w, C, A, retain32=False):
    W = np.asarray(w, dtype=np.float64); S = np.asarray(A, dtype=np.float64); n = len(S)
    wt2 = W.T * W.T
    m = wt2.mean(1)
    off = ~np.eye(n, dtype=bool)
    exact = wt2 @ S.T @ W
    one = np.ones(n)
    first = np.outer(m, (S @ one) @ W)
    f = lambda X: float(np.sqrt((X[off] ** 2).mean()))
    g = lambda X: float(np.linalg.norm(np.diag(X)))
    row = dict(call=len(rows), exact_rms=f(exact), first_part_rms=f(first), rest_rms=f(exact - first), mean_m=float(m.mean() * n), by_rank={})
    for K in RANKS:
        entry = {}
        for name, low in factors(S, K).items():
            moved = wt2 @ low.T @ W
            fix = np.outer(m, ((S - low) @ one) @ W)
            entry[name] = dict(plain=f(moved - exact) / f(exact), sums=f(moved + fix - exact) / f(exact), row_sum_miss=float(np.linalg.norm((S - low) @ one) / np.linalg.norm(S @ one)),
                               diag_plain=g(moved - exact) / g(exact), diag_sums=g(moved + fix - exact) / g(exact))
        row['by_rank'][K] = entry
    rows.append(row)
    print(json.dumps(row), flush=True)
    return original(w, C, A, retain32)


mod._flms_cov_star = probe
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
out.write_text(json.dumps(dict(candidate=str(candidate), rank=rank, finite=bool(np.isfinite(y).all()), rows=rows), indent=1))
print(json.dumps(dict(done=True, calls=len(rows))))
