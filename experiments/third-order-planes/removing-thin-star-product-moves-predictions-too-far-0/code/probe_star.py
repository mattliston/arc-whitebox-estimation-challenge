"""TAG opus_engineer a31 (2026-10-02): can the low-rank star do with one product less? In-process probe, one network, no file changed.
Every layer the parent factors the star plane A (n x n) as Q B with rank 32: Q0 = orth(32 columns of A), Q1 = orth(A' Q0),
Q = orth(A Q1), B = Q' A: THREE thin products with A (67 Mflop each as plain matmuls). After the second one the product Z = A Q1 is
already a rank-32 factorization, A ~ Z Q1' (the projection of A's rows on span(Q1)); taking it would drop the third product (Q' A)
and the second QR: 0.36% of the bill, no request.
Here the star planes of one prediction are captured as the estimator forms them, and for each the two factorizations are compared
in float64 numpy (outside the budget) against the plane itself and against the best possible rank-32 approximation, on the two
things the estimator uses: the plane and its transport through the next layer's weights, (W o W)' A' W (off-diagonal part and
diagonal).
usage: probe_star.py <candidate.py> <rank> <out.json (new)>"""
from pathlib import Path
import json
import sys
import time
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, rank, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_star_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
K, POWER, ITER, R = int(mod._LR_K), bool(mod._LRN_POWER), int(mod._LR_Q), mod._LR_R
assert R is None and POWER and ITER == 1, (K, POWER, ITER, R)
rows = []
original = mod._flms_cov_star
rel = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))


def probe(w, C, A, retain32=False):
    t0 = time.monotonic()
    W = np.asarray(w, dtype=np.float64); S = np.asarray(A, dtype=np.float64)
    cn = (S * S).sum(0)
    cols = np.sort(np.argsort(-cn, kind='stable')[:K])
    Q0 = np.linalg.qr(S[:, cols])[0]
    Q1 = np.linalg.qr(S.T @ Q0)[0]
    Z = S @ Q1
    Q = np.linalg.qr(Z)[0]
    B = Q.T @ S
    now, alt = Q @ B, Z @ Q1.T                                   # the parent's factorization and the one with a product less
    U, sv, Vt = np.linalg.svd(S)
    best = (U[:, :K] * sv[:K]) @ Vt[:K]
    wt, wt2 = W.T, W.T * W.T
    move = lambda X: wt2 @ X.T @ W                               # what the estimator forms: T2 @ P1' = (W o W)' (Q B)' W
    exact = move(S)
    off = ~np.eye(len(S), dtype=bool)
    row = dict(call=len(rows), norm=float(np.linalg.norm(S)), tail=float(np.sqrt((sv[K:] ** 2).sum()) / np.linalg.norm(S)),
               plane=dict(now=rel(now, S), alt=rel(alt, S), best=rel(best, S), alt_against_now=rel(alt, now)))
    moved = {name: move(X) for name, X in (('now', now), ('alt', alt), ('best', best))}
    row['moved_off'] = {name: rel(X[off], exact[off]) for name, X in moved.items()}
    row['moved_off']['alt_against_now'] = rel(moved['alt'][off], moved['now'][off])
    row['moved_diag'] = {name: rel(np.diag(X), np.diag(exact)) for name, X in moved.items()}
    row['moved_off_rms'] = dict(exact=float(np.sqrt((exact[off] ** 2).mean())), now_error=float(np.sqrt(((moved['now'] - exact)[off] ** 2).mean())),
                                alt_error=float(np.sqrt(((moved['alt'] - exact)[off] ** 2).mean())), alt_minus_now=float(np.sqrt(((moved['alt'] - moved['now'])[off] ** 2).mean())))
    row['seconds'] = round(time.monotonic() - t0, 2)
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
    bill = int(ctx.summary_dict()['flops_used'])
out.write_text(json.dumps(dict(candidate=str(candidate), rank=rank, bill=bill, finite=bool(np.isfinite(y).all()), rank_k=K, rows=rows), indent=1))
print(json.dumps(dict(done=True, calls=len(rows), bill=bill)))
