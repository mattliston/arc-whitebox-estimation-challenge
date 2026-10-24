"""TAG opus_engineer a44 probe v2 (2026-10-10): how much of what the dead cut drops from C_pre = W' C W is held by a few directions of
the cut units' rows of C. With a = the cut roster of the layer below (count at or above the gate floor rounded up to 64, by gate),
c = the rest, Cc = C[c, :] (the cut units' rows, all columns), the dropped part is exactly
    D = W_c' (Cc W) + (W_c' Cc[:, a] W_a)'
and with Cc replaced by its best rank-r approximation (exact SVD here; natively a range finder) the same two products cost about
4 r n^2 flops. Per layer: the relative error of C_pre (Frobenius, off-diagonal, diagonal rms) for the cut alone and with the
rank-r term, r = 1 2 4 8 16 32. In process, one network; the prediction is the parent's (the wrapper returns the file's result).
usage: probe_deadcov_v2.py <candidate.py> <rank>"""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe_deadcov2'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
parent = mod._flms_cov_star
RANKS = (1, 2, 4, 8, 16, 32)


def metrics(X, full, off):
    d = np.diag(full)
    return dict(F=float(np.linalg.norm(X - full) / np.linalg.norm(full)), off=float(np.linalg.norm((X - full)[off]) / np.linalg.norm(full[off])),
                drms=float(np.sqrt(np.mean(((np.diag(X) - d) / d) ** 2))))


def wrap(w, C, A, retain32=False):
    out = parent(w, C, A, retain32)
    layer = int(mod._PHYS_LAYER[0]) + 1
    gate = np.asarray(mod._E2_LAYER['g'], dtype=np.float64)
    n = gate.shape[0]
    m = min(n, (int((gate >= 0.01).sum()) + 63) // 64 * 64)
    if m >= n:
        print(json.dumps(dict(layer=layer, m=m)), flush=True)
        return out
    order = np.argsort(-gate, kind='stable')
    a, c = order[:m], order[m:]
    W, Cn = np.asarray(w, dtype=np.float64), np.asarray(C, dtype=np.float64)
    full = W.T @ Cn @ W
    core = W[a].T @ Cn[np.ix_(a, a)] @ W[a]
    off = ~np.eye(n, dtype=bool)
    row = dict(layer=layer, m=m, cut=metrics(core, full, off))
    Cc = Cn[c, :]
    U, s, Vt = np.linalg.svd(Cc, full_matrices=False)
    row['sv_share'] = [float((s[:r] ** 2).sum() / (s ** 2).sum()) for r in RANKS]
    Wc = W[c]
    for r in RANKS:
        L = U[:, :r] * s[:r]                                   # q x r
        R = Vt[:r]                                             # r x n
        left = Wc.T @ L                                        # n x r
        D = left @ (R @ W) + (left @ (R[:, a] @ W[a])).T
        row['r%d' % r] = metrics(core + D, full, off)
    print(json.dumps(row), flush=True)
    return out


mod._flms_cov_star = wrap
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(json.dumps(dict(rank=rank, done=True)), flush=True)
