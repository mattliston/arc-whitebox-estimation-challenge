"""TAG opus_engineer a44 probe (2026-10-10): what the covariance sandwich C_pre = W' C W of layers 1-14 loses when the units of the
layer below outside its live set (the select() roster, gate >= 0.01 rounded up to 32) are cut. In process, one network, one thread;
the prediction is the parent's (the wrapper returns the file's own result); the forms are computed on the side in float64 numpy:
    core  W_a' C_aa W_a                                   (a = the live units of the layer below)
    A     core + sum_cold C_kk w_k w_k'                    (the file's old compact form: the cold variances, full Gram)
    B     core + diag(sum_cold C_kk w_ki^2)                (the cold variances on the diagonal only)
against the full product in float64; 'file' = the file's own float32 product against the same.
usage: probe_deadcov.py <candidate.py> <rank>   (one JSON line per layer, then a summary line)"""
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
mod = types.ModuleType('candidate_probe_deadcov'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
rows = []
parent = mod._flms_cov_star


def rel(x, y):
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def wrap(w, C, A, retain32=False):
    out = parent(w, C, A, retain32)
    layer = int(mod._PHYS_LAYER[0]) + 1          # _PHYS_LAYER holds the layer below at this call
    live = np.asarray(mod._E2_LAYER['ecut_cols']).astype(np.int64)
    gate = np.asarray(mod._E2_LAYER['g'], dtype=np.float64)
    W, Cn = np.asarray(w, dtype=np.float64), np.asarray(C, dtype=np.float64)
    n = W.shape[0]
    cold = np.setdiff1d(np.arange(n), live)
    full = W.T @ Cn @ W
    Wa = W[live]
    core = Wa.T @ Cn[np.ix_(live, live)] @ Wa
    dC = np.diag(Cn)[cold]
    Wc = W[cold]
    formA = core + (Wc * dC[:, None]).T @ Wc
    formB = core + np.diag((Wc * Wc * dC[:, None]).sum(axis=0))
    off = ~np.eye(n, dtype=bool)
    d = np.diag(full)
    row = dict(layer=layer, live=int(len(live)), above_floor=int((gate >= 0.01).sum()),
               cold_var_share=float(dC.sum() / np.trace(Cn)), file=rel(np.asarray(out[0], dtype=np.float64), full))
    for name, X in (('core', core), ('A', formA), ('B', formB)):
        row[name] = dict(F=rel(X, full), off=rel(X[off], full[off]),
                         diag_rms=float(np.sqrt(np.mean(((np.diag(X) - d) / d) ** 2))), diag_max=float(np.max(np.abs(np.diag(X) - d) / d)))
    rows.append(row)
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
print(json.dumps(dict(rank=rank, layers=len(rows), live_mean=float(np.mean([r['live'] for r in rows])),
                      live=[r['live'] for r in rows])), flush=True)
