"""TAG opus_engineer a44 probe v3 (2026-10-10; lead 03:0x: cut only the units the push's dead side cuts, alpha < -3.5): the covariance
sandwich's dead cut at several gate floors. For each floor f: the roster = the units of the layer below with gate >= f, by gate, the
count rounded up to 64 (m); the cut form W_a' C_aa W_a against the full product (float64 numpy on the side; the prediction is the
parent's). Floors 0.01 (the live roster, alpha -2.33), 1e-3 (-3.09), Phi(-3.5) = 2.33e-4 (the dead side of the push), Phi(-4) = 3.17e-5.
Per layer and floor: m, the units under the floor, C_pre's error (Frobenius, off-diagonal, diagonal rms).
usage: probe_deadcov_v3.py <candidate.py> <rank>"""
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
mod = types.ModuleType('candidate_probe_deadcov3'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
parent = mod._flms_cov_star
FLOORS = (('a2.33', 0.01), ('a3.09', 1e-3), ('a3.5', 2.326290790355e-4), ('a4.0', 3.167124183311e-5))


def wrap(w, C, A, retain32=False):
    out = parent(w, C, A, retain32)
    layer = int(mod._PHYS_LAYER[0]) + 1
    gate = np.asarray(mod._E2_LAYER['g'], dtype=np.float64)
    n = gate.shape[0]
    W, Cn = np.asarray(w, dtype=np.float64), np.asarray(C, dtype=np.float64)
    full = None
    off = ~np.eye(n, dtype=bool)
    row = dict(layer=layer)
    order = np.argsort(-gate, kind='stable')
    for name, floor in FLOORS:
        under = int((gate < floor).sum())
        m = min(n, (n - under + 63) // 64 * 64)
        if m >= n:
            row[name] = dict(under=under, m=m)
            continue
        if full is None:
            full = W.T @ Cn @ W
            d = np.diag(full)
        a = order[:m]
        core = W[a].T @ Cn[np.ix_(a, a)] @ W[a]
        row[name] = dict(under=under, m=m, F=float(np.linalg.norm(core - full) / np.linalg.norm(full)),
                         off=float(np.linalg.norm((core - full)[off]) / np.linalg.norm(full[off])),
                         drms=float(np.sqrt(np.mean(((np.diag(core) - d) / d) ** 2))))
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
