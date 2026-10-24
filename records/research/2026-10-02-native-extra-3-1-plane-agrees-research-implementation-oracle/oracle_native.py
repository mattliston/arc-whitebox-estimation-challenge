"""TAG opus_engineer a34 (2026-10-02): the native pair map's extra (3,1) plane (its `extra31` argument, standardised inside the pair
map as x31_ij = plane_ij / (sigma_i^3 sigma_j), diagonal zero, added to T31) fed with the physicist's ORACLE matrix, for parity
against his torch arms K31f / K31r64 / K31r32 (O/physicist/k4l_pred_rank<rank>_ora.npz; ntor.py: remainder31 = k31 (3/n) R_ij / (sigma_i^3 sigma_j)
at layers 1-14, R_l = t19_oracle_rank<rank>.npz['R_L<l>'] symmetrised; rank r = its r eigen-directions of largest |eigenvalue|).
The candidate file is not changed: _T31_LAYERS and _t31_extra are set on the loaded module (the plane is built outside the budget).
In-process, one network, one thread. usage: oracle_native.py <candidate.py> <rank> <oracle.npz> <k31> <eig rank, 0 = full> <out.npy (new)>"""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, rank, oracle, k31, cut, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), np.load(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
assert not out.exists()
planes = {}
for l in range(1, 15):
    M = oracle['R_L%d' % l].astype(np.float64); M = (M + M.T) / 2
    if cut:
        lam, V = np.linalg.eigh(M); o = np.argsort(-np.abs(lam))[:cut]; M = (V[:, o] * lam[o][None, :]) @ V[:, o].T
    planes[l] = k31 * (3.0 / 1024) * M
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_r31_oracle'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
used = []


def extra(l, ctx):
    used.append(int(l))
    return fnp.asarray(planes[int(l)])


mod._T31_LAYERS = tuple(planes)
mod._t31_extra = extra
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    bill = int(ctx.summary_dict()['flops_used'])
truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
np.save(out, np.asarray(y, dtype=np.float32))
print(json.dumps(dict(done=True, rank=rank, k31=k31, eig_rank=cut, layers=used, flops=bill, mse=float(np.mean((y[15] - truth) ** 2)), finite=bool(np.isfinite(y).all()))))
