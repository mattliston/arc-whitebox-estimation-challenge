"""TAG opus_engineer a36 (2026-10-02): store what a fourth-order carrier would read from the native estimator, per layer 1-14, so that
its building blocks (the top eigenvectors of the pre-activation covariance by warm-started block power iteration, products with
the covariance and the weights) can be studied offline in numpy and compared with the torch prototypes without a native run:
  C_<l>     pre-activation covariance of layer l (float32, as the estimator forms it)
  aux_<l>   rows gate, host mean, sigma, alpha, pdf of layer l (float64, 5 x n; from the pair map's aux and its returned mean)
  radial_<l> the radial scalar entering layer l
In-process, one network, one thread. usage: dump_state.py <candidate.py> <rank> <out.npz (new)>"""
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
mod = types.ModuleType('candidate_state_dump'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
store = {}
pair = mod.nonlinear
calls = [0]


def keep(mu, C, K21, k4, k22_override=None, metric='identity', **kw):
    result = pair(mu, C, K21, k4, k22_override, metric, **kw)
    calls[0] += 1
    l = calls[0]                                    # the pair map `nonlinear` runs at layers 1..14, once each
    m, cov, post21, radial, aux = result
    store['C_%02d' % l] = np.asarray(C, dtype=np.float32)
    store['aux_%02d' % l] = np.stack([np.asarray(x, dtype=np.float64) for x in (aux['gate'], m, aux['sigma'], aux['alpha'], aux['pdf'])])
    store['radial_out_%02d' % l] = np.float64(radial)
    return result


mod.nonlinear = keep
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
assert calls[0] == 14 and np.isfinite(y).all(), calls
np.savez(out, **store)
print(json.dumps(dict(done=True, layers=calls[0], candidate=str(candidate), rank=rank)))
