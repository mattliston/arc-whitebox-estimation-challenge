"""TAG opus_engineer a42 debug (one network, in process): which of the final-layer column's functions run, and with what.
usage: debug_hcf.py <candidate.py> <rank>"""
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
mod = types.ModuleType('candidate_debug_hcf'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
calls = []
for name in ('_hcf_put', '_hcf_k4', '_hcf_bias', '_k4l_final', 'final_delta', '_l15_rowcut_k3'):
    f = getattr(mod, name)

    def wrap(*a, _f=f, _n=name, **k):
        before = dict(mod._HCF_STATE['raw']) if _n == '_hcf_bias' else None
        out = _f(*a, **k)
        info = dict(name=_n)
        if _n == '_hcf_bias':
            info.update(present=sorted(before), changed=out is not a[0])
        if _n == '_k4l_final':
            info.update(kind=type(out).__name__)
        calls.append(info)
        return out
    setattr(mod, name, wrap)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(json.dumps(dict(calls=calls[-12:], n_calls=len(calls), stats=dict(mod._HCF_STATS))))
