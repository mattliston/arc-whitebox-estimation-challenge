"""TAG opus_engineer a42 (2026-10-02): store the head columns of one prediction as the estimator forms them: for every layer 1-14 and
every column the candidate puts, the raw per-unit vector, its unit mask and the standardised, clipped float32 column the first layer
receives (computed here by the trainer's rule from the raw vector, in numpy). For a column-by-column comparison with the torch
card's columns; no change of the prediction. In-process, one network, one thread.
usage: dump_columns.py <candidate.py built with the several-column fragment (lever HC@<weights with "columns">)> <rank> <out.npz (new)>"""
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
mod = types.ModuleType('candidate_column_dump'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
store = {}
put = mod._hc_put


def keep(l, name, vector, mask=None, count=None):
    if name in mod._HC_COLUMNS and 1 <= l <= 14:
        x = np.asarray(vector, dtype=np.float64).copy()
        m = np.ones(len(x), bool) if mask is None else np.asarray(mask).astype(bool)
        assert count is None or int(m.sum()) == int(count), (l, name, int(m.sum()), count)
        y = x[m]; z = np.zeros(len(x)); sd = y.std(ddof=1)
        if np.isfinite(sd) and sd > 0:
            z[m] = np.clip((y - y.mean()) / sd, -8, 8)
        store['raw_%s_%02d' % (name, l)] = x; store['mask_%s_%02d' % (name, l)] = m; store['z_%s_%02d' % (name, l)] = z.astype(np.float32)
    return put(l, name, vector, mask, count)


mod._hc_put = keep
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
assert np.isfinite(y).all()
np.savez(out, prediction=y.astype(np.float32), columns=np.asarray(mod._HC_COLUMNS), **store)
print(json.dumps(dict(done=True, arrays=len(store), columns=list(mod._HC_COLUMNS), candidate=candidate.name[-70:], rank=rank,
                      rows={name: [int(store['mask_%s_%02d' % (name, l)].sum()) if 'mask_%s_%02d' % (name, l) in store else 0 for l in range(1, 15)] for name in mod._HC_COLUMNS})))
