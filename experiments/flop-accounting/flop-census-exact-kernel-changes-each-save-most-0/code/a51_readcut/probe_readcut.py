"""TAG opus_engineer a51 probe (2026-10-10; lead 16:2xZ, bill for first place): how concentrated the young plane READ is over its columns. The
read is young_plane = (left[:, young] * lam) @ right' (left = the response rows of pp^2 with the pair law folded in, right = pp's young columns
on the live rows); column r adds lam_r left_r right_r' with Frobenius size c_r = |lam_r| |left_r| |right_r|. Per layer: K, the share of columns
needed for 99 / 99.9 / 99.99% of sum c, and the ACTUAL relative Frobenius error of the young plane and of the whole plane (young + old)
when the read skips the columns outside each set (float64 numpy on the side; the prediction is the parent's).
usage: probe_readcut.py <candidate.py> <rank>"""
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
mod = types.ModuleType('candidate_probe_readcut'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
parent = mod._cp_read_young_sq


def wrap(left, young_right, lam, live_indices, state):
    out = parent(left, young_right, lam, live_indices, state)
    layer = int(mod._PHYS_LAYER[0]) + 1
    old = int(state['columns'])
    L = np.asarray(left, dtype=np.float64)[:, old:]
    R = np.asarray(young_right, dtype=np.float64)
    lm = np.asarray(lam, dtype=np.float64)[old:]
    full = (L * lm) @ R.T
    whole = np.asarray(out, dtype=np.float64)
    c = np.abs(lm) * np.linalg.norm(L, axis=0) * np.linalg.norm(R, axis=0)
    order = np.argsort(-c)
    cum = np.cumsum(c[order]) / c.sum()
    row = dict(layer=layer, K=int(len(c)), old=old, young_share=float(np.linalg.norm(full) / max(np.linalg.norm(whole), 1e-300)))
    for f in (0.99, 0.999, 0.9999):
        k = int(np.searchsorted(cum, f) + 1)
        keep = order[:k]
        part = (L[:, keep] * lm[keep]) @ R[:, keep].T
        row['f%g' % f] = dict(kept=k, frac=round(k / len(c), 4), young_err=float(np.linalg.norm(part - full) / np.linalg.norm(full)),
                             plane_err=float(np.linalg.norm(part - full) / max(np.linalg.norm(whole), 1e-300)))
    print(json.dumps(row), flush=True)
    return out


mod._cp_read_young_sq = wrap
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(json.dumps(dict(rank=rank, done=True)), flush=True)
