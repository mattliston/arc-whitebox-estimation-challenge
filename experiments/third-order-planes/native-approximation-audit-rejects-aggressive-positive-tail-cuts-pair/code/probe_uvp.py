"""TAG opus_engineer a53 probe (2026-10-10): how much pair-mean energy the u/v split's r_u truncation drops, and how much a better choice of
the kept coordinates would drop. At every split (F: rank x cols in roster order, the pairs' t and s columns side by side per cohort):
F_u = (F_t + F_s) / 2 (rank x P); e_k = sum_p F_u[k, p]^2 (unweighted; the frame is orthonormal at its join). Prints the dropped share of
sum e for: the leading-r_u rule (today), the best permutation (the r_u largest e_k), and the best rotation (the SVD of F_u).
float64 numpy on the side; the prediction is the parent's.
usage: probe_uvp.py <candidate.py> <rank>"""
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
mod = types.ModuleType('candidate_probe_uvp'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
parent = mod._uvp_split


def wrap(F, blocks):
    out = parent(F, blocks)
    ru = int(out[3])
    A = np.asarray(F, dtype=np.float64)
    Fu = np.concatenate([(A[:, a:a + h] + A[:, a + h:a + 2 * h]) / 2 for a, h in blocks], axis=1)
    Fv = np.concatenate([(A[:, a:a + h] - A[:, a + h:a + 2 * h]) / 2 for a, h in blocks], axis=1)
    e = np.sum(Fu * Fu, axis=1)
    tot = float(e.sum())
    s = np.linalg.svd(Fu, compute_uv=False) ** 2
    row = dict(layer=int(mod._UVP_CUR[0]), rank=int(A.shape[0]), P=int(Fu.shape[1]), ru=ru,
               drop_leading=float(e[ru:].sum() / tot), drop_best_perm=float(np.sort(e)[:len(e) - ru].sum() / tot),
               drop_svd=float(s[ru:].sum() / s.sum()), u_share=float(tot / (tot + float(np.sum(Fv * Fv)))),
               kept_in_leading_of_best=int(np.sum(np.argsort(-e)[:ru] < ru)))
    print(json.dumps(row), flush=True)
    return out


mod._uvp_split = wrap
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
print(json.dumps(dict(rank=rank, done=True)), flush=True)
