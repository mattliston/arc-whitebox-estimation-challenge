"""TAG fable-lead 2026-10-20: native probe of the TERMINAL k3 rows for a sensitivity-ranked row cut. Records at the terminal
(nonlinear_scalar inputs) mu, v, k3, k4 and the output y15 vs truth, and at L14 the pair map's aux post_k3 / post_v / m (for a
pre-push proxy of k3_15 = diag transport sum_a W_ia^3 post_k3_a). Saves l15probe_rank<r>.npz. usage: native_l15_probe_v1.py <candidate> <rank>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
if rank < 100000:
    W = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(W[l]) for l in range(16)]
    tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
else:
    sys.path.insert(0, str(O / 'analyst/fresh')); import loader
    W, tp = loader.load(rank)
REC = {}
_pm0 = mod.nonlinear
def pm_probe(mu, C, K21, k4, *a, **kw):
    out = _pm0(mu, C, K21, k4, *a, **kw)
    aux = out[4]
    REC['l14_post_k3'] = np.asarray(aux['post_k3'], dtype=np.float64).copy(); REC['l14_post_v'] = np.asarray(aux['post_v'], dtype=np.float64).copy()
    REC['l14_m'] = np.asarray(out[0], dtype=np.float64).copy(); REC['l14_pre_k3'] = np.asarray(aux['pre_k3'], dtype=np.float64).copy()
    REC['l14_mu'] = np.asarray(mu, dtype=np.float64).copy(); REC['l14_v'] = np.asarray(aux['pre_v'], dtype=np.float64).copy()
    return out
mod.nonlinear = pm_probe
_ns0 = mod.nonlinear_scalar
def ns_probe(mu, v, k3, k4, *a, **kw):
    REC['mu15'] = np.asarray(mu, dtype=np.float64).copy(); REC['v15'] = np.asarray(v, dtype=np.float64).copy()
    REC['k3_15'] = (np.asarray(k3, dtype=np.float64) * np.ones(1024)).copy(); REC['k4_15'] = (np.asarray(k4, dtype=np.float64) * np.ones(1024)).copy()
    return _ns0(mu, v, k3, k4, *a, **kw)
mod.nonlinear_scalar = ns_probe
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed
mlp = MLP(W, rank)
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
y15 = y[15] if y.ndim == 2 else y
REC['y15'] = y15; REC['truth15'] = tp[15]; REC['W15'] = np.asarray(W[15], dtype=np.float64)
np.savez_compressed('l15probe_rank%d.npz' % rank, **REC)
print('rank %d FINAL %.4f e-9 wall %.0fs keys %s' % (rank, np.mean((y15 - tp[15]) ** 2) * 1e9, time.time() - t0, sorted(REC)), flush=True)
