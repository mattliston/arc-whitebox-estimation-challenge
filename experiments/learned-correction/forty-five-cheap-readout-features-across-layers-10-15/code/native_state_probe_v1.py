"""TAG fable-lead 2026-10-20: NATIVE head-input screen probe. Per store network, records per layer l = 1..15 what the learned mean
correction sees and does (gnn_step: x = the 30 per-unit features, state in / out, delta), the pair map's per-unit aux (pre v / k3 / k4,
post v / k3 / k4, alpha, gate, pdf, gaussian mean), the pair map's mean output before the head, row energies of the covariance /
K21 plane inputs and of the GN covariance correction, plus the all-layer output y (post-head means) and the truth688 post means.
Purpose: screen candidate NEW head inputs offline against the per-layer residual and the final residual (LOO by network).
usage: native_state_probe_v1.py <candidate> <out dir> <ranks comma-separated>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
candidate, outdir, ids = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')]
outdir.mkdir(parents=True, exist_ok=True)
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
REC = {}
f64 = lambda a: np.asarray(a, dtype=np.float64).copy()
_gs0 = mod.gnn_step
def gnn_step_probe(l, x, sigma, W, state, P):
    out = _gs0(l, x, sigma, W, state, P)
    l = int(l)
    if l >= 1:
        REC['x_%d' % l] = f64(x); REC['state_in_%d' % l] = f64(state); REC['state_out_%d' % l] = f64(out[1]); REC['delta_%d' % l] = f64(out[0]); REC['sigma_%d' % l] = f64(sigma)
    return out
mod.gnn_step = gnn_step_probe
_pm0 = mod.nonlinear
LAYER = [0]
def pm_probe(mu, C, K21, k4, *a, **kw):
    out = _pm0(mu, C, K21, k4, *a, **kw)
    LAYER[0] += 1; l = LAYER[0]      # nonlinear is called once per layer 1..14 in order (layer 0 = first_gaussian, 15 = terminal)
    aux = out[4]
    for k in ('pre_v', 'pre_k3', 'pre_k4', 'post_v', 'post_k3', 'post_k4', 'alpha', 'gate', 'pdf', 'gaussian_mean'):
        if k in aux: REC['%s_%d' % (k, l)] = f64(aux[k])
    REC['m0_%d' % l] = f64(out[0]); REC['mu_%d' % l] = f64(mu)
    Cn = np.asarray(C, dtype=np.float64); Kn = np.asarray(K21, dtype=np.float64)
    Co = Cn.copy(); np.fill_diagonal(Co, 0.0); Ko = Kn.copy(); np.fill_diagonal(Ko, 0.0)
    REC['crow2_%d' % l] = (Co ** 2).sum(1); REC['k21row2_%d' % l] = (Ko ** 2).sum(1); REC['k21col2_%d' % l] = (Ko ** 2).sum(0)
    REC['k21rowabs_%d' % l] = np.abs(Ko).sum(1); REC['k21colsum_%d' % l] = Ko.sum(0); REC['k21rowsum_%d' % l] = Ko.sum(1)
    Cp = np.asarray(out[1], dtype=np.float64); Cpo = Cp.copy(); np.fill_diagonal(Cpo, 0.0); REC['cpostrow2_%d' % l] = (Cpo ** 2).sum(1)
    return out
mod.nonlinear = pm_probe
if '_gn_dcov' in mod.__dict__:
    _gd0 = mod._gn_dcov
    def gn_dcov_probe(l, mu, Cpre, K21, k4, aux):
        d = _gd0(l, mu, Cpre, K21, k4, aux)
        dn = np.asarray(d, dtype=np.float64); REC['dcovrow2_%d' % int(l)] = (dn ** 2).sum(1); REC['dcovrowsum_%d' % int(l)] = dn.sum(1)
        return d
    mod._gn_dcov = gn_dcov_probe
_ns0 = mod.nonlinear_scalar
def ns_probe(mu, v, k3, k4, *a, **kw):
    out = _ns0(mu, v, k3, k4, *a, **kw)
    REC['mu_15'] = f64(mu); REC['pre_v_15'] = f64(v); REC['pre_k3_15'] = f64(k3) * np.ones(1024); REC['pre_k4_15'] = f64(k4) * np.ones(1024); REC['m0_15'] = f64(out[0])
    aux = out[1]
    for k in ('alpha', 'gate', 'pdf', 'gaussian_mean'):
        if k in aux: REC['%s_15' % k] = f64(aux[k])
    return out
mod.nonlinear_scalar = ns_probe
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed
for rank in ids:
    Wn = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(Wn[l]) for l in range(16)]
    tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    REC.clear(); LAYER[0] = 0; t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP(W, rank), 2**41), dtype=np.float64)
    REC['y'] = y; REC['truth'] = tp
    np.savez_compressed(outdir / ('sp_%d.npz' % rank), **{k: (v.astype(np.float32) if v.ndim == 2 else v) for k, v in REC.items()})
    print('rank %d FINAL %.4f e-9 all-layer %.4e layers seen %d wall %.0fs keys %d' % (rank, np.mean((y[15] - tp[15]) ** 2) * 1e9, np.mean((y - tp) ** 2), LAYER[0], time.time() - t0, len(REC)), flush=True)
print('PROBE_DONE', flush=True)
