"""TAG fable-lead 2026-10-20: EXACT ATTRIBUTION of the final-layer error to the four inputs of the terminal mean formula. Wraps the
card's terminal scalar map (nonlinear_scalar, as armed by _k4l_final for layer 15) to capture (mu, v, k3, k4) and, with the 64M loader
truth of the same fresh seed, recomputes the terminal mean with each input replaced by its Monte Carlo truth (pre-activation mu, var, k3,
k4 at layer 15 from mc_cumulants; needs mc_cum_v1/mc_cum_seed<seed>.npz) and with all four true. Reports cross-half U of each variant.
usage: probe_l15attr_v1.py <science candidate.py> <seed> [<seed> ...]"""
import json
import sys
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, seeds = Path(sys.argv[1]).resolve(), [int(s) for s in sys.argv[2:]]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
CAP = {}
PARENT = mod.nonlinear_scalar


def wrapped(mu, v, k3, k4):
    if CAP.get('arm'):
        CAP['args'] = tuple(np.array(np.asarray(x, dtype=np.float64), copy=True) for x in (mu, v, k3, k4))
        CAP['arm'] = False
    return PARENT(mu, v, k3, k4)


mod.nonlinear_scalar = wrapped
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
import flopscope.numpy as fnp


class MLP:
    def __init__(self, w, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, seed, 'fresh-%d' % seed


def mean_formula(mu, v, k3, k4):
    m, aux = PARENT(fnp.asarray(mu), fnp.asarray(v), fnp.asarray(k3), fnp.asarray(k4))
    return np.asarray(m, dtype=np.float64)


print('seed | U(card) | U with true mu | true v | true k3 | true k4 | true mu+v | all four true | (percent change vs card)')
agg = {}
for seed in seeds:
    w = loader.weights(seed); t = loader.truth(seed, check_weights=False); M = np.load('mc_cum_v1/mc_cum_seed%d.npz' % seed)
    A, B = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64)
    CAP['arm'] = True
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
    mu, v, k3, k4 = CAP['args']
    # the card's final mean = formula + whatever the final hook adds; attribute via the formula difference
    base = mean_formula(mu, v, k3, k4); extra = y[15] - base
    U = lambda p: float(np.mean((p - A) * (p - B)))
    tm, tv, tk3, tk4 = M['mu'][15], M['var'][15], M['k3'][15], M['k4'][15]
    variants = {'card': (mu, v, k3, k4), 'mu': (tm, v, k3, k4), 'v': (mu, tv, k3, k4), 'k3': (mu, v, tk3, k4), 'k4': (mu, v, k3, tk4), 'mu+v': (tm, tv, k3, k4), 'all': (tm, tv, tk3, tk4)}
    us = {k: U(mean_formula(*a) + extra) for k, a in variants.items()}
    u0 = us['card']
    print('%d | %.4e | %s' % (seed, u0, ' | '.join('%s %+.1f%%' % (k, 100 * (us[k] / u0 - 1)) for k in ('mu', 'v', 'k3', 'k4', 'mu+v', 'all'))), flush=True)
    for k in us: agg.setdefault(k, []).append(100 * (us[k] / u0 - 1))
    print('   formula check: U(card formula) %.4e vs U(predict) %.4e; |extra| rms %.2e; card vs true: corr(mu) %.5f, v ratio %.4f, k3 scale %.3f, k4 ratio %.3f' % (U(base + extra), U(y[15]), np.sqrt(np.mean(extra**2)), np.corrcoef(mu, tm)[0, 1], np.mean(v / tv), float(np.dot(k3, tk3) / np.dot(k3, k3)), np.mean(k4) / np.mean(tk4)))
print('MEAN over %d seeds: %s' % (len(seeds), ' | '.join('%s %+.1f%%' % (k, np.mean(agg[k])) for k in ('mu', 'v', 'k3', 'k4', 'mu+v', 'all'))))
