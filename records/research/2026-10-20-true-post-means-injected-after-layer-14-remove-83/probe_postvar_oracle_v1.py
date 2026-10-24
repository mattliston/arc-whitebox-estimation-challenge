"""TAG fable-lead 2026-10-20: POST-VARIANCE ORACLE per layer. Replaces the card's per-unit post-activation variance diagonal (the
`fnp.fill_diagonal(C, _ps_Cdiag)` of the layer loop, both the float64 C and its float32 copy) with the 64M-sample TRUE post variance
(loader truth mc_post_var[l]) at the listed layers, and measures the final cross-half U on fresh nets. Says how much final MSE each
layer's per-unit variance error carries (the covariance DIAGONAL only; the off-diagonal stays the card's).
usage: probe_postvar_oracle_v1.py <science candidate.py> <first seed> <count>"""
import json
import sys
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, first, count = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3])
src = candidate.read_text()
A = "                fnp.fill_diagonal(C, _ps_Cdiag)\n"
B = "                    fnp.fill_diagonal(_ps_post_cov32, _ps_Cdiag)\n"
assert src.count(A) == 4 and src.count(B) == 4
src = src.replace(A, "                _ps_Cdiag = _ORACLE_DIAG(l, _ps_Cdiag)\n" + A)
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
STATE = {'layers': set(), 'truth': None, 'hits': 0}


def oracle(l, d):
    if int(l) in STATE['layers'] and STATE['truth'] is not None:
        STATE['hits'] += 1
        import flopscope.numpy as fnp
        return fnp.asarray(STATE['truth'][int(l)].astype(np.float64))
    return d


mod._ORACLE_DIAG = oracle
exec(compile(src, str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops


class MLP:
    def __init__(self, w, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, seed, 'fresh-%d' % seed


SETS = [(), (14,), (13,), (12,), (10,), (8,), (12, 13, 14), tuple(range(8, 15)), tuple(range(1, 15))]
agg = {s: [] for s in SETS}
for seed in range(first, first + count):
    w = loader.weights(seed); t = loader.truth(seed, check_weights=False)
    A_, B_ = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64)
    STATE['truth'] = t['mc_post_var']
    us = {}
    for s in SETS:
        STATE['layers'] = set(s); STATE['hits'] = 0
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
        us[s] = float(np.mean((y[15] - A_) * (y[15] - B_)))
        if s and STATE['hits'] != len(s):
            print('  warning: oracle hits %d for layers %s' % (STATE['hits'], s))
    u0 = us[()]
    print('seed %d: base U %.4e | %s' % (seed, u0, ' | '.join('L%s %+.1f%%' % ('-'.join(map(str, s)) if len(s) < 4 else '%d..%d' % (s[0], s[-1]), 100 * (us[s] / u0 - 1)) for s in SETS[1:])), flush=True)
    for s in SETS: agg[s].append(100 * (us[s] / u0 - 1))
print('MEAN over %d seeds: %s' % (count, ' | '.join('L%s %+.1f%%' % ('-'.join(map(str, s)) if len(s) < 4 else '%d..%d' % (s[0], s[-1]), np.mean(agg[s])) for s in SETS[1:])))
print('ORACLE_DONE')
