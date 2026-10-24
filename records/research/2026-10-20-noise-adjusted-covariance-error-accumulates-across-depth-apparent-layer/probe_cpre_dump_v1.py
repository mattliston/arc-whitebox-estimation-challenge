"""TAG fable-lead 2026-10-20: dump the card's PRE-activation covariance Cpre at layers 1-14 for one store network (wraps the live pair
map, which receives Cpre as its second argument) and compare with the Monte Carlo truth cov_pre_l: |dC|_F/|Ct|_F per layer, the diag
share, the top eigen share and the |cos| of the top error mode with mu. Purpose: WHERE is the covariance error born?
usage: probe_cpre_dump_v1.py <sci candidate> <rank>"""
import json, sys, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.load(open(STORE / 'manifest.json'))['roles']['observer_fit']['row_manifests']
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
p = str(STORE / MAN[rank]['path']); p = p[:-len('/row_manifest.json')] if p.endswith('row_manifest.json') else p
W = np.load(p + '/weights.npy')
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
CAP = {}
orig = mod.nonlinear
def wrapped(mu, C, *a, **k):
    l = int(mod._PHYS_LAYER[0]) if mod._PHYS_LAYER[0] is not None else -1
    CAP[l] = (np.asarray(mu, dtype=np.float64).copy(), np.asarray(C, dtype=np.float64).copy())
    return orig(mu, C, *a, **k)
mod.nonlinear = wrapped
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    y = np.asarray(est.predict(MLP([W[l] for l in range(16)], rank), 2**41), dtype=np.float64)
T = np.load(P2 / ('fable_covhunter_v1/mcall_rank%04d_P32M.npz' % rank))
out = {}
for l in sorted(CAP):
    key = 'cov_pre_%d' % l
    if key not in T.files: continue
    mu, Cc = CAP[l]; Ct = np.asarray(T[key], dtype=np.float64); dC = Cc - Ct
    fro2 = (dC ** 2).sum(); ev, EV = np.linalg.eigh(dC); o = np.argsort(-np.abs(ev)); u0 = EV[:, o[0]]
    vt = np.diag(Ct); dv = np.diag(dC)
    print('L%2d |dC|/|Ct| %.4f  diag rms/vt rms %.4f  top eig share %.3f sign %+d  |cos(u0,mu)| %.2f  rel dv mean %+.5f' % (
        l, np.sqrt(fro2 / (Ct ** 2).sum()), np.sqrt((dv ** 2).sum() / (vt ** 2).sum()), ev[o[0]] ** 2 / fro2, int(np.sign(ev[o[0]])), abs(float(u0 @ mu) / np.linalg.norm(mu)), (dv / vt).mean()), flush=True)
    out['Cpre_%d' % l] = Cc.astype(np.float32); out['mu_%d' % l] = mu
np.savez(str(candidate.parent / ('cpre_dump_rank%d_v1.npz' % rank)), **out)
print('DUMP_DONE')
