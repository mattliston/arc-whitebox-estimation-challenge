"""TAG fable-lead 2026-10-20: probe of the learned FINAL HEAD on the native card: wraps final_delta(h, sigma, Q) to record its inputs and
output per network; prints the head's share (MSE with / without the head, LS scale of the head), and saves h / sigma / delta / y15 / truth
for an offline refit study. Store ranks use truth688, fresh seeds (>= 7000000) the 64M loader truths.
usage: native_head_probe_v1.py <candidate> <out prefix> <ranks or seeds comma-separated>"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, prefix, ids = Path(sys.argv[1]).resolve(), sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
REC = {}
_gs0 = mod.gnn_step      # the live terminal correction: scalar_terminal_hook -> gnn_step(15, x, sigma, w, state, P)
def gnn_step_probe(l, x, sigma, W, state, P):
    out = _gs0(l, x, sigma, W, state, P)
    REC.setdefault('deltas', {})[int(l)] = np.asarray(out[0], dtype=np.float64).copy()
    if int(l) == 15:
        REC['h'] = np.asarray(x, dtype=np.float64).copy(); REC['sigma'] = np.asarray(sigma, dtype=np.float64).copy(); REC['delta'] = np.asarray(out[0], dtype=np.float64).copy()
        REC['state15'] = np.asarray(out[1], dtype=np.float64).copy(); REC['state_in'] = np.asarray(state, dtype=np.float64).copy()
        REC['P15'] = {k: np.asarray(P[k], dtype=np.float64)[15].copy() if np.asarray(P[k]).ndim >= 1 and np.asarray(P[k]).shape[0] == 16 else np.asarray(P[k], dtype=np.float64).copy() for k in ('head', 'head_b', 'rms', 'gain') if k in P}
        REC['Q'] = {k: tuple(np.asarray(v).shape) for k, v in P.items()} if isinstance(P, dict) else str(type(P))
    return out
mod.gnn_step = gnn_step_probe
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'net-%d' % seed


for rid in ids:
    if rid >= 7000000:
        w = loader.weights(rid); t = loader.truth(rid, check_weights=False); W = [np.asarray(w[l]) for l in range(16)]
        A, B = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64); truth = (A + B) / 2
    else:
        Wn = np.load(STORE / Path(MAN[rid]['path']).parent / 'weights.npy'); W = [np.asarray(Wn[l]) for l in range(16)]
        truth = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rid:04d}.npz')['mc_post'].astype(np.float64)[15]; A = B = truth
    REC.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP(W, rid), 2**41), dtype=np.float64)
    y15 = y[15] if y.ndim == 2 else y; d = REC.get('delta'); m0 = y15 - d
    U = lambda p: float(np.mean((p - A) * (p - B)))
    al = float(((truth - m0) * d).sum() / (d * d).sum())
    print('net %d: U with head %.4e | without head %.4e (%+.1f%%) | head LS scale %.3f -> U %.4e (%+.2f%%) | |delta|/|m| %.4f | h %s Q %s | %.0fs' % (rid, U(y15), U(m0), 100 * (U(m0) / U(y15) - 1), al, U(m0 + al * d), 100 * (U(m0 + al * d) / U(y15) - 1), np.linalg.norm(d) / np.linalg.norm(y15), REC['h'].shape, REC.get('Q'), time.time() - t0), flush=True)
    np.savez_compressed('%s_%d.npz' % (prefix, rid), h=REC['h'], sigma=REC['sigma'], delta=d, y15=y15, m0=m0, A=A, B=B, truth=truth, deltas_norm=np.array([np.linalg.norm(REC['deltas'].get(k, np.zeros(1))) for k in range(16)]), state15=REC['state15'], state_in=REC['state_in'], **{'P15_' + k: v for k, v in REC['P15'].items()})
print('PROBE_DONE', flush=True)
