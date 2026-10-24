"""TAG fable-lead 2026-10-20: PER-LAYER / PER-MODULE FLOP PROFILE of a candidate on one fresh net (in process, flopscope bill).
Wraps module-level physics entry points by name (call-time lookup) and records the bill inside each call (inclusive) and at every
pair-map call (layer boundary). usage: probe_layer_bill_v1.py <candidate> <seed> [names,comma,separated]"""
import json, sys, time, types
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
names = sys.argv[3].split(',') if len(sys.argv) > 3 else ['gnn_step', '_gn_dcov', '_astra_bcc_push', '_message_edge_source', '_np_update', '_k4l_local', '_f4c_birth', 'star_push', '_slr_reconstruct', '_push_bank_shared_newborn', 'nonlinear_scalar', 'scalar_terminal_hook']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
import flopscope as flops
CTX = [None]; REC = []; BOUND = []; DEPTH = [0]
def used():
    return float(CTX[0].flops_used) if CTX[0] is not None else 0.0
def layer():
    try:
        return int(mod._PHYS_LAYER[0]) if mod._PHYS_LAYER[0] is not None else -1
    except Exception:
        return -1
def wrap(name):
    orig = getattr(mod, name, None)
    if orig is None or not callable(orig):
        print('no such callable', name); return
    def w(*a, **k):
        f0 = used(); t0 = time.time(); DEPTH[0] += 1
        try:
            return orig(*a, **k)
        finally:
            DEPTH[0] -= 1
            REC.append((name, layer(), used() - f0, time.time() - t0, DEPTH[0]))
    w.__name__ = name; setattr(mod, name, w)
for n in names:
    wrap(n)
orig_nl = mod.nonlinear
def nl(*a, **k):
    BOUND.append((layer(), used(), time.time()))
    return orig_nl(*a, **k)
mod.nonlinear = nl
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'fresh-%d' % seed
w = loader.weights(seed); t = loader.truth(seed, check_weights=False)
A, B = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64)
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx
    y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
    total = float(ctx.flops_used)
CTX[0] = None
U = float(np.mean((y[15] - A) * (y[15] - B)))
print('seed %d U %.4e total bill %.6e util %.6f wall %.0f s' % (seed, U, total, total / 2 ** 41, time.time() - t0))
print('LAYER BOUNDARIES (bill at each pair-map call; delta = bill of the work between consecutive calls, attributed to the EARLIER layer):')
prev = 0.0; tprev = t0
for i, (l, f, tt) in enumerate(BOUND):
    nxt = BOUND[i + 1][1] if i + 1 < len(BOUND) else total
    print('  L%2d  up to pair map %.4e (%.2f%%)  work after this pair map until the next: %.4e (%.2f%%)' % (l, f - prev, 100 * (f - prev) / total, nxt - f, 100 * (nxt - f) / total))
    prev = f
print('MODULES (inclusive bill per call, summed by name and layer; depth 0 = not nested in another wrapped call):')
agg = {}
for name, l, f, dt, d in REC:
    k = (name, l); a = agg.setdefault(k, [0, 0.0, 0.0, set()]); a[0] += 1; a[1] += f; a[2] += dt; a[3].add(d)
bynm = {}
for (name, l), (n, f, dt, ds) in sorted(agg.items()):
    bynm.setdefault(name, [0, 0.0, 0.0]); bynm[name][0] += n; bynm[name][1] += f; bynm[name][2] += dt
    print('  %-26s L%2d calls %3d bill %.4e (%.3f%%) time %.2fs depths %s' % (name, l, n, f, 100 * f / total, dt, sorted(ds)))
print('TOTAL BY MODULE:')
for name, (n, f, dt) in sorted(bynm.items(), key=lambda x: -x[1][1]):
    print('  %-26s calls %4d bill %.4e (%.3f%%) time %.1fs' % (name, n, f, 100 * f / total, dt))
json.dump(dict(seed=seed, U=U, total=total, bound=[(l, f) for l, f, _ in BOUND], rec=[(n, l, f, dt, d) for n, l, f, dt, d in REC]), open(str(candidate.parent / ('layer_bill_seed%d_v1.json' % seed)), 'w'))
print('PROBE_DONE')
