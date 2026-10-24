"""TAG fable-lead 2026-10-20: one in-process predict of the shipped card on a store net with the flopscope summary: flops by dtype / op kind
(f64 is billed 2x f32 by the platform's weights). usage: flops_by_dtype_v1.py <candidate> <rank>"""
import json, sys, types, collections
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'); P2 = O.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__: mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops
W = np.load(STORE / Path(MAN[rank]['path']).parent / 'weights.npy'); W = [np.asarray(W[l]) for l in range(16)]
class MLP:
    def __init__(self, weights, seed): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'store-%d' % seed
for i in range(2):      # second predict = warm (no lazy binds)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP(W, rank), 2**41)
print('flops_used %.4e util %.5f' % (ctx.flops_used, ctx.flops_used / 2 ** 41))
sd = ctx.summary_dict() if callable(getattr(ctx, 'summary_dict', None)) else None
if sd: print('summary_dict keys:', list(sd)[:30])
log = getattr(ctx, 'op_log', None)
print('op_log type', type(log).__name__, 'len', len(log) if hasattr(log, '__len__') else '?')
if log:
    e = log[0]; print('entry type', type(e).__name__, 'fields:', (list(e.keys()) if isinstance(e, dict) else [a for a in dir(e) if not a.startswith('_')][:30]))
    def get(e, k):
        return e.get(k) if isinstance(e, dict) else getattr(e, k, None)
    by = collections.Counter(); byop = collections.Counter(); n = 0
    for e in log:
        f = float(get(e, 'flop_cost') or 0); dt = str(get(e, 'resolved_dtype') or '?'); op = str(get(e, 'op_name') or '?')
        by[dt] += f; byop[(op, dt)] += f; n += 1
    tot = sum(by.values()) or 1
    print('flops by dtype:', {k: '%.1f%%' % (100 * v / tot) for k, v in by.most_common()})
    print('top ops:', [(k, '%.1f%%' % (100 * v / tot)) for k, v in byop.most_common(16)])
    # the largest f64 ops by shape
    big = collections.Counter()
    for e in log:
        if 'float64' in str(get(e, 'resolved_dtype')): big[(str(get(e, 'op_name')), str(get(e, 'shapes'))[:60])] += float(get(e, 'flop_cost') or 0)
    print('top f64 ops by shape:', [(k, '%.2f%%' % (100 * v / tot)) for k, v in big.most_common(12)])
    cnt = collections.Counter(str(get(e, 'op_name')) for e in log); print('op counts (requests):', cnt.most_common(12), 'total', len(log))
    ns = collections.Counter(); nsn = collections.Counter()
    for e in log:
        if str(get(e, 'op_name')) in ('add', 'subtract', 'multiply'): k = (str(get(e, 'op_name')), str(get(e, 'namespace'))[-50:], str(get(e, 'shapes'))[:48]); ns[k] += float(get(e, 'flop_cost') or 0); nsn[k] += 1
    print('ADD/SUB/MUL by site:'); [print('   %5.2f%% x%4d %s' % (100 * v / tot, nsn[k], k)) for k, v in ns.most_common(22)]
    mm = collections.Counter(); mmn = collections.Counter()
    for e in log:
        if str(get(e, 'op_name')) == 'matmul': k = (str(get(e, 'namespace'))[-50:], str(get(e, 'shapes'))[:48]); mm[k] += float(get(e, 'flop_cost') or 0); mmn[k] += 1
    print('MATMUL by site:'); [print('   %5.2f%% x%4d %s' % (100 * v / tot, mmn[k], k)) for k, v in mm.most_common(16)]
