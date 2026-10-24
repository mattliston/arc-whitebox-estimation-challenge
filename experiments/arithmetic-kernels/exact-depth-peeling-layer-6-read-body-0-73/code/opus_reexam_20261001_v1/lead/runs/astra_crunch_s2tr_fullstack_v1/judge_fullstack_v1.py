"""[opus-kinkoracle] v2 (+dump mode) one store rank, one mode: setup, predict once (cold), noise-free cross-half U of the final layer vs the rank's 32M A/B truths.
modes: ctl (no oracle) | v<c> (true L14 pre-activation variance on kink units |a14| < c; c = 'all' for every unit).
usage: oracle_judge_v1.py <candidate.py> <rank> <mode> <out.json>"""
import json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
STORE = O.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
cand, rank, mode, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]); assert not out.exists(); assert mode=='ctl'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_ko'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
A = np.load(O / f'analyst/fresh/pu/mps_truth_rank{rank:04d}_A32M.npz'); B = np.load(O / f'analyst/fresh/pu/mps_truth_rank{rank:04d}_B32M.npz')
if mode == 'dump':
    mod._KO_ORACLE = dict(layer=14, dump=str(out).replace('.json', '_dump.npz'))
elif mode != 'ctl':
    c = float('inf') if mode == 'vall' else float(mode[1:])
    mod._KO_ORACLE = dict(layer=14, v=0.5 * (A['mc_pre_v'][14] + B['mc_pre_v'][14]), c=c)
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = np.load(STORE / f'rows/observer_fit/{rank:04d}/weights.npy')
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'store-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx: y = est.predict(mlp, 2**41)
y = np.asarray(y, dtype=np.float64); yf = y[-1] if y.ndim == 2 else y
a, b = A['mc_post'][15].astype(np.float64), B['mc_post'][15].astype(np.float64)
U = float(np.mean((yf - a) * (yf - b)))
res = dict(rank=rank, mode=mode, U=U, flops=int(ctx.summary_dict()['flops_used']), ko=dict(getattr(mod,'_KO_STATS',{})), y_shape=list(y.shape))
res['bini'] = getattr(mod, '_BIN_STATS', {})
res['peel'] = getattr(mod, '_PEEL_STATS', {})
res['w3'] = getattr(mod, '_W3R_STATS', {})
res['peel_budget'] = getattr(mod, '_PEEL_BUDGET_STATS', {})
res['rtt_reads'] = getattr(mod, '_RTT_STATS', {}).get('reads', [])
res['source_sha'] = __import__('hashlib').sha256(cand.read_bytes()).hexdigest()
np.save(out.with_suffix('.pred.npy'), yf)
out.write_text(json.dumps(res)); print({k:v for k,v in res.items() if k not in ('rtt_reads',)})
