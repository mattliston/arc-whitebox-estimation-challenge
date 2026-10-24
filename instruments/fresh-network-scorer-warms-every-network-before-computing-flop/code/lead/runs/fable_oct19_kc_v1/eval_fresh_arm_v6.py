"""TAG fable-lead 2026-10-20: v6 = v5 + per-network warm-up and high-precision rows; v5 = v4 + int layer keys for per-layer dicts (JSON gives str keys; _GN_KC.get(l) etc. silently miss). v4 = v3 + deepcopy of every setting value before setattr (cache dicts passed as settings get filled with arrays by the estimator, which broke json.dumps). v3 = eval_fresh_arm_v2 + a WARM-UP predict of the first setting before the loop (the first predict of a process
pays the cold frame-binding bill, ~0.44% on 7100005, which biased every v2 bill delta by that amount); v2 = (adjusted = U x bill, vs the base setting).
In-process OUT-OF-SAMPLE evaluation of an arm on fresh nets against the 64M-sample loader truths (cross-half U = mean((p-A)(p-B))
on the final layer). Loads the arm module once; for each seed predicts under each setting of the listed module globals (first
setting = base; settings do NOT reset, so each one restates every knob it needs).
usage: eval_fresh_arm_v2.py <candidate> <first seed> <count> <settings json: [{"NAME": value, ...}, ...]>"""
import copy
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
# Native value screens use the same official tariff/API as the judge.
# Parent-authorized bootstrap only: candidate source and settings stay intact.
OFFICIAL_FLOPSCOPE = O / 'lead/runs/astra_crunch_runtime_v1/failure336631_v1/flopscope_0.12.3'
assert not any(name.startswith('flopscope') for name in sys.modules)
assert not os.environ.get('FLOPSCOPE_WEIGHTS_FILE'), 'Official packaged tariff required'
assert os.environ.get('FLOPSCOPE_DISABLE_WEIGHTS', '').lower() not in ('1', 'true', 'yes', 'on')
sys.path.insert(0, str(OFFICIAL_FLOPSCOPE))
import flopscope as flops
assert flops.__version__.startswith('0.12.3+'), flops.__version__
assert Path(flops.__file__).resolve().parent == OFFICIAL_FLOPSCOPE / 'flopscope'
flops.configure(symmetry_warnings=False)
print('OFFICIAL_FLOPSCOPE %s %s' % (flops.__version__, flops.__file__), flush=True)
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, first, count, settings = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
def intkeys(v):
    """v5: JSON settings carry string keys; the card's per-layer dicts are looked up with int layers, so convert digit keys back."""
    if isinstance(v, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k): intkeys(x) for k, x in v.items()}
    if isinstance(v, list):
        return [intkeys(x) for x in v]
    return v


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'fresh-%d' % seed


tot = [[] for _ in settings]; bills = [[] for _ in settings]
for seed in range(first, first + count):
    w = loader.weights(seed); t = loader.truth(seed, check_weights=False)
    A, B = t['mc_post_A'][15].astype(np.float64), t['mc_post_B'][15].astype(np.float64)
    for k, v in settings[0].items():
        setattr(mod, k, intkeys(copy.deepcopy(v)))
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as warm_ctx:
        est.predict(MLP([w[l] for l in range(16)], seed), 2**41)
    print('warm-up seed %d bill %.12e' % (seed, float(warm_ctx.flops_used)), flush=True)
    us, fl = [], []
    for st in settings:
        for k, v in st.items():
            setattr(mod, k, intkeys(copy.deepcopy(v)))   # v5: int layer keys; v4: deepcopy keeps the printed setting pristine
        t0 = time.time()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
        us.append(float(np.mean((y[15] - A) * (y[15] - B)))); fl.append(float(ctx.flops_used))
        print('  seed %d %s U %.12e bill %.12e (%.0f s)' % (seed, json.dumps(st), us[-1], fl[-1], time.time() - t0), flush=True)
    for i in range(len(settings)):
        tot[i].append(us[i]); bills[i].append(fl[i])
    print('seed %d: base U %.4e bill %.4e | %s' % (seed, us[0], fl[0], ' | '.join('%s U %+.3f%% bill %+.3f%% adj %+.3f%%' % (
        json.dumps(st), 100 * (u / us[0] - 1), 100 * (f / fl[0] - 1), 100 * (u * f / (us[0] * fl[0]) - 1)) for st, u, f in zip(settings[1:], us[1:], fl[1:]))), flush=True)
print('MEAN over %d seeds: %s' % (count, ' | '.join('%s U %+.3f%% bill %+.3f%% adj %+.3f%% (U lower %d/%d)' % (
    json.dumps(st), 100 * (np.mean(tot[i]) / np.mean(tot[0]) - 1), 100 * (np.mean(bills[i]) / np.mean(bills[0]) - 1),
    100 * (np.mean(tot[i]) * np.mean(bills[i]) / (np.mean(tot[0]) * np.mean(bills[0])) - 1),
    sum(1 for a, b in zip(tot[i], tot[0]) if a < b), count) for i, st in enumerate(settings) if i)))
print('EVAL_DONE')
