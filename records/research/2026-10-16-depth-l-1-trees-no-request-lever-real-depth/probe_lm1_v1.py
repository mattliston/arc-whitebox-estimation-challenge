"""[opus-lm1] empirical L-1 price of the alt-basis tree product (_ipt_product_ab) per key, in process (local flopscope, exact flops and op
records = requests incl. views): setup from the captured boundary manifest, then the real predictions of ranks 35 and 100 (warm caches as in
production), then per key (m, k, n, L): two calls at L and two at L-1 on fresh float32 operands (the second call = steady state), plus the
float32 max-abs difference of the two outputs (Strassen depth changes rounding only). usage: probe_lm1_v1.py <candidate.py> <keys.json> <out.json>
env: CAPTURED_SETUP_MANIFEST, CANDIDATE_SHA256, BLAS threads 1"""
import hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline')
import captured_setup_replay_generic_v1 as replay

P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
candidate, keys, out = Path(sys.argv[1]).resolve(), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]).resolve()
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256'] and not out.exists()
store = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_lm1'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec'), mod.__dict__)
expected = replay.install(mod, os.environ['CAPTURED_SETUP_MANIFEST'])


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'phase2-{rank}'


est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
for rank in (35, 100):
    w = np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        est.predict(MLP([w[l] for l in range(16)], rank), 2**41)
rng = np.random.default_rng(0)
res = []
for m, k, n, L in keys:
    row = dict(m=m, k=k, n=n, L=L)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        a = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); b = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
        outs = {}
        for lev in (L, L - 1):
            for rep in (0, 1):
                o = fnp.empty((m, n), dtype=fnp.float32)
                n0, f0, t0 = len(ctx.op_log), int(ctx.flops_used), time.perf_counter()
                try:
                    mod._ipt_product_ab(a, b, o, lev)
                    row[f'L{lev}_{rep}'] = dict(ops=len(ctx.op_log) - n0, flops=int(ctx.flops_used) - f0, s=round(time.perf_counter() - t0, 4))
                    outs[lev] = np.asarray(o)
                except Exception as exc:
                    row[f'L{lev}_{rep}'] = dict(error=f'{type(exc).__name__}: {str(exc)[:120]}')
        if L in outs and L - 1 in outs:
            ref = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
            row['maxabs_L_vs_ref'] = float(np.max(np.abs(outs[L] - ref))); row['maxabs_Lm1_vs_ref'] = float(np.max(np.abs(outs[L - 1] - ref)))
    res.append(row); print(json.dumps(row), flush=True)
out.write_text(json.dumps(res, indent=1) + '\n')
print('LM1_PROBE_DONE', flush=True)
