"""Meter real S30 matrix kernels against one rectangular Hopcroft-Kerr level.

Run with the OpenBLAS interpreter through guard.py. This is a kernel screen,
not an estimator accuracy qualification or a claim about platform runtime.
"""
from pathlib import Path
import gc
import json
import statistics
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
out = HERE / 'hk233_bench_v2.json'
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
candidate = HERE / 'candidate_bundle_diagreuse_v1.py'
mod = types.ModuleType('hk233_bench'); mod.__file__ = str(candidate)
sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec'), mod.__dict__)
exec((HERE / 'hk233_fragment_v1.py').read_text(), mod.__dict__)
flops.configure(symmetry_warnings=False)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
    api_version='2', scratch_dir=None, submission_dir=None, seed=0))
for cache in (mod._IPT_A, mod._IPT_B, mod._IPT_C):
    mod._CAP_BY_ID[id(cache)] = 100000
def direct6(a, b):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    assert m % 64 == k % 64 == n % 64 == 0
    assert mod._ipt_fits(6, m, k, n), (m, k, n)
    mod._TR_LAST[0] = None
    out = fnp.empty((m, n), dtype=fnp.float32)
    mod._ipt_product_ab(a, b, out, 6)
    return out
rng = np.random.default_rng(20260926)
results = []
shapes = [(896, 384, 11776), (576, 11776, 384),
          (1024, 384, 8192), (896, 9984, 960)]
for m, k, n in shapes:
    aa = rng.normal(size=(m, k)).astype(np.float32)
    bb = rng.normal(size=(k, n)).astype(np.float32)
    ref = aa.astype(np.float64) @ bb.astype(np.float64)
    norm = np.linalg.norm(ref)
    a, b = fnp.asarray(aa), fnp.asarray(bb)
    del aa, bb
    rows = []
    for rep in range(3):
        methods = ('s30', 'hk6') if rep % 2 == 0 else ('hk6', 's30')
        for method in methods:
            mod._TR_LAST[0] = None
            mod._LVF_EAGER[0] = rep == 0
            t0 = time.process_time()
            with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                if method == 's30':
                    product = mod._shared_old384_mm(a, b)
                else:
                    product = mod._hk233_product(a, b, direct6, quantum=64)
            cpu = time.process_time() - t0
            mod._LVF_EAGER[0] = False
            got = np.asarray(product, dtype=np.float64)
            row = dict(shape=[m, k, n], method=method, rep=rep,
                cold=rep == 0, flops=int(ctx.flops_used), calls=len(ctx.op_log),
                cpu_s=cpu, backend_s=ctx.flopscope_backend_time_s,
                relative_error=float(np.linalg.norm(got - ref) / norm),
                max_abs_error=float(np.max(np.abs(got - ref))))
            rows.append(row); print(json.dumps(row), flush=True)
            del product, got
    results.extend(rows)
    base = [r for r in rows if r['method'] == 's30' and not r['cold']]
    for method in ('hk6',):
        trial = [r for r in rows if r['method'] == method and not r['cold']]
        summary = dict(shape=[m, k, n], method=method,
            bill_pct=100*(statistics.mean(r['flops'] for r in trial)/statistics.mean(r['flops'] for r in base)-1),
            cpu_ratio=statistics.median(r['cpu_s'] for r in trial)/statistics.median(r['cpu_s'] for r in base),
            extra_calls=statistics.mean(r['calls'] for r in trial)-statistics.mean(r['calls'] for r in base))
        print(json.dumps(summary), flush=True)
    del a, b, ref
    mod._TR_LAST[0] = None
    gc.collect()
with out.open('x') as f:
    json.dump(results, f, indent=2)
