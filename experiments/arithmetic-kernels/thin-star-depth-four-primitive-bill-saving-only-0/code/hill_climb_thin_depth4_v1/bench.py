"""Meter the retained card's existing thin-product kernels; no network data."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
protocol = json.loads((HERE / 'protocol.json').read_text())
source = Path(protocol['parent'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['parent_sha256']
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('kernel_bench')
mod.__file__ = str(source)
sys.modules[mod.__name__] = mod
exec(compile(source.read_text(), str(source), 'exec'), mod.__dict__)
flops.configure(symmetry_warnings=False)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
    api_version='2', scratch_dir=None, submission_dir=None, seed=0))
rng = np.random.default_rng(20260928)
rows = []
with flops.BudgetContext(flop_budget=2**44, quiet=True) as ctx:
    for shape in protocol['shapes']:
        m, k, n = shape
        an = (rng.standard_normal((m, k)) / np.sqrt(k)).astype(np.float32)
        bn = rng.standard_normal((k, n)).astype(np.float32)
        reference = an.astype(np.float64) @ bn.astype(np.float64)
        a, b = fnp.asarray(an), fnp.asarray(bn)
        outputs = {depth: fnp.empty((m, n), dtype=fnp.float32) for depth in (3, 4)}
        def measure(depth, phase, repetition):
            assert mod._ipt_fits(depth, m, k, n)
            mod._TR_LAST[0] = None
            f0, q0 = ctx._flops_used, len(ctx._op_log)
            t0 = time.process_time()
            out = mod._ipt_product_ab(a, b, outputs[depth], depth)
            seconds = time.process_time() - t0
            bill, requests = ctx._flops_used - f0, len(ctx._op_log) - q0
            actual = np.asarray(out, dtype=np.float64)
            rel = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
            assert np.isfinite(actual).all() and rel <= protocol['fidelity_relative_l2_max']
            row = dict(shape=shape, depth=depth, phase=phase, repetition=repetition,
                       bill=int(bill), requests=requests, cpu_s=seconds, relative_l2=rel)
            rows.append(row)
            print(json.dumps(row), flush=True)
        for depth in (3, 4):
            measure(depth, 'binding', 0)
        for repetition in range(protocol['repetitions']):
            for depth in ((3, 4) if repetition % 2 == 0 else (4, 3)):
                measure(depth, 'warm', repetition)
        assert np.array_equal(np.asarray(a), an) and np.array_equal(np.asarray(b), bn)
        mod._TR_LAST[0] = None
        del a, b, outputs, an, bn, reference
with (HERE / 'bench_result.json').open('x') as f:
    json.dump(dict(source_sha256=protocol['parent_sha256'], rows=rows,
                   fidelity_pass=True, inputs_unchanged=True), f, indent=2)
print('BENCH_COMPLETE', flush=True)
