"""TAG opus_engineer a57 unit check (2026-10-11; v2: + the 7-level strip reads when the fragment has them): the 7-level strip push (fragment_hz7_v1.py appended to a parent file) against float64
numpy on random operands, for the push shapes of the census, and the rule's decision per shape. In process (local flopscope), after the
estimator's setup (the arena).
usage: unit_hz7.py <parent candidate.py> <fragment.py>"""
from pathlib import Path
import sys
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy
flops.configure(symmetry_warnings=False)
parent, frag = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_unit_hz7'); mod.__file__ = str(parent); sys.modules[mod.__name__] = mod
exec(compile(parent.read_text() + '\n\n' + frag.read_text(), str(parent), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
rng = np.random.default_rng(0)
shapes = [(1024, 960, 6464), (896, 832, 5248), (832, 832, 4160), (960, 960, 3008), (960, 928, 4096), (896, 800, 5248),
          (704, 256, 2112), (1024, 960, 2112), (768, 384, 6144), (1024, 896, 4096)]
reads = [(896, 9984, 960), (640, 6464, 832), (640, 7552, 832), (672, 6208, 832), (800, 7360, 928), (704, 10432, 832)] if hasattr(mod, '_hz7r_product') else []
for m, k, n in reads:
    a = rng.standard_normal((m, k)).astype(np.float32); b = rng.standard_normal((n, k)).astype(np.float32).T   # reads pass a transposed right operand
    want = a.astype(np.float64) @ b.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        A, Bt = fnp.asarray(a), fnp.asarray(np.ascontiguousarray(b.T))
        B = Bt.T
        take = mod._hz7r_plan(m, k, n)
        b7, b6 = mod._hz7r_bill(7, m, k, n), mod._hz7r_bill(6, m, k, n)
        f0, o0 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
        got = np.asarray(mod._hz7r_product(A, B, m, k, n), dtype=np.float64)
        f1, o1 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
        ref = np.asarray(mod._HZ7R_OLD_MM(A, B), dtype=np.float64)
        f2, o2 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want)); err0 = float(np.linalg.norm(ref - want) / np.linalg.norm(want))
    print(f'READ m {m:4d} k {k:5d} n {n:4d}: rule {"TAKE" if take else "skip"} (model 7 / 6 {b7 / b6:.4f}, model 7 / measured parent {b7 / max(f2 - f1, 1):.4f}); 7-level strips: rel err {err:.2e}, '
          f'{(f1 - f0) / 1e6:.1f} Mflop, {o1 - o0} ops | parent: rel err {err0:.2e}, {(f2 - f1) / 1e6:.1f} Mflop, {o2 - o1} ops | bill {100 * ((f1 - f0) / max(f2 - f1, 1) - 1):+.2f}%', flush=True)
for m, k, n in shapes:
    a = rng.standard_normal((m, k)).astype(np.float32); b = rng.standard_normal((k, n)).astype(np.float32)
    want = a.astype(np.float64) @ b.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        A, B = fnp.asarray(a), fnp.asarray(b)
        take = mod._hz7_plan(m, k, n)
        b7, b6 = mod._hz7_bill(7, m, k, n, 2048), mod._hz7_bill(6, m, k, n, 4096)
        f0, o0 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
        got = np.asarray(mod._hz7_product(A, B, m, k, n), dtype=np.float64) if mod._hz7_dim(m) and mod._hz7_dim(k) else None
        f1, o1 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
        ref = np.asarray(mod._HZ7_OLD_MM(A, B), dtype=np.float64)
        f2, o2 = ctx.summary_dict()['flops_used'], len(ctx._op_log)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want)) if got is not None else float('nan')
    err0 = float(np.linalg.norm(ref - want) / np.linalg.norm(want))
    print(f'm {m:4d} k {k:4d} n {n:5d}: rule {"TAKE" if take else "skip"} (model 7 / 6 levels {b7 / b6:.4f}); 7-level strips: rel err {err:.2e}, '
          f'{(f1 - f0) / 1e6:.1f} Mflop, {o1 - o0} ops | parent path: rel err {err0:.2e}, {(f2 - f1) / 1e6:.1f} Mflop, {o2 - o1} ops | '
          f'bill {100 * ((f1 - f0) / max(f2 - f1, 1) - 1):+.2f}%', flush=True)
