#!/usr/bin/env python3
"""TAG fable-engineer (E-72, lead's ask: price one more Winograd level on the big products): S23's alt-basis tree kernel
(_ipt_product_ab) in-process (.venv flopscope, 1 BLAS thread) on random float32 operands, per product class, at depth 6 (as S23) and
depth 7 (region _IPT_REGION[7] sized to the class; S23's arena). For each: steady-state bill (second call; the first binds), op count
(= requests at the boundary), backend time (sum of per-op flopscope backend time; the platform-kernel proxy), wall, and accuracy
(relative Frobenius error vs a float64 numpy product). usage: .venv/bin/python e72_depth7_bench.py <cand.py> <m,k,n;m,k,n;...>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand = Path(sys.argv[1]); mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
arena = int(mod._SCRATCH[0].shape[0])
classes = [tuple(int(x) for x in c.split(',')) for c in sys.argv[2].split(';')]
rng = np.random.default_rng(0)
def run(a, b, out, L):
    with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
        t0 = time.time(); mod._ipt_product_ab(a, b, out, L); wall = time.time() - t0
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=round(sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values()), 4), wall_s=round(wall, 3),
                matmul_backend_s=round(float(ops.get('matmul', {}).get('flopscope_backend_time_s', 0.0)), 4))
for (m, k, n) in classes:
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32); ref = A.astype(np.float64) @ B.astype(np.float64)
    rec = dict(m=m, k=k, n=n, mkn=m * k * n)
    for L in (6, 7):
        u = 1 << L
        if m % u or k % u or n % u:
            rec[f'L{L}'] = 'dims not divisible'; continue
        if L == 7:
            need = [7 ** 7 * (m // u) * (k // u), 7 ** 7 * (k // u) * (n // u), 7 ** 7 * (m // u) * (n // u)]
            if sum(need) > arena:
                rec['L7'] = f'does not fit the arena ({sum(need)/1e6:.0f}M > {arena/1e6:.0f}M floats)'; continue
            mod._IPT_REGION[7] = tuple(need)
        with flops.BudgetContext(flop_budget=2**42, quiet=True):
            a, b, out = fnp.asarray(A), fnp.asarray(B), fnp.empty((m, n), dtype=fnp.float32)
            mod._ipt_product_ab(a, b, out, L)                      # binds this depth's tree views
        r = run(a, b, out, L)
        err = float(np.linalg.norm(np.asarray(out, dtype=np.float64) - ref) / np.linalg.norm(ref))
        r['rel_err'] = float('%.3g' % err); rec[f'L{L}'] = r
    if isinstance(rec.get('L6'), dict) and isinstance(rec.get('L7'), dict):
        a6, a7 = rec['L6'], rec['L7']
        rec['delta'] = dict(bill_rel=round(a7['flops'] / a6['flops'] - 1, 4), bill_abs=a7['flops'] - a6['flops'], ops=a7['ops'] - a6['ops'],
                            backend_rel=round(a7['backend_s'] / a6['backend_s'] - 1, 3), backend_abs_s=round(a7['backend_s'] - a6['backend_s'], 3),
                            err_ratio=round(a7['rel_err'] / a6['rel_err'], 2))
    print(json.dumps(rec), flush=True)
