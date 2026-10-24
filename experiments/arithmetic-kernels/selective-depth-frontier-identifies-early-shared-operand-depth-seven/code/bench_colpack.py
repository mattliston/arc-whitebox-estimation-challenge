#!/usr/bin/env python3
"""TAG fable-engineer2 (lead 2026-09-25, selective depth-7 frontier): the PRODUCTION dispatch vs depth-7 column packets, in process on a
candidate (setup() first; OpenBLAS env, 1 thread). native = the module's live _mm(a, b) (the colpack push's own call, line 5055 of S27:
_mm(wp, p[:, :old_columns]); rule depth, _cp_d6_shared's shared depth-6 A tree for n > 4096, squares for 1024^3). alt = ONE depth-7 A tree
and column packets of <= 2048 (fixed depth-7 region 52.7M / 105.4M / 105.4M floats); a last packet whose width is not a multiple of 128 is
zero-padded to one (fnp.zeros + fnp.copyto in, its leading columns copied out; metered). Keys bound FUSED before the timed reps; the
transform-reuse key is cleared before every call; reps alternate. usage: bench_colpack.py <cand.py> <reps> <m,k,n;m,k,n;...>"""
import sys, json, types, time, statistics as st
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand = Path(sys.argv[1]); mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
REPS = int(sys.argv[2])
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
U7 = 7 ** 7; mod._IPT_REGION[7] = (U7 * 64, U7 * 128, U7 * 128); PK = 2048
assert sum(mod._IPT_REGION[7]) <= int(mod._SCRATCH[0].shape[0])
def summ(ctx, wall):
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=float(sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values())), wall_s=wall)
def native(a, b, m, k, n):
    mod._TR_LAST[0] = None
    return mod._mm(a, b)
def d7(a, b, m, k, n):
    mod._TR_LAST[0] = None
    out = fnp.empty((m, n), dtype=fnp.float32)
    TA = mod._ipt_forward_a_ab(a, 7, m, k)
    hold = []
    for j0 in range(0, n, PK):
        j1 = min(n, j0 + PK); w = j1 - j0
        if w % 128:
            wp = -(-w // 128) * 128
            bj = fnp.zeros((k, wp), dtype=fnp.float32); hold.append(fnp.copyto(bj[:, :w], b[:, j0:j1]))
            oj = fnp.empty((m, wp), dtype=fnp.float32)
            TB = mod._ipt_forward_b_ab(bj, 7, k, wp); mod._ipt_leaf_unwind_ab(TA, TB, oj, 7, m, wp)
            hold.append(fnp.copyto(out[:, j0:j1], oj[:, :w]))
        else:
            bj, oj = b[:, j0:j1], out[:, j0:j1]
            mod._BH_HOLD.append((bj, oj))
            TB = mod._ipt_forward_b_ab(bj, 7, k, w); mod._ipt_leaf_unwind_ab(TA, TB, oj, 7, m, w)
    mod._BH_HOLD.clear(); del hold
    return out
rng = np.random.default_rng(11)
for spec in sys.argv[3].split(';'):
    m, k, n = (int(x) for x in spec.split(','))
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        a, b = fnp.asarray(A), fnp.asarray(B)
        mod._LVF_EAGER[0] = True; native(a, b, m, k, n); d7(a, b, m, k, n); mod._LVF_EAGER[0] = False
    runs = {'nat': [], 'alt': []}; last = {}
    for r in range(REPS):
        for v in (('nat', 'alt') if r % 2 == 0 else ('alt', 'nat')):
            f = native if v == 'nat' else d7
            with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                t1 = time.time(); o = f(a, b, m, k, n); w = time.time() - t1
            runs[v].append(summ(ctx, w)); last[v] = np.asarray(o, dtype=np.float64)
    rec = dict(dims=(m, k, n))
    for v in ('nat', 'alt'):
        x = runs[v]
        rec[v] = dict(flops=x[0]['flops'], ops=x[0]['ops'], backend_s=round(st.median(y['backend_s'] for y in x), 4),
                      backend_reps=[round(y['backend_s'], 4) for y in x], wall_s=round(st.median(y['wall_s'] for y in x), 4),
                      rel_err=float('%.3g' % (np.linalg.norm(last[v] - ref) / np.linalg.norm(ref))))
    rec['delta'] = dict(bill=rec['alt']['flops'] - rec['nat']['flops'], bill_rel=round(rec['alt']['flops'] / rec['nat']['flops'] - 1, 4),
                        ops=rec['alt']['ops'] - rec['nat']['ops'], backend_s=round(rec['alt']['backend_s'] - rec['nat']['backend_s'], 4),
                        backend_rel=round(rec['alt']['backend_s'] / rec['nat']['backend_s'] - 1, 3), err_ratio=round(rec['alt']['rel_err'] / rec['nat']['rel_err'], 2))
    print(json.dumps(rec), flush=True)
print("BENCH_DONE", flush=True)
