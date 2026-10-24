#!/usr/bin/env python3
"""TAG fable-engineer2 (d7cp CPU refinement option): depth-7 column packets of 2048 vs 4096 on the same product (one A7 tree each), in
process on a candidate whose arena is grown to hold 4096-wide depth-7 B/C trees (A 52.7M + B/C 210.8M = 474.4M floats; S28 has 391.6M).
Same bill by construction (checked); reports backend time and ops, 5 alternating reps. usage: bench_pk.py <cand.py> <reps> <m,k,n;...>"""
import sys, json, types, time, statistics as st
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand = Path(sys.argv[1]); mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
U7 = 7 ** 7; mod._PACKED_ELEMENTS = U7 * (64 + 256 + 256)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
assert int(mod._SCRATCH[0].shape[0]) >= U7 * 576, int(mod._SCRATCH[0].shape[0])
mod._IPT_REGION[7] = (U7 * 64, U7 * 256, U7 * 256)
def summ(ctx):
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=float(sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values())))
def packets(PK):
    def f(a, b, m, k, n):
        mod._TR_LAST[0] = None
        out = fnp.empty((m, n), dtype=fnp.float32)
        TA = mod._ipt_forward_a_ab(a, 7, m, k)
        for j0 in range(0, n, PK):
            j1 = min(n, j0 + PK); bj, oj = b[:, j0:j1], out[:, j0:j1]; mod._BH_HOLD.append((bj, oj))
            TB = mod._ipt_forward_b_ab(bj, 7, k, j1 - j0); mod._ipt_leaf_unwind_ab(TA, TB, oj, 7, m, j1 - j0)
        mod._BH_HOLD.clear()
        return out
    return f
REPS = int(sys.argv[2]); rng = np.random.default_rng(5)
V = {'pk2048': packets(2048), 'pk4096': packets(4096)}
for spec in sys.argv[3].split(';'):
    m, k, n = (int(x) for x in spec.split(','))
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32)
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        a, b = fnp.asarray(A), fnp.asarray(B)
        mod._LVF_EAGER[0] = True
        for f in V.values(): f(a, b, m, k, n)
        mod._LVF_EAGER[0] = False
    runs = {v: [] for v in V}; outs = {}
    for r in range(REPS):
        for v in (('pk2048', 'pk4096') if r % 2 == 0 else ('pk4096', 'pk2048')):
            with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                outs[v] = V[v](a, b, m, k, n)
            runs[v].append(summ(ctx))
    diff = float(np.abs(np.asarray(outs['pk2048']) - np.asarray(outs['pk4096'])).max())
    rec = dict(dims=(m, k, n), max_diff=diff)
    for v in V:
        rec[v] = dict(flops=runs[v][0]['flops'], ops=runs[v][0]['ops'], backend_s=round(st.median(x['backend_s'] for x in runs[v]), 4), reps=[round(x['backend_s'], 4) for x in runs[v]])
    rec['delta_4096_vs_2048'] = dict(bill=rec['pk4096']['flops'] - rec['pk2048']['flops'], ops=rec['pk4096']['ops'] - rec['pk2048']['ops'], backend_s=round(rec['pk4096']['backend_s'] - rec['pk2048']['backend_s'], 4))
    print(json.dumps(rec), flush=True)
print('BENCH_DONE', flush=True)
