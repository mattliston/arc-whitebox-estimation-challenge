#!/usr/bin/env python3
"""TAG fable-engineer (E-74 bench): the candidate's OWN k-split product (_ks_product, v1 or v2 fragment) vs the depth-5 tree product on
each shape the fragment splits (_KS_SHAPES), in process after Estimator.setup (so phase 3 has bound the chunk keys as in production).
The depth-5 keys are (re)bound FUSED, as S24's setup leaves them (the fragment drops them from its own prebind). Tree-cache caps lifted
(no evictions inside the bench); the transform-reuse key is cleared before every call. Per shape: bill, ops (= requests), backend time
(median of REPS alternating calls), accuracy vs float64. usage: <python> e74_ksprod_bench.py <cand.py> <tag> [REPS=3]"""
import sys, json, types, time, statistics as st
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, tag = Path(sys.argv[1]).resolve(), sys.argv[2]; REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
cfg = np.__config__.CONFIG if hasattr(np.__config__, 'CONFIG') else {}
print(json.dumps(dict(tag=tag, cand=cand.name, blas=cfg.get('Build Dependencies', {}).get('blas', {}).get('name', '?'), pb_log=[list(x) for x in mod._PB_LOG])), flush=True)
CACHES = (mod._IPT_A, mod._IPT_B, mod._IPT_C)
for c in CACHES:
    mod._CAP_BY_ID[id(c)] = 100000


def measure(fn, eager=False):
    if hasattr(mod, '_TR_LAST'):
        mod._TR_LAST[0] = None
    mod._LVF_EAGER[0] = eager
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True) as ctx:
        fn()
    mod._LVF_EAGER[0] = False
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values()))


rng = np.random.default_rng(0); tot = dict(d5_flops=0, ks_flops=0, d5_ops=0, ks_ops=0, d5_s=0.0, ks_s=0.0)
for (m, k, n) in mod._KS_SHAPES:
    plan = mod._ks_plan(m, k, n)
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32); ref = A.astype(np.float64) @ B.astype(np.float64)
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        a, b = fnp.asarray(A), fnp.asarray(B); o5, oks = fnp.empty((m, n), dtype=fnp.float32), fnp.empty((m, n), dtype=fnp.float32)
    for kind, key in ((0, (5, m, k)), (1, (5, k, n)), (2, (5, m, n))):
        e = dict.get(CACHES[kind], key)
        if e is not None and e[3] is None:
            dict.pop(CACHES[kind], key)
    f5 = lambda: mod._ipt_product_ab(a, b, o5, 5)
    fk = lambda: mod._ks_product(a, b, oks, m, k, n, plan)
    first5, firstk = measure(f5, eager=True), measure(fk)
    r5, rk = [], []
    for _ in range(REPS):
        r5.append(measure(f5)); rk.append(measure(fk))
    assert len({x['flops'] for x in r5}) == 1 and len({x['flops'] for x in rk}) == 1
    e5 = float(np.linalg.norm(np.asarray(o5, dtype=np.float64) - ref) / np.linalg.norm(ref)); ek = float(np.linalg.norm(np.asarray(oks, dtype=np.float64) - ref) / np.linalg.norm(ref))
    b5, bk = st.median(x['backend_s'] for x in r5), st.median(x['backend_s'] for x in rk)
    rec = dict(shape=(m, k, n), plan=plan, d5=dict(flops=r5[0]['flops'], ops=r5[0]['ops'], backend_s=round(b5, 4), rel_err=float('%.3g' % e5)),
               ks=dict(flops=rk[0]['flops'], ops=rk[0]['ops'], backend_s=round(bk, 4), rel_err=float('%.3g' % ek), first_call_extra_ops=firstk['ops'] - rk[0]['ops'], first_call_extra_flops=firstk['flops'] - rk[0]['flops']),
               delta=dict(bill_rel=round(rk[0]['flops'] / r5[0]['flops'] - 1, 4), bill_M=round((rk[0]['flops'] - r5[0]['flops']) / 1e6, 1), ops=rk[0]['ops'] - r5[0]['ops'], backend_ratio=round(bk / b5, 2)))
    for key5, keyk, f in (('d5_flops', 'ks_flops', 'flops'), ('d5_ops', 'ks_ops', 'ops')):
        tot[key5] += rec['d5'][f]; tot[keyk] += rec['ks'][f]
    tot['d5_s'] += b5; tot['ks_s'] += bk
    print(json.dumps(rec), flush=True)
print(json.dumps(dict(summary=tag, shapes=len(mod._KS_SHAPES), bill_delta_G=round((tot['ks_flops'] - tot['d5_flops']) / 1e9, 3), ops_delta=tot['ks_ops'] - tot['d5_ops'],
                      backend_d5_s=round(tot['d5_s'], 3), backend_ks_s=round(tot['ks_s'], 3), backend_ratio=round(tot['ks_s'] / tot['d5_s'], 3))), flush=True)
print('E74_KSPROD_DONE', flush=True)
