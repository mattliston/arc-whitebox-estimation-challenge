#!/usr/bin/env python3
"""TAG fable-engineer (E-72d, lead GO 2026-09-25: pin E-72c's CPU band): in process, 1 BLAS thread, <cand> after Estimator.setup.
Per product class (m, k, n) of S23's depth-5 products with k >= 4096 (net-100 census; count = products per net), biggest mkn first:
  d5    : the tree kernel at depth 5 on the exact dims (_ipt_product_ab, as production runs these products today)
  split : m and n zero-padded to multiples of 64 the way S24 pads (fnp.zeros + fnp.copyto, billed), k cut into balanced chunks
          (multiples of 64, at most CAP and small enough for S24's depth-6 regions), each chunk through the tree kernel at depth 6 into
          its own padded output (fnp.empty), then the chunk outputs' leading m x n windows added into the result (nch - 1 adds).
Per path: the first call binds its tree keys (bind_ops, bind_wall_s), then REPS alternating steady-state calls: bill, ops (= requests
at the boundary), backend time (sum of per-op flopscope backend time), matmul backend, wall; accuracy vs a float64 product. The
transform-reuse key is cleared before every call, so no call reuses a tree. Summary: sums weighted by the census count (per net).
usage: <python> e72d_ksplit_bench.py <cand.py> <tag> [CAP=4096] [REPS=3] [max classes=99]
Platform-like CPU needs the OpenBLAS env python (astra_phase2_reexamination_v1/openblas_runtime_env_v1); the .venv numpy is Accelerate."""
import sys, json, types, time, statistics as st
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
H = Path(__file__).resolve().parent
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, tag = Path(sys.argv[1]).resolve(), sys.argv[2]
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MAXC = int(sys.argv[5]) if len(sys.argv) > 5 else 99
mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
est = mod.Estimator()
t0 = time.time()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
cfg = np.__config__.CONFIG if hasattr(np.__config__, 'CONFIG') else {}
blas = cfg.get('Build Dependencies', {}).get('blas', {}).get('name', '?')
print(json.dumps(dict(tag=tag, cand=cand.name, python=sys.executable, numpy=np.__version__, blas=blas, setup_s=round(time.time() - t0, 2),
                      cap=CAP, reps=REPS, region6=list(mod._IPT_REGION[6]))), flush=True)
REG6 = mod._IPT_REGION[6]
up = lambda x, q: -(-x // q) * q
C = json.load(open(H / 'request_census' / 'product_classes_S23_r100.json'))
classes = sorted([(m, k, n, c) for (m, k, n, L), c in C if L == 5 and k >= 4096], key=lambda r: -r[0] * r[1] * r[2])[:MAXC]


def chunk_sizes(M, K, N):
    cmax = min(CAP, 64 * min(REG6[0] // (7 ** 6 * (M // 64)), REG6[1] // (7 ** 6 * (N // 64))))
    nb = K // 64
    nch = -(-nb // (cmax // 64))
    base, r = divmod(nb, nch)
    sizes = [(base + 1) * 64] * r + [base * 64] * (nch - r)
    assert sum(sizes) == K and all(mod._ipt_fits(6, M, q, N) for q in sizes), (M, K, N, sizes)
    return sizes


def split_call(a, b, out, m, k, n, M, N, sizes):
    hold = []
    if M != m:
        aa = fnp.zeros((M, k), dtype=fnp.float32); hold.append(fnp.copyto(aa[:m, :], a))
    else:
        aa = a
    if N != n:
        bb = fnp.zeros((k, N), dtype=fnp.float32); hold.append(fnp.copyto(bb[:, :n], b))
    else:
        bb = b
    outs, k0 = [], 0
    for q in sizes:
        o = fnp.empty((M, N), dtype=fnp.float32)
        mod._ipt_product_ab(aa[:, k0:k0 + q], bb[k0:k0 + q, :], o, 6)
        outs.append(o); k0 += q
    if len(outs) == 1:
        hold.append(fnp.copyto(out, outs[0][:m, :n]))
    else:
        fnp.add(outs[0][:m, :n], outs[1][:m, :n], out=out)
        for o in outs[2:]:
            fnp.add(out, o[:m, :n], out=out)
    del hold
    return out


def measure(fn):
    if hasattr(mod, '_TR_LAST'):
        mod._TR_LAST[0] = None
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True) as ctx:
        t0 = time.time(); fn(); wall = time.time() - t0
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values()), wall_s=wall,
                matmul_s=float(ops.get('matmul', {}).get('flopscope_backend_time_s', 0.0)))


rng = np.random.default_rng(0)
tot = {'d5': dict(flops=0, ops=0, backend_s=0.0, bind_ops=0, bind_wall_s=0.0), 'split': dict(flops=0, ops=0, backend_s=0.0, bind_ops=0, bind_wall_s=0.0)}
for (m, k, n, cnt) in classes:
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    M, N = up(m, 64), up(n, 64); sizes = chunk_sizes(M, k, N)
    with flops.BudgetContext(flop_budget=2 ** 42, quiet=True):
        a, b = fnp.asarray(A), fnp.asarray(B)
        o5, os_ = fnp.empty((m, n), dtype=fnp.float32), fnp.empty((m, n), dtype=fnp.float32)
    paths = {'d5': lambda: mod._ipt_product_ab(a, b, o5, 5), 'split': lambda: split_call(a, b, os_, m, k, n, M, N, sizes)}
    rec = dict(m=m, k=k, n=n, count=cnt, M=M, N=N, chunks=sizes)
    first = {p: measure(f) for p, f in paths.items()}
    runs = {p: [] for p in paths}
    for _ in range(REPS):
        for p, f in paths.items():
            runs[p].append(measure(f))
    for p, out in (('d5', o5), ('split', os_)):
        rs = runs[p]
        assert len({r['flops'] for r in rs}) == 1 and len({r['ops'] for r in rs}) == 1, (p, rs)
        err = float(np.linalg.norm(np.asarray(out, dtype=np.float64) - ref) / np.linalg.norm(ref))
        rec[p] = dict(flops=rs[0]['flops'], ops=rs[0]['ops'], backend_s=round(st.median(r['backend_s'] for r in rs), 4),
                      backend_all=[round(r['backend_s'], 4) for r in rs], matmul_s=round(st.median(r['matmul_s'] for r in rs), 4),
                      wall_s=round(st.median(r['wall_s'] for r in rs), 4), rel_err=float('%.3g' % err),
                      bind_ops=first[p]['ops'] - rs[0]['ops'], bind_wall_s=round(first[p]['wall_s'] - st.median(r['wall_s'] for r in rs), 4))
        for key in ('flops', 'ops', 'backend_s', 'bind_ops', 'bind_wall_s'):
            tot[p][key] += cnt * rec[p][key]
    d5, sp = rec['d5'], rec['split']
    rec['delta'] = dict(bill_rel=round(sp['flops'] / d5['flops'] - 1, 4), bill_abs_M=round((sp['flops'] - d5['flops']) / 1e6, 1), ops=sp['ops'] - d5['ops'],
                        backend_abs_s=round(sp['backend_s'] - d5['backend_s'], 4), backend_ratio=round(sp['backend_s'] / d5['backend_s'], 2),
                        matmul_ratio=round(sp['matmul_s'] / max(d5['matmul_s'], 1e-9), 2))
    print(json.dumps(rec), flush=True)
t5, ts = tot['d5'], tot['split']
print(json.dumps(dict(summary=tag, classes=len(classes), products_per_net=sum(c for *_, c in classes),
                      bill_d5_G=round(t5['flops'] / 1e9, 3), bill_split_G=round(ts['flops'] / 1e9, 3), bill_delta_G=round((ts['flops'] - t5['flops']) / 1e9, 3),
                      ops_d5=t5['ops'], ops_split=ts['ops'], ops_delta=ts['ops'] - t5['ops'],
                      backend_d5_s=round(t5['backend_s'], 3), backend_split_s=round(ts['backend_s'], 3), backend_delta_s=round(ts['backend_s'] - t5['backend_s'], 3),
                      bind_ops_d5=t5['bind_ops'], bind_ops_split=ts['bind_ops'], bind_wall_d5_s=round(t5['bind_wall_s'], 3), bind_wall_split_s=round(ts['bind_wall_s'], 3))), flush=True)
print('E72D_DONE', flush=True)
