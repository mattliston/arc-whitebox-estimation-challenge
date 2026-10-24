#!/usr/bin/env python3
"""TAG fable-engineer2 (lead 2026-09-25: selective depth-7 frontier + depth-5 padding lever): S27's alt-basis tree kernel in process
(setup() first, then optionally full nets for the backend base), per product class, native depth vs the alternative, same random
float32 operands:
  d7 : native depth 6 (_ipt_product_ab) vs depth 7. ONE fixed depth-7 region for every class, as a build would have: A 52.7M, B/C
       105.4M floats (1024 x 1024 A tree, 1024 x 2048 B/C trees). A class whose depth-7 trees do not fit runs as column packets of
       2048 sharing ONE depth-7 A tree (as _cp_d6_shared does at depth 6 with 4096-wide packets).
  p6 : native depth 5 vs depth 6 on m, k, n zero-padded to multiples of 64 (fnp.zeros + fnp.copyto per padded operand, the product
       into an (M, N) buffer, its leading m x n window copied into out, all metered), as _ks_product pads. Unpadded when dims divide.
Every key is bound FUSED (the level-op tuples a prebound key carries) in an unmetered binding call; the transform-reuse key is
cleared before every call; reps alternate the native/alternative order. One JSON row per class: flops, ops, backend_s (sum of
flopscope per-op backend time = the platform-kernel proxy), wall_s, rel_err vs float64, deltas.
usage: <openblas python> bench_frontier.py <cand.py> <reps> <base nets, e.g. 35,100 or -> <kind:m,k,n;kind:m,k,n;...>"""
import sys, json, types, time, statistics as st
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
P2 = Path(__file__).resolve().parents[2]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand = Path(sys.argv[1]); mod = types.ModuleType('cand'); mod.__file__ = str(cand); sys.modules['cand'] = mod
exec(compile(cand.read_text(), str(cand), 'exec'), mod.__dict__)
REPS = int(sys.argv[2])
est = mod.Estimator(); t0 = time.time()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
arena = int(mod._SCRATCH[0].shape[0])
print(json.dumps(dict(file=cand.name, setup_s=round(time.time() - t0, 2), arena_floats=arena, region6=mod._IPT_REGION[6])), flush=True)
def summ(ctx, wall):
    s = ctx.summary_dict(); ops = s['operations']
    return dict(flops=int(s['flops_used']), ops=int(sum(int(v.get('calls', v.get('count', 0))) for v in ops.values())),
                backend_s=float(sum(float(v.get('flopscope_backend_time_s', 0.0)) for v in ops.values())), wall_s=wall,
                matmul_backend_s=float(ops.get('matmul', {}).get('flopscope_backend_time_s', 0.0)))
if sys.argv[3] != '-':    # the net-level base in the same process and env: setup, then nets in order (first = cold)
    STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
    class MLP:
        def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
    for r in [int(x) for x in sys.argv[3].split(',')]:
        W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            t1 = time.time(); est.predict(MLP([W[l] for l in range(16)], f"d7b-{r}"), 2**41); w = time.time() - t1
        x = summ(ctx, w); print(json.dumps(dict(base_net=r, flops=x['flops'], ops=x['ops'], backend_s=round(x['backend_s'], 3),
                                                  matmul_backend_s=round(x['matmul_backend_s'], 3), wall_s=round(w, 2))), flush=True)
U7 = 7 ** 7
mod._IPT_REGION[7] = (U7 * 8 * 8, U7 * 8 * 16, U7 * 8 * 16)   # fixed depth-7 region: 52.7M / 105.4M / 105.4M floats
assert sum(mod._IPT_REGION[7]) <= arena
PK = 2048
up = lambda x, q: -(-x // q) * q
def fits(L, m, k, n):
    reg = mod._IPT_REGION[L]; u = 1 << L
    if m % u or k % u or n % u: return False
    return 7 ** L * (m // u) * (k // u) <= reg[0] and 7 ** L * (k // u) * (n // u) <= reg[1] and 7 ** L * (m // u) * (n // u) <= reg[2]
def native(L):
    def f(a, b, out, dims):
        mod._TR_LAST[0] = None
        mod._ipt_product_ab(a, b, out, L)
    return f
def d7_packets(a, b, out, dims):
    m, k, n = dims
    mod._TR_LAST[0] = None
    TA = mod._ipt_forward_a_ab(a, 7, m, k)
    for j0 in range(0, n, PK):
        j1 = min(n, j0 + PK)
        bj, oj = b[:, j0:j1], out[:, j0:j1]
        mod._BH_HOLD.append((bj, oj))
        TB = mod._ipt_forward_b_ab(bj, 7, k, j1 - j0)
        mod._ipt_leaf_unwind_ab(TA, TB, oj, 7, m, j1 - j0)
    mod._BH_HOLD.clear()
def pad6(a, b, out, dims):
    m, k, n = dims; M, K, N = up(m, 64), up(k, 64), up(n, 64)
    mod._TR_LAST[0] = None
    hold = []
    if (M, K) != (m, k):
        aa = fnp.zeros((M, K), dtype=fnp.float32); hold.append(fnp.copyto(aa[:m, :k], a))
    else:
        aa = a
    if (K, N) != (k, n):
        bb = fnp.zeros((K, N), dtype=fnp.float32); hold.append(fnp.copyto(bb[:k, :n], b))
    else:
        bb = b
    if (M, N) != (m, n):
        o = fnp.empty((M, N), dtype=fnp.float32)
        mod._ipt_product_ab(aa, bb, o, 6)
        hold.append(fnp.copyto(out, o[:m, :n]))
    else:
        mod._ipt_product_ab(aa, bb, out, 6)
    del hold
rng = np.random.default_rng(7)
for spec in sys.argv[4].split(';'):
    kind, dims = spec.split(':'); m, k, n = (int(x) for x in dims.split(','))
    rec = dict(kind=kind, dims=(m, k, n))
    if kind == 'd7':
        if not fits(6, m, k, n): rec['skip'] = 'native depth 6 does not fit'; print(json.dumps(rec), flush=True); continue
        nat, Ln = native(6), 6
        if fits(7, m, k, n): alt, rec['alt'] = native(7), 'depth7'
        elif n > PK and all(fits(7, m, k, min(n, j + PK) - j) for j in range(0, n, PK)): alt, rec['alt'] = d7_packets, f'depth7 packets of {PK}'
        else: rec['skip'] = 'depth 7 does not fit the fixed region'; print(json.dumps(rec), flush=True); continue
    else:
        M, K, N = up(m, 64), up(k, 64), up(n, 64); rec['padded'] = (M, K, N)
        if not fits(5, m, k, n): rec['skip'] = 'native depth 5 does not fit'; print(json.dumps(rec), flush=True); continue
        if not fits(6, M, K, N): rec['skip'] = 'padded depth 6 does not fit the depth-6 region'; print(json.dumps(rec), flush=True); continue
        nat, alt, rec['alt'] = native(5), pad6, 'padded depth 6'
    A = rng.standard_normal((m, k)).astype(np.float32); B = rng.standard_normal((k, n)).astype(np.float32)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        a, b = fnp.asarray(A), fnp.asarray(B)
        outs = {'nat': fnp.empty((m, n), dtype=fnp.float32), 'alt': fnp.empty((m, n), dtype=fnp.float32)}
        mod._LVF_EAGER[0] = True                         # bind every key of both variants FUSED
        nat(a, b, outs['nat'], (m, k, n)); alt(a, b, outs['alt'], (m, k, n))
        mod._LVF_EAGER[0] = False
    runs = {'nat': [], 'alt': []}
    for r in range(REPS):
        for v in (('nat', 'alt') if r % 2 == 0 else ('alt', 'nat')):
            f = nat if v == 'nat' else alt
            with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                t1 = time.time(); f(a, b, outs[v], (m, k, n)); w = time.time() - t1
            runs[v].append(summ(ctx, w))
    for v in ('nat', 'alt'):
        x = runs[v]
        rec[v] = dict(flops=x[0]['flops'], ops=x[0]['ops'], backend_s=round(st.median(y['backend_s'] for y in x), 4),
                      matmul_backend_s=round(st.median(y['matmul_backend_s'] for y in x), 4), wall_s=round(st.median(y['wall_s'] for y in x), 4),
                      backend_reps=[round(y['backend_s'], 4) for y in x],
                      rel_err=float('%.3g' % (np.linalg.norm(np.asarray(outs[v], dtype=np.float64) - ref) / np.linalg.norm(ref))))
    rec['delta'] = dict(bill=rec['alt']['flops'] - rec['nat']['flops'], bill_rel=round(rec['alt']['flops'] / rec['nat']['flops'] - 1, 4),
                        ops=rec['alt']['ops'] - rec['nat']['ops'], backend_s=round(rec['alt']['backend_s'] - rec['nat']['backend_s'], 4),
                        backend_rel=round(rec['alt']['backend_s'] / rec['nat']['backend_s'] - 1, 3), err_ratio=round(rec['alt']['rel_err'] / rec['nat']['rel_err'], 2))
    print(json.dumps(rec), flush=True)
print("BENCH_DONE", flush=True)
