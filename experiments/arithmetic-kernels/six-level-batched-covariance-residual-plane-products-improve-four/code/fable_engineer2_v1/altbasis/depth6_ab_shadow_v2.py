#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-D6 v2; v1 = depth6_ab_shadow.py was superseded before it ran: it shared the native caches): DEPTH UNDER THE
ALT BASIS, in situ on S15 (S14 + --altbasis). Every depth-5 alt-basis tree product is ALSO run at depth 6 on the same operands (shadow);
only the depth-5 output goes on, so the network is exactly S15 (checked: final predictions bit-identical to parity's par4_ab_S14.npz and
the native bill equal to parity's). Shadowed classes:
  single  : _ipt_product_ab at L=5 (from _packed_product_cached / _packed_read5), shadowed at L=6 in the standard _IPT_REGION[6] when it
            fits, else in a region sized to the product (arena 391.6M floats);
  batched : _bb_product at L=5 (P = 2 or 3 independent 1024^3 products from _mm_batch), shadowed at L=6 in a region sized to it
            (P=2: 60.2M floats per side, the engineer's bench);
  w5port  : _w5_port_product_body at levels=5 (cached A tree), shadowed at L=6 with its own cached depth-6 A tree (formed on the calls
            where the native one is formed).
Shadow calls use their OWN tree/view caches (native caches untouched); L5/L6 order alternates per call. Per call: bill, user CPU
(getrusage), op calls, max |L6 - L5| / max |L5|, and whether it was the class's first call in the process (binding).
usage: depth6_ab_shadow_v2.py <S15.py> <ranks> <tag>   (writes depth6_ab_<tag>.json next to this file)"""
import sys, json, types, time, resource
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("d6_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
assert getattr(mod, "_AB_ON", False), "candidate lacks --altbasis"
single, batched, w5 = mod._ipt_product_ab, mod._bb_product, mod._w5_port_product_body
CACHES = ("_IPT_A", "_IPT_B", "_IPT_C", "_IPT_BA", "_IPT_BB", "_IPT_BC", "_AB_VIEWS")
SH = {}; W5LEAF = {}; seen = set(); log = []; ctx_ref = [None]; turn = [0]; OVER = dict(bill=0, ops=0, user=0.0)
def region6(P, m, k, n):
    b = 7 ** 6 * P; u = 64
    return (b * (m // u) * (k // u), b * (k // u) * (n // u), b * (m // u) * (n // u))
def shadow(region, fn):
    # run fn with _IPT_REGION[6] = region and a private cache set for that region; native state restored after
    saved = {c: getattr(mod, c) for c in CACHES}; old = mod._IPT_REGION[6]
    mine = SH.setdefault(region, {c: mod._CapLRU() for c in CACHES})
    for c in CACHES: setattr(mod, c, mine[c])
    mod._IPT_REGION[6] = region
    try: return fn()
    finally:
        mod._IPT_REGION[6] = old
        for c, v in saved.items(): setattr(mod, c, v)
def meter(fn):
    c = ctx_ref[0]; f0, q0 = c._flops_used, len(c._op_log); r0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    o = fn(); r1 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    return o, dict(bill=int(c._flops_used - f0), ops=len(c._op_log) - q0, user=round(r1 - r0, 5))
def rel(xs, ys):
    num = max(float(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).max()) for x, y in zip(xs, ys))
    return num / max(max(float(np.abs(np.asarray(x, np.float64)).max()) for x in xs), 1e-30)
def site():
    f2 = sys._getframe(2); f3 = f2.f_back; return f"{f2.f_code.co_name}<{f3.f_code.co_name}:{f3.f_lineno}"
def pair(kind, key, native, shade, region, outs5, outs6, where):
    turn[0] += 1; first = key not in seen; seen.add(key)
    if turn[0] % 2:
        r5, m5 = meter(native); _, m6 = meter(lambda: shadow(region, shade))
    else:
        _, m6 = meter(lambda: shadow(region, shade)); r5, m5 = meter(native)
    mr, mo = meter(lambda: rel(outs5, outs6)); overhead(mo)
    log.append(dict(kind=kind, key=list(key), site=where, first=first, L5=m5, L6=m6, maxrel=mr, region=list(region)))
    return r5
def fits_arena(region): return sum(region) <= int(mod._SCRATCH[0].shape[0])
def overhead(m):
    for x in OVER: OVER[x] += m[x]
def alloc(shapes):
    outs, m = meter(lambda: [fnp.empty(s, dtype=fnp.float32) for s in shapes]); overhead(m); return outs
def w_single(a, b, out, L):
    if ctx_ref[0] is None or L != 5:
        return single(a, b, out, L)
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    region = tuple(mod._IPT_REGION[6]) if mod._ipt_fits(6, m, k, n) else region6(1, m, k, n)
    if m % 64 or k % 64 or n % 64 or not fits_arena(region):
        log.append(dict(kind="single", key=["single", m, k, n], site=site(), skip=True)); return single(a, b, out, L)
    (o6,) = alloc([(m, n)])
    return pair("single", ("single", m, k, n), lambda: single(a, b, out, 5), lambda: single(a, b, o6, 6), region, [out], [o6], site())
def w_batched(As, Bs, outs, L):
    if ctx_ref[0] is None or L != 5:
        return batched(As, Bs, outs, L)
    P = len(As); m, k, n = int(As[0].shape[0]), int(As[0].shape[1]), int(Bs[0].shape[1]); region = region6(P, m, k, n)
    if m % 64 or k % 64 or n % 64 or not fits_arena(region):
        log.append(dict(kind=f"batched{P}", key=[f"batched{P}", m, k, n], site=site(), skip=True)); return batched(As, Bs, outs, L)
    outs6 = alloc([(m, n)] * P)
    return pair(f"batched{P}", (f"batched{P}", m, k, n), lambda: batched(As, Bs, outs, 5), lambda: batched(As, Bs, outs6, 6), region, outs, outs6, site())
def w_w5(a, b, out, views, levels, cache):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    if (ctx_ref[0] is None or levels != 5 or not mod._ipt_ok(a, b, out, 5)
            or tuple(cache._shape) != (7,) * 5 + (m >> 5, k >> 5)):
        return w5(a, b, out, views, levels, cache)
    region = tuple(mod._IPT_REGION[6]) if mod._ipt_fits(6, m, k, n) else region6(1, m, k, n)
    if m % 64 or k % 64 or n % 64 or not fits_arena(region):
        log.append(dict(kind="w5port", key=["w5port", m, k, n], site=site(), skip=True)); return w5(a, b, out, views, levels, cache)
    creator = cache._leaf is None; (o6,) = alloc([(m, n)])
    def shade():
        if creator or id(cache) not in W5LEAF:
            TA = mod._ipt_forward_a_ab(a, 6, m, k); leaf = fnp.empty(tuple(TA.shape), dtype=fnp.float32); fnp.copyto(leaf, TA)
            W5LEAF.pop(id(cache), None); W5LEAF[id(cache)] = leaf
            while len(W5LEAF) > 3: W5LEAF.pop(next(iter(W5LEAF)))
        TB = mod._ipt_forward_b_ab(b, 6, k, n); mod._ipt_leaf_unwind_ab(W5LEAF[id(cache)], TB, o6, 6, m, n); mod._BH_HOLD.clear()
    return pair("w5port", ("w5port", m, k, n, "creator" if creator else "reuse"), lambda: w5(a, b, out, views, levels, cache), shade,
                region, [out], [o6], site())
mod._ipt_product_ab = w_single; mod._bb_product = w_batched; mod._w5_port_product_body = w_w5
ref = np.load(HERE.parent / "par4_ab_S14.npz")
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); res = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); log.clear(); t0 = time.time()
    for x in OVER: OVER[x] = 0 * OVER[x]
    u0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx; pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"d6-{rank}"), 2**41)); ctx_ref[0] = None
    total_user = resource.getrusage(resource.RUSAGE_SELF).ru_utime - u0
    total = int(ctx.summary_dict()["flops_used"]); sh = [r for r in log if not r.get("skip")]
    native_bill = total - sum(r["L6"]["bill"] for r in sh) - OVER["bill"]
    native_user = total_user - sum(r["L6"]["user"] for r in sh) - OVER["user"]
    native_ops = len(ctx._op_log) - sum(r["L6"]["ops"] for r in sh) - OVER["ops"]
    same = bool(np.array_equal(pred, ref[f"pred_{rank}"])) if f"pred_{rank}" in ref else None
    ref_bill = int(ref[f"flops_{rank}"]) if f"flops_{rank}" in ref else None
    res.append(dict(rank=rank, native_bill=native_bill, ref_bill=ref_bill, bit_identical=same, native_user=round(native_user, 3),
                    native_ops=native_ops, harness_overhead=dict(OVER), wall=round(time.time() - t0, 1), calls=list(log)))
    print(json.dumps({k: v for k, v in res[-1].items() if k != "calls"}), f"shadowed {len(sh)}, skipped {len(log) - len(sh)}", flush=True)
    json.dump(res, open(HERE / f"depth6_ab_{tag}.json", "w"))
print("DONE", tag, flush=True)
