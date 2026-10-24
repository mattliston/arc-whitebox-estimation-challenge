"""K_probe_strassen — attack the BATCHING MECHANISM of the Strassen recursion.

Codex's Winograd-15 takes the arithmetic (18 -> 15 arith ops per level). What is left per level is
the batching: 2 `stack`s + 1 `block`. Those are 3 ops/level x 3 levels x 31 layers = 279 charged
dispatches AND ~2.2e9 FLOPs of pure data movement (stack and concatenate both bill 2/elem).

Probes, at the real shapes of each recursion level:
  1. what `stack` / `block` actually cost, in ops and FLOPs
  2. whether an `empty` + basic-setitem assembly is cheaper (and byte-equal)
  3. whether `concatenate` of `expand_dims` views beats `stack`
  4. what a repeated `p[i]` costs vs a hoisted local (the getitem-hoist win)

Usage: ./safepy K_probe_strassen.py
"""
from __future__ import annotations
import warnings, statistics
warnings.filterwarnings("ignore")
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

B = 10 ** 17
F32 = fnp.float32
N, KD, ND = 8600, 256, 256          # deployed forward: h (8600,256) @ Wsub (256,~256)


def bo(x):
    return np.asarray(x).tobytes()


with flops.BudgetContext(flop_budget=B, quiet=True):
    rng = np.random.default_rng(0)

    def mk(shape):
        return fnp.asarray(rng.standard_normal(shape).astype(np.float32), dtype=F32)

    def cost(fn):
        p = flops.budget_summary_dict()["flops_used"]
        r = fn()
        return flops.budget_summary_dict()["flops_used"] - p, r

    print("== 1. stack / block cost at each recursion level (a-side shapes) ==")
    print(f"{'level':<8}{'piece shape':<22}{'stack flops':>14}{'block flops':>14}")
    tot_s = tot_b = 0
    for lvl, shp in ((3, (N // 2, KD // 2)), (2, (7, N // 4, KD // 4)), (1, (7, 7, N // 8, KD // 8))):
        parts = [mk(shp) for _ in range(7)]
        cs, aa = cost(lambda: fnp.stack(tuple(parts), axis=0))
        q = [mk(shp) for _ in range(4)]
        cb, blk = cost(lambda: fnp.block([[q[0], q[1]], [q[2], q[3]]]))
        tot_s += cs; tot_b += cb
        print(f"{lvl:<8}{str(shp):<22}{cs:14,d}{cb:14,d}")
        if lvl == 3:
            keep = (parts, aa, q, blk)
    print(f"{'per layer a-side':<30}{tot_s:14,d}{tot_b:14,d}   "
          f"(x31 layers = {31*(tot_s+tot_b)/1e9:.2f}e9 FLOPs of pure movement)")

    parts, aa, q, blk = keep
    shp = (N // 2, KD // 2)

    print("\n== 2. stack alternatives at the level-3 shape ==")
    c_stack, r_stack = cost(lambda: fnp.stack(tuple(parts), axis=0))

    def viacat():
        return fnp.concatenate(tuple(fnp.expand_dims(p, 0) for p in parts), axis=0)
    c_cat, r_cat = cost(viacat)

    def viaset():
        out = fnp.zeros((7,) + shp, dtype=F32)
        for i, p in enumerate(parts):
            out[i] = p
        return out
    try:
        c_set, r_set = cost(viaset)
        set_ok = bo(r_set) == bo(r_stack)
    except Exception as exc:
        c_set, set_ok = -1, f"RAISED {type(exc).__name__}: {str(exc)[:60]}"
    print(f"   stack(7 parts)                  {c_stack:12,d}  1 op")
    print(f"   concat(expand_dims) x7          {c_cat:12,d}  8 ops   byte-eq={bo(r_cat)==bo(r_stack)}")
    print(f"   zeros + 7 basic setitem         {c_set if c_set>=0 else 'n/a':>12}  8 ops   byte-eq={set_ok}")

    print("\n== 3. block alternatives ==")
    c_blk, r_blk = cost(lambda: fnp.block([[q[0], q[1]], [q[2], q[3]]]))
    c_cc, r_cc = cost(lambda: fnp.concatenate(
        (fnp.concatenate((q[0], q[1]), axis=-1), fnp.concatenate((q[2], q[3]), axis=-1)), axis=-2))
    print(f"   block([[a,b],[c,d]])            {c_blk:12,d}  1 op")
    print(f"   3x concatenate                  {c_cc:12,d}  3 ops   byte-eq={bo(r_cc)==bo(r_blk)}")

    print("\n== 4. repeated p[i] vs hoisted local (the getitem hoist) ==")
    p = fnp.stack(tuple(mk((N // 2, ND // 2)) for _ in range(7)), axis=0)
    K = 400

    def repeated():
        for _ in range(K):
            _ = p[0] + p[3] - p[4] + p[6]
            _ = p[0] - p[1] + p[2] + p[5]

    def hoisted():
        for _ in range(K):
            p0, p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            _ = p0 + p3 - p4 + p6
            _ = p0 - p1 + p2 + p5
    import time
    for lab, fn in (("repeated p[i] (12 getitem)", repeated), ("hoisted (7 getitem)", hoisted)):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        print(f"   {lab:<30} wall/iter {statistics.median(ts)/K*1e9:8.0f} ns")
