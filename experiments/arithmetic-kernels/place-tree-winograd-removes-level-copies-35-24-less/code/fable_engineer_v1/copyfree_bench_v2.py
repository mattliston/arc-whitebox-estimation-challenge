#!/usr/bin/env python3
"""TAG fable-engineer (E-52 v2): v1 plus the layout fix. In the six-level layout, the level-6 A array shares arena offsets with the
level-4 B array, so the level-6 A transforms overwrote the plain quadrants before the leaf read them (v1: not bit-identical). v2 moves
the level-6 A array into the free region between the level-5 B capacity and the leaf tail ([236642560, 266760704), unused during the
six-level forward), binds that relocated layout once, and runs the copy-free packet on it. Everything else as v1:
prototype of COPY-FREE level L-1 on the B side of the six-level push packet, against the candidate's own
_packed_product_cached on the same inputs. The B side of level L-1 normally copies its plain quadrants (b11, b21, b22 of the level L-2
blocks) into slots 0-2 of the level L-1 array so level L runs as one batch; here that copy is skipped and level L reads those blocks in
place: the level-L B transform runs in four groups (level L-1 slots 3-6 from the array, slots 0, 1, 2 from the level L-2 quadrants), and
the three leaf products that read B's level L-1 quadrants (p0, p1, p2) split the same way. A side, p3, the batched slots 4-6 and the
unwind are unchanged. Reports: bit-identity of the output, billed flops, counted ops, user CPU (median of warm calls).
usage: copyfree_bench_v2.py <candidate.py> [reps]"""
import sys, types, resource, statistics
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
cand = Path(sys.argv[1]).resolve()
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("cf_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
M = mod

def cf_packet(a, b, out, views, L):
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    forward, leaf_a, leaf_b = M._bh_fwd(views, L, m, k, n)
    leaf, backward, (prods, l47, l37, lused) = M._bh_o_frame(views, L, m, n)
    elim_a = M._bh_disjoint(M._bh_extent(views, L, 0, m, k), lused)
    elim_b = M._bh_disjoint(M._bh_extent(views, L, 1, k, n), lused)
    assert elim_a and elim_b
    h = M._BH_HOLD.append
    plain = None
    for index, (aq, bq, aa, bb, acopies, bcopies) in enumerate(forward):
        if index == 0:
            a_top, a_bot = fnp.split(a, 2, axis=0); (a11, a12), (a21, a22) = fnp.split(a_top, 2, axis=1), fnp.split(a_bot, 2, axis=1)
            b_top, b_bot = fnp.split(b, 2, axis=0); (b11, b12), (b21, b22) = fnp.split(b_top, 2, axis=1), fnp.split(b_bot, 2, axis=1)
            h((a_top, a_bot, a11, a12, a21, a22, b_top, b_bot, b11, b12, b21, b22))
        else:
            a11, a12, a21, a22 = aq
            b11, b12, b21, b22 = bq
        last = index == L - 1
        h(fnp.add(a21, a22, out=aa[4])); h(fnp.subtract(aa[4], a11, out=aa[5])); h(fnp.subtract(a11, a21, out=aa[6])); h(fnp.subtract(a12, aa[5], out=aa[2]))
        if not last:
            h(fnp.stack((a11, a12), axis=0, out=acopies)); h(fnp.copyto(aa[3], a22))
        if index < L - 1:
            h(fnp.subtract(b12, b11, out=bb[4])); h(fnp.subtract(b22, bb[4], out=bb[5])); h(fnp.subtract(b22, b12, out=bb[6])); h(fnp.subtract(bb[5], b21, out=bb[3]))
            if index < L - 2:
                h(fnp.stack((b11, b21, b22), axis=0, out=bcopies))
            else:
                plain = (b11, b21, b22)          # level L-1 slots 0-2 stay in place (no copy)
        else:
            # level L, B side: slots 3-6 of level L-1 from the array (quadrants sliced [3:7]), slots 0-2 from the level L-2 quadrants
            groups = [(slice(3, 7), (b11[3:7], b12[3:7], b21[3:7], b22[3:7]))]
            for g, X in enumerate(plain):
                kk, nn = int(X.shape[-2]) // 2, int(X.shape[-1]) // 2
                groups.append((g, (X[..., :kk, :nn], X[..., :kk, nn:], X[..., kk:, :nn], X[..., kk:, nn:])))
            for sel, (c11, c12, c21, c22) in groups:
                h(fnp.subtract(c12, c11, out=bb[4][sel])); h(fnp.subtract(c22, bb[4][sel], out=bb[5][sel]))
                h(fnp.subtract(c22, c12, out=bb[6][sel])); h(fnp.subtract(bb[5][sel], c21, out=bb[3][sel]))
            qa = (a11, a12, a21, a22); B_groups = groups
    a11, a12, a21, a22 = qa
    for sel, (c11, c12, c21, c22) in B_groups:
        h(fnp.matmul(a11[sel], c11, out=prods[0][sel]))
        h(fnp.matmul(a12[sel], c21, out=prods[1][sel]))
        h(fnp.matmul(aa[2][sel], c22, out=prods[2][sel]))
    h(fnp.matmul(a22, bb[3], out=prods[3]))
    la47, la37, lb47, lb37 = M._bh_leafx(views, L, m, k, n, leaf_a, leaf_b)
    h(fnp.matmul(la47, lb47, out=l47))
    M._bh_backward(backward, out, m, n)
    M._BH_HOLD.clear()
    return out


import time
specs = list(M._PACKED_PUSH6)
lo = 236642560
assert specs[9][1] == lo and specs[10][1] - specs[10][0] == 30118144 and lo + 30118144 <= max(s[1] for s in specs)
assert not [i for i, s in enumerate(specs) if i != 24 and s[0] < lo + 30118144 and lo < s[1]]      # 24 = unused fallback leaf spec
specs[10] = (lo, lo + 30118144, specs[10][2])
rng = np.random.default_rng(0)
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
with flops.BudgetContext(flop_budget=2**55, quiet=True) as ctx:
    arena = fnp.empty((M._PACKED_ELEMENTS,), dtype=fnp.float32)
    M._SCRATCH = [arena, M._cp3_bind_arena_views(arena, M._PACKED_PUSH), M._cp3_bind_arena_views(arena, M._PACKED_READ), None, None, None, None]
    M._bind_push6(arena)
    views_cf = M._cp3_bind_arena_views(arena, tuple(specs))
    for name, m, k, n in (("young push packet", 1024, 896, 4096), ("young push packet 2", 1024, 896, 1920), ("rebuild packet", 1024, 384, 4096),
                          ("young push k1024", 1024, 1024, 4096)):
        a = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); b = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
        arms = (("S11", lambda o: M._packed_product_cached(a, b, o, M._SCRATCH_PUSH6, 6)), ("copy-free", lambda o: cf_packet(a, b, o, views_cf, 6)))
        res = {label: ([], [], [], []) for label, _ in arms}
        for rep in range(REPS):
            for label, fn in (arms if rep % 2 == 0 else arms[::-1]):
                outs, bills, ops, cpus = res[label]
                o = fnp.empty((m, n), dtype=fnp.float32)
                f0, q0, u0 = ctx._flops_used, len(ctx._op_log), resource.getrusage(resource.RUSAGE_SELF).ru_utime
                fn(o)
                u1 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
                if rep:
                    bills.append(ctx._flops_used - f0); ops.append(len(ctx._op_log) - q0); cpus.append(u1 - u0)
                outs[:] = [np.asarray(o).copy()]
        s = {k2: (v[0][0], statistics.median(v[1]), statistics.median(v[2]), statistics.median(v[3])) for k2, v in res.items()}
        s, c = s["S11"], s["copy-free"]
        dcpu = [x - y for x, y in zip(res["copy-free"][3], res["S11"][3])]
        print(f"{name} ({m}x{k}x{n}): bit-identical {s[0].tobytes() == c[0].tobytes()}; bill {s[1]:,.0f} -> {c[1]:,.0f} ({c[1] - s[1]:+,.0f} = {100 * (c[1] / s[1] - 1):+.2f}%); "
              f"ops {s[2]:.0f} -> {c[2]:.0f} ({c[2] - s[2]:+.0f}); user CPU {1e3 * s[3]:.1f} -> {1e3 * c[3]:.1f} ms (paired diffs, ms: median {1e3 * statistics.median(dcpu):+.1f}, all {[round(1e3 * x, 1) for x in dcpu]})", flush=True)
