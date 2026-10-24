#!/usr/bin/env python3
"""TAG fable-engineer (E-53 v2): the in-place tree Winograd kernel (IPT) with the tree axes REVERSED in memory. v1 stored the tree as
(c1, c2, ..., cL, leaf) (root room outermost): an op on chunk c of every deepest room touched one leaf block in seven across the whole
tree, and the deepest level (most of the volume) ran 2x slower than S11's contiguous slot arrays (+100-150 ms per packet). v2 stores
(cL, ..., c2, c1, leaf): an op at level j is 4^(L-j) contiguous runs of 7^(j-1) leaf blocks, so the deepest level is fully contiguous.
Same tree semantics as v1 (see ipt_bench.py): A quadrant (i,j) at chunk 3i+j, B at i+2j, C at 2i+j; products m1..m7 at chunks
0 1 3 4 2 5 6 by slot (m1 0, m2 1, m5 2, m3 3, m4 4, m6 5, m7 6); no level copies below level 1; one leaf matmul; in-place unwind.
usage: ipt_bench_v2.py <candidate.py> [reps]"""
import sys, types, resource, statistics
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
_FR = {}


def _restrict(T, nr, pos6):
    # restrict the first nr axes of T to the 4 data chunks: {0,1,2,3} by a free slice, {0,1,3,4} as (i, j) pairs by a reshape
    if nr == 0:
        return T
    if not pos6:
        return T[(slice(0, 4),) * nr]
    rest = tuple(int(x) for x in T.shape[nr:])
    V = fnp.reshape(T[(slice(0, 6),) * nr], (2, 3) * nr + rest)
    return V[(slice(None), slice(0, 2)) * nr]


def ipt_frames(m, k, n, L):
    key = (m, k, n, L)
    f = _FR.get(key)
    if f is not None:
        return f
    rA, kA, nB = m >> L, k >> L, n >> L
    TA = fnp.empty((7,) * L + (rA, kA), dtype=fnp.float32)
    TB = fnp.empty((7,) * L + (kA, nB), dtype=fnp.float32)
    TC = fnp.empty((7,) * L + (rA, nB), dtype=fnp.float32)
    A_ch, B_ch, C_ch = [None] * (L + 1), [None] * (L + 1), [None] * (L + 1)
    for j in range(2, L + 1):                                        # level j: axes 0..L-j-1 restricted, axis L-j = the chunk
        nr = L - j
        VA = _restrict(TA, nr, True); VB = _restrict(TB, nr, False); VC = _restrict(TC, nr, False)
        A_ch[j] = [VA[(slice(None),) * (2 * nr) + (c,)] for c in range(7)]
        B_ch[j] = [VB[(slice(None),) * nr + (c,)] for c in range(7)]
        C_ch[j] = [VC[(slice(None),) * nr + (c,)] for c in range(7)]
    nr = L - 1                                                       # level 1: pairs for levels L..2 in (i, j) order, then the chunk
    VA = _restrict(TA, nr, True)
    VB = fnp.reshape(_restrict(TB, nr, False), (2, 2) * nr + (7, kA, nB))
    VB = fnp.transpose(VB, [x for p in range(nr) for x in (2 * p + 1, 2 * p)] + [2 * nr, 2 * nr + 1, 2 * nr + 2])   # B pairs are (j, i)
    VC = fnp.reshape(_restrict(TC, nr, False), (2, 2) * nr + (7, rA, nB))
    A_ch[1] = [VA[(slice(None),) * (2 * nr) + (c,)] for c in range(7)]
    B_ch[1] = [VB[(slice(None),) * (2 * nr) + (c,)] for c in range(7)]
    C_ch[1] = [VC[(slice(None),) * (2 * nr) + (c,)] for c in range(7)]
    A01 = VA[(slice(None),) * (2 * nr) + (slice(0, 2),)]; B01 = VB[(slice(None),) * (2 * nr) + (slice(0, 2),)]
    f = _FR[key] = (TA, TB, TC, A_ch, B_ch, C_ch, A01, B01)
    return f


def _bits(X, L, r, c):
    # (2^L r, 2^L c) matrix -> view with axes (i1, j1, iL, jL, ..., i2, j2, row, col)
    V = fnp.reshape(X, (2,) * L + (r,) + (2,) * L + (c,))
    perm = [0, L + 1] + [x for p in range(L, 1, -1) for x in (p - 1, L + p)] + [L, 2 * L + 1]
    return fnp.transpose(V, perm)


def ipt_product(a, b, out, L):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    TA, TB, TC, A_ch, B_ch, C_ch, A01, B01 = ipt_frames(m, k, n, L)
    rA, kA, nB = m >> L, k >> L, n >> L
    nr = L - 1
    hold = []; h = hold.append
    Aq = _bits(a, L, rA, kA); Bq = _bits(b, L, kA, nB); Cq = _bits(out, L, rA, nB)
    a11, a12, a21, a22 = Aq[0, 0], Aq[0, 1], Aq[1, 0], Aq[1, 1]
    b11, b12, b21, b22 = Bq[0, 0], Bq[0, 1], Bq[1, 0], Bq[1, 1]
    h((Aq, Bq, Cq, a11, a12, a21, a22, b11, b12, b21, b22))
    X = A_ch[1]
    h(fnp.add(a21, a22, out=X[2])); h(fnp.subtract(a11, a21, out=X[6])); h(fnp.subtract(X[2], a11, out=X[5])); h(fnp.subtract(a12, X[5], out=X[3]))
    h(fnp.copyto(A01, fnp.moveaxis(Aq[0], 0, 2 * nr))); h(fnp.copyto(X[4], a22))
    Y = B_ch[1]
    h(fnp.subtract(b22, b12, out=Y[6])); h(fnp.subtract(b12, b11, out=Y[2])); h(fnp.subtract(b22, Y[2], out=Y[5])); h(fnp.subtract(Y[5], b21, out=Y[4]))
    h(fnp.copyto(B01, fnp.moveaxis(Bq[:, 0], 0, 2 * nr))); h(fnp.copyto(Y[3], b22))
    for j in range(2, L + 1):
        X = A_ch[j]
        h(fnp.add(X[3], X[4], out=X[2])); h(fnp.subtract(X[0], X[3], out=X[6])); h(fnp.subtract(X[2], X[0], out=X[5])); h(fnp.subtract(X[1], X[5], out=X[3]))
        Y = B_ch[j]
        h(fnp.subtract(Y[3], Y[2], out=Y[6])); h(fnp.subtract(Y[2], Y[0], out=Y[2])); h(fnp.subtract(Y[3], Y[2], out=Y[5])); h(fnp.subtract(Y[5], Y[1], out=Y[4]))
    h(fnp.matmul(TA, TB, out=TC))
    for j in range(L, 1, -1):
        Z = C_ch[j]
        h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[1], out=Z[0])); h(fnp.add(Z[5], Z[6], out=Z[6])); h(fnp.add(Z[5], Z[2], out=Z[5]))
        h(fnp.add(Z[5], Z[3], out=Z[1])); h(fnp.add(Z[6], Z[2], out=Z[3])); h(fnp.subtract(Z[6], Z[4], out=Z[2]))
    Z = C_ch[1]
    h(fnp.add(Z[0], Z[5], out=Z[5])); h(fnp.add(Z[0], Z[1], out=Cq[0, 0])); h(fnp.add(Z[5], Z[6], out=Z[6])); h(fnp.add(Z[5], Z[2], out=Z[5]))
    h(fnp.add(Z[5], Z[3], out=Cq[0, 1])); h(fnp.add(Z[6], Z[2], out=Cq[1, 1])); h(fnp.subtract(Z[6], Z[4], out=Cq[1, 0]))
    del hold
    return out


if __name__ == "__main__":
    cand = Path(sys.argv[1]).resolve()
    REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
    flops.configure(symmetry_warnings=False)
    M = types.ModuleType("ipt2_mod"); M.__file__ = str(cand); sys.modules[M.__name__] = M
    exec(compile(cand.read_text(), str(cand), "exec"), M.__dict__)
    rng = np.random.default_rng(0)
    with flops.BudgetContext(flop_budget=2**55, quiet=True) as ctx:
        arena = fnp.empty((M._PACKED_ELEMENTS,), dtype=fnp.float32)
        M._SCRATCH = [arena, M._cp3_bind_arena_views(arena, M._PACKED_PUSH), M._cp3_bind_arena_views(arena, M._PACKED_READ), None, None, None, None]
        M._bind_push6(arena)
        shapes = (("young push packet", 1024, 896, 4096), ("rebuild packet", 1024, 384, 4096), ("young push packet 2", 1024, 896, 1920))
        for name, m, k, n in shapes:
            a = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); b = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
            f0 = ctx._flops_used; ipt_frames(m, k, n, 6); bind = ctx._flops_used - f0
            arms = (("S11", lambda o: M._packed_product_cached(a, b, o, M._SCRATCH_PUSH6, 6)), ("IPT", lambda o: ipt_product(a, b, o, 6)))
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
            s, c = s["S11"], s["IPT"]
            dcpu = [x - y for x, y in zip(res["IPT"][3], res["S11"][3])]
            same = s[0].tobytes() == c[0].tobytes()
            md = float(np.max(np.abs(s[0].astype(np.float64) - c[0].astype(np.float64)))) if not same else 0.0
            print(f"{name} ({m}x{k}x{n}): bit-identical {same} (max diff {md:.3g}); bill {s[1]:,.0f} -> {c[1]:,.0f} ({c[1] - s[1]:+,.0f} = {100 * (c[1] / s[1] - 1):+.2f}%); "
                  f"IPT tree binding once {bind:,.0f}; ops {s[2]:.0f} -> {c[2]:.0f} ({c[2] - s[2]:+.0f}); user CPU {1e3 * s[3]:.1f} -> {1e3 * c[3]:.1f} ms "
                  f"(paired median {1e3 * statistics.median(dcpu):+.1f}, all {[round(1e3 * x, 1) for x in dcpu]})", flush=True)
            _FR.clear()
