"""TAG opus-release: micro-benchmark (numpy only, under 1 GB, light): cost of the tree kernel's level ops as single-component ops
against the fused pair ops on strided views that the level-fusion layer installs, per level, in instructions and best-of-5 cycles.
Replicates the candidate's views (_ipt_side / _lvf_ops): tree (7,)*L + (r, c); kinds A (left operand), B (right operand), C (output).
usage (OpenBLAS env python): fuse_bench.py <L> <r> <c>"""
import sys, numpy as np
import rusage
L = int(sys.argv[1]); r, c = int(sys.argv[2]), int(sys.argv[3]); REPS = 5
def measure(f):
    best = None
    for _ in range(REPS):
        c0, i0 = rusage.raw(); f(); c1, i1 = rusage.raw()
        cur = (c1 - c0, i1 - i0)
        best = cur if best is None or cur[0] < best[0] else best
    return best
rng = np.random.default_rng(0)
for kind, data in ((0, slice(0, 7, 2)), (1, slice(0, 4)), (2, slice(0, 4))):
    T = rng.standard_normal((7,) * L + (r, c), dtype=np.float32)
    tot = [0, 0, 0, 0]
    print(f"kind {'ABC'[kind]} tree {(7,)*L + (r, c)} = {T.size/1e6:.1f}M floats")
    for j in (range(L, 1, -1) if kind == 2 else range(L, 0, -1)):
        nr = L - j
        if j == 1:
            # level-1 chunk views: T[(data,)*(L-1)] -> last tree axis is the component axis
            Tj = T[(data,) * nr]
        else:
            Tj = T[(data,) * nr] if nr else T
        X = [np.take(Tj, i, axis=nr) if False else Tj[(slice(None),) * nr + (i,)] for i in range(7)]
        V = np.moveaxis(Tj, nr, 0)
        elems = X[0].size
        if kind == 0:
            def single(): np.subtract(X[4], X[0], out=X[5]); np.subtract(X[2], X[5], out=X[3]); np.subtract(X[6], X[5], out=X[1])
            def fused(): np.subtract(X[4], X[0], out=X[5]); np.subtract(V[2:7:4], X[5], out=V[3:0:-2])
        elif kind == 1:
            Vb = Tj
            sl = lambda s: Vb[(slice(None),) * nr + (s,)]
            def single(): np.add(X[1], X[0], out=X[5]); np.subtract(X[5], X[2], out=X[6]); np.subtract(X[3], X[5], out=X[4])
            def fused(): np.add(X[1], X[0], out=X[5]); np.subtract(sl(slice(5, 2, -2)), sl(slice(2, 6, 3)), out=sl(slice(6, 3, -2)))
        else:
            def single():
                np.add(X[0], X[5], out=X[5]); np.add(X[0], X[2], out=X[0]); np.add(X[5], X[4], out=X[5])
                np.add(X[5], X[3], out=X[2]); np.add(X[5], X[1], out=X[3]); np.add(X[4], X[6], out=X[1])
            def fused():
                np.add(X[0], V[5:1:-3], out=V[5::-5]); np.add(X[5], X[4], out=X[5]); np.add(X[5], V[3:0:-2], out=V[2:4]); np.add(X[4], X[6], out=X[1])
        s = measure(single); f = measure(fused)
        tot[0] += s[0]; tot[1] += s[1]; tot[2] += f[0]; tot[3] += f[1]
        print(f"  level {j}: {elems/1e6:6.2f}M elems/op | single {s[0]/1e6:8.1f} Mcyc {s[1]/1e6:8.1f} Minstr ({s[0]/elems:5.2f} cyc/elem) | "
              f"fused {f[0]/1e6:8.1f} Mcyc {f[1]/1e6:8.1f} Minstr | fused/single cycles {f[0]/s[0]:.2f} instr {f[1]/s[1]:.2f}")
    print(f"  ALL LEVELS: single {tot[0]/1e6:.0f} Mcyc {tot[1]/1e6:.0f} Minstr | fused {tot[2]/1e6:.0f} Mcyc {tot[3]/1e6:.0f} Minstr | fused/single cycles {tot[2]/tot[0]:.2f} instr {tot[3]/tot[1]:.2f}")
