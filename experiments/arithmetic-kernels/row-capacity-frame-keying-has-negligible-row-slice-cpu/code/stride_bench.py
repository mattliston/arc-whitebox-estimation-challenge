#!/usr/bin/env python3
"""TAG fable-engineer (E-50): user-CPU cost of capacity-keyed (sliced, strided) Winograd frames against S11's exact contiguous
frames, at S11's real level shapes, in plain numpy (1 thread). Per level: one transform op (add with out=: inputs are quadrant views
of the parent level array, output a slot of the child array), the level's quadrant copies (stack with out=), and at the leaf the
batched matmul. EXACT = arrays of the exact shape (as S11 binds them); CAP = the same shapes sliced from the layout's capacity arrays.
usage: stride_bench.py -> prints per-shape ratios CAP/EXACT (median of repeats)"""
import time, numpy as np
rng = np.random.default_rng(0)
def t(f, reps):
    xs = []
    for _ in range(reps):
        t0 = time.thread_time(); f(); xs.append(time.thread_time() - t0)
    return float(np.median(xs))
def level(par_lead, r, c, R, C, reps=15):
    """one level: parent blocks (lead, 2r, 2c) -> child slot (lead, r, c); capacity parent (lead, 2R, 2C), child (lead, R, C)"""
    pe = rng.standard_normal(par_lead + (2 * r, 2 * c), dtype=np.float32); ce = np.empty((3,) + par_lead + (r, c), np.float32)
    pc_full = rng.standard_normal(par_lead + (2 * R, 2 * C), dtype=np.float32); pc = pc_full[..., :2 * r, :2 * c]
    cc_full = np.empty((3,) + par_lead + (R, C), np.float32); cc = cc_full[..., :r, :c]
    def add(p, ch):
        np.add(p[..., r:, :c], p[..., r:, c:], out=ch[0])
    def stk(p, ch):
        np.stack((p[..., :r, :c], p[..., :r, c:]), axis=0, out=ch[:2])
    return (t(lambda: add(pe, ce), reps), t(lambda: add(pc, cc), reps), t(lambda: stk(pe, ce), reps), t(lambda: stk(pc, cc), reps))
def leaf(nb, r, k, c, R, K, C, reps=7):
    ae = rng.standard_normal((nb, r, k), dtype=np.float32); be = rng.standard_normal((nb, k, c), dtype=np.float32); oe = np.empty((nb, r, c), np.float32)
    ac = rng.standard_normal((nb, R, K), dtype=np.float32)[:, :r, :k]; bc = rng.standard_normal((nb, K, C), dtype=np.float32)[:, :k, :c]
    oc = np.empty((nb, R, C), np.float32)[:, :r, :c]
    return t(lambda: np.matmul(ae, be, out=oe), reps), t(lambda: np.matmul(ac, bc, out=oc), reps)
cases = [
    # young read L11 (resp 640, n_y 7360, live 864) at D5; capacity of the read layout: m 1024, k 8192, n 1024
    ("read A L2", (7,), 160, 1840, 256, 2048), ("read A L3", (7, 7), 80, 920, 128, 1024), ("read A L4", (7, 7, 7), 40, 460, 64, 512),
    ("read B L2", (7,), 1840, 216, 2048, 256), ("read B L3", (7, 7), 920, 108, 1024, 128), ("read B L4", (7, 7, 7), 460, 54, 512, 64),
    # young push L11 (k 896 padded, packet n 4096) at D6; capacity of the push6 layout: k 1024, n 4096
    ("push B L2", (7,), 224, 1024, 256, 1024), ("push B L3", (7, 7), 112, 512, 128, 512), ("push B L4", (7, 7, 7), 56, 256, 64, 256), ("push B L5", (7, 7, 7, 7), 28, 128, 32, 128),
]
print("case: add EXACT ms, CAP ms, ratio | stack EXACT ms, CAP ms, ratio")
tot = np.zeros(4)
for name, lead, r, c, R, C in cases:
    a0, a1, s0, s1 = level(lead, r, c, R, C)
    tot += (a0, a1, s0, s1)
    print(f"{name:10s} add {a0 * 1e3:7.2f} {a1 * 1e3:7.2f} {a1 / a0:5.2f} | stack {s0 * 1e3:7.2f} {s1 * 1e3:7.2f} {s1 / s0:5.2f}")
print(f"ALL levels: add ratio {tot[1] / tot[0]:.3f}, stack ratio {tot[3] / tot[2]:.3f}")
for name, nb, r, k, c, R, K, C in (("read leaf D5", 7 ** 4, 20, 230, 27, 32, 256, 32), ("push leaf D6", 7 ** 5, 16, 14, 64, 16, 16, 64)):
    m0, m1 = leaf(nb, r, k, c, R, K, C)
    print(f"{name}: matmul EXACT {m0 * 1e3:.1f} ms, CAP {m1 * 1e3:.1f} ms, ratio {m1 / m0:.3f}")
