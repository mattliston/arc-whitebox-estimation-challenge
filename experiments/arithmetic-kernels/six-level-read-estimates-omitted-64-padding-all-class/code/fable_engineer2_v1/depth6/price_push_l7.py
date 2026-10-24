#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-P7): PAPER price of SEVEN-level pushes on the engineer's E-46 in-situ data: every push kept at six levels
(S7 rule), moved to seven. Bill: the kernel's Winograd formula at L=7 minus L=6 (on the padded dims, as the push5p6 path pads to 64;
L=7 pads to 128). User CPU: the fitted in-situ model (depth_model_s5_user.json) with push,6 coefficients at L=7 minus L=6
(the per-leaf-matrix term carries 7^7 = 823,543 leaf matrices, 8 rows each at m=1024). Also the layout the kernel would need.
usage: price_push_l7.py"""
import json, collections, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent; ENG = H.parent.parent / "fable_engineer_v1"
c6 = json.load(open(ENG / "depth_model_s5_user.json"))["coef"]["push,6"]
def s_(L): return sum((7 / 4) ** j for j in range(1, L + 1))
def bill(m, k, n, L):
    g = (7 / 4) ** L
    return 2 * m * k * n * (7 / 8) ** L - m * n * g + (7 / 3) * (m * k + k * n + m * n) * (g - 1)
def user(m, k, n, L):
    P = 2 if n > 4096 else 1; s = s_(L)
    return c6[0] + c6[1] * m * k * s * P + c6[2] * k * n * s + c6[3] * m * n * s + c6[4] * 7 ** L * P + c6[5] * 2 * m * k * n * (7 / 8) ** L
def pad(x, u): return (x + u - 1) // u * u
nets = []
for f in ("depth_shadow_s5A.json", "depth_shadow_s5B.json"):
    d = json.load(open(ENG / f))
    for r in d[1:]:
        kept = {tuple(k[0]): k[2] for k in r["classes"]}
        db = du = 0.0; cls = 0; biggest = None
        for cell in r["cells"]:
            kind, m, k, n, L = cell[0]
            if kind != "push" or L != 6 or kept.get((kind, m, k, n)) != 6:
                continue
            calls = cell[1]; k6, n6 = pad(k, 64), pad(n, 64); k7, n7 = pad(k, 128), pad(n, 128)
            b = (bill(m, k7, n7, 7) - bill(m, k6, n6, 6)) * calls; u = (user(m, k7, n7, 7) - user(m, k6, n6, 6)) * calls
            db += b; du += u; cls += 1
            if biggest is None or cell[4] > biggest[1]: biggest = ((m, k, n), cell[4] / calls, b / calls, u / calls)
        nets.append((r["rank"], r["kept_bill"], db, du, cls, biggest))
b = [100 * x[2] / x[1] for x in nets]; u = [x[3] for x in nets]
print(f"{len(nets)} warm nets; six-level push classes per net {st.mean(x[4] for x in nets):.1f}")
print(f"ALL six-level pushes -> seven: bill {st.mean(b):+.3f}% per net, user CPU {st.mean(u):+.2f} s (platform kernel {1.2*st.mean(u):+.2f}..{1.5*st.mean(u):+.2f} s)")
m0 = nets[0][5]; print(f"example (net r{nets[0][0]}) biggest push {m0[0]}: bill per call {m0[1]/1e9:.2f} Gflop, L6->L7 {m0[2]/1e9:+.3f} Gflop, user {m0[3]:+.3f} s")
m, k, n = 1024, 1024, 4096
arena = sum(7 ** l * ((m >> l) * (k >> l) + (k >> l) * (n >> l) + (m >> l) * (n >> l)) for l in range(1, 8))
print(f"a 1024x1024x4096 push at 7 levels: leaf matrices 7^7 = {7**7:,} of {m>>7}x{k>>7} by {k>>7}x{n>>7}; all level arrays (no reuse) {arena/1e6:.0f}M floats vs arena 391.6M")
