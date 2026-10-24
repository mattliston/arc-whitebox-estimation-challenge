#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-R6D): the six-level-read DESIGN cost that the paper price (price_reads_l6.py, kernel user CPU only) left out.
Feasible design under the 3.9e8-float arena: level 6 inside read5's packets with ONE level-4 slot per packet (7 packets per read; A5+A6
= 54.1M floats at m=1024, k=8192, inside the 59.0M read5 child region), the three untransformed level-5 slots per side copied into a
contiguous stack so level 6 runs batched: ~41 ops per packet = 287 per read vs read5's 60 (3 packets x 20) -> +227 requests per read,
copies 7 x 3 x 343 x (mk + kn)/1024 billed floats per read. Platform cost per request 0.41 ms (0.22 ms round trip + ~0.19 ms server
overhead: S11 overhead ~8.2 s / ~43k requests). Per warm net (E-46 s5A+s5B), over the best-quarter read classes (variant R ranking).
usage: price_read6_design.py"""
import json, collections, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent; ENG = H.parent.parent / "fable_engineer_v1"
M = json.load(open(ENG / "depth_model_s5_user.json"))["coef"]
def s_(L): return sum((7 / 4) ** j for j in range(1, L + 1))
def bill(m, k, n, L):
    g = (7 / 4) ** L
    return 2 * m * k * n * (7 / 8) ** L - m * n * g + (7 / 3) * (m * k + k * n + m * n) * (g - 1)
def user(m, k, n, L, c):
    s = s_(L); return c[0] + c[1] * m * k * s + c[2] * k * n * s + c[3] * m * n * s + c[4] * 7 ** L + c[5] * 2 * m * k * n * (7 / 8) ** L
r5, p6 = M["read,5"], M["push,6"]; CR = [p6[0], r5[1], r5[2], r5[3], p6[4], r5[5]]
nets = []
for f in ("depth_shadow_s5A.json", "depth_shadow_s5B.json"):
    for r in json.load(open(ENG / f))[1:]:
        rows = [(tuple(c[0][1:4]), c[1]) for c in r["cells"] if c[0][0] == "read" and c[0][4] == 5]
        nets.append((r["kept_bill"], rows))
pool = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for tot, rows in nets:
    for (m, k, n), calls in rows:
        p = pool[(m, k, n)]; p[0] += calls / len(nets); p[1] += (bill(m, k, n, 6) - bill(m, k, n, 5)) * calls / tot / len(nets)
        p[2] += (user(m, k, n, 6, CR) - user(m, k, n, 5, CR)) * calls / len(nets)
ranked = sorted(pool.items(), key=lambda kv: kv[1][1] / max(kv[1][2], 1e-9))
best = ranked[: len(ranked) // 4]
calls = sum(v[0] for _, v in best); db = sum(v[1] for _, v in best); du = sum(v[2] for _, v in best)
tot = st.mean(t for t, _ in nets)
copy = sum(v[0] * 7 * 3 * 343 * (m * k + k * n) / 1024 for (m, k, n), v in best) / tot
req = 227 * calls
print(f"best quarter: {len(best)} classes, {calls:.1f} read calls per net")
print(f"  paper (kernel only): bill {100*db:+.3f}%, user {du:+.2f} s -> platform kernel {1.2*du:+.2f}..{1.5*du:+.2f} s")
print(f"  design extras: copy bill {100*copy:+.3f}% -> net bill {100*(db+copy):+.3f}% ({100*(db+copy)/db:.0f}% of paper); requests +{req:.0f}/net -> platform +{req*0.41e-3:.2f} s")
print(f"  design total platform wall: {1.2*du + req*0.41e-3:+.2f}..{1.5*du + req*0.41e-3:+.2f} s for bill {100*(db+copy):+.3f}%  (paper: {1.2*du:+.2f}..{1.5*du:+.2f} s for {100*db:+.3f}%)")
for top in (5, 10, 15):
    sel = ranked[:top]; c = sum(v[0] for _, v in sel); b = sum(v[1] for _, v in sel); u = sum(v[2] for _, v in sel)
    cp = sum(v[0] * 7 * 3 * 343 * (m * k + k * n) / 1024 for (m, k, n), v in sel) / tot
    print(f"  top {top:2d} classes ({c:.1f} calls/net): net bill {100*(b+cp):+.3f}%, platform {1.2*u + 227*c*0.41e-3:+.2f}..{1.5*u + 227*c*0.41e-3:+.2f} s; shapes {[s for s, _ in sel[:3]]}")
