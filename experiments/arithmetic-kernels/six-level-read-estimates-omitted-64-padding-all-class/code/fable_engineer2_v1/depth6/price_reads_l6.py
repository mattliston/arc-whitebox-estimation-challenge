#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-R6): PAPER price of six-level READS on the engineer's in-situ E-46 data (fable_engineer_v1/depth_shadow_s5A/
s5B.json: S7 file, every read executed at depths 4 and 5 in place; warm nets only). A six-level read cannot run today: the read
layout has 4 levels + read5 packets, and level 6 at k = 8192 needs ~5e8 floats (> the 3.9e8-float arena), so it would need
packetized levels 5-6. Per read class and warm net: measured depth-5 bill and user CPU per call (the kept depth); depth-6 change =
(a) bill: the kernel's Winograd bill formula (as _dr_value: 2mkn(7/8)^L - mn g + (7/3)(mk+kn+mn)(g-1), g=(7/4)^L) at L=6 minus L=5;
(b) user CPU: the engineer's in-situ-fitted model (depth_model_s5_user.json, features const, mk s, kn s, mn s, 7^L, 2mkn(7/8)^L)
    at L=6 minus L=5, with transform/GEMM coefficients from read,5 (variant R) or push,6 (variant P), and const/leaf terms from push,6.
Aggregates per net: all reads at six levels, and slices ranked by Gflop saved per user-second. Platform kernel = user x 1.2-1.5.
usage: price_reads_l6.py"""
import json, collections, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent; ENG = H.parent.parent / "fable_engineer_v1"
M = json.load(open(ENG / "depth_model_s5_user.json"))["coef"]
def s_(L): return sum((7 / 4) ** j for j in range(1, L + 1))
def bill(m, k, n, L):
    g = (7 / 4) ** L
    return 2 * m * k * n * (7 / 8) ** L - m * n * g + (7 / 3) * (m * k + k * n + m * n) * (g - 1)
def user(m, k, n, L, c):
    s = s_(L)
    return c[0] + c[1] * m * k * s + c[2] * k * n * s + c[3] * m * n * s + c[4] * 7 ** L + c[5] * 2 * m * k * n * (7 / 8) ** L
r5, p6 = M["read,5"], M["push,6"]
VAR = {"R": [p6[0], r5[1], r5[2], r5[3], p6[4], r5[5]], "P": [p6[0], p6[1], p6[2], p6[3], p6[4], max(p6[5], r5[5])]}
nets = []
for f in ("depth_shadow_s5A.json", "depth_shadow_s5B.json"):
    d = json.load(open(ENG / f))
    for r in d[1:]:                                   # the first net of each run is the cold warm-up
        ucpu = collections.defaultdict(float)
        for c in r["calls"]:
            ucpu[(c[1], c[2], c[3], c[4], c[5])] += c[11]
        kept = {tuple(k[0]): k[1][1] if isinstance(k[1], list) else k[1] for k in r["classes"]}
        rows = []
        for cell in r["cells"]:
            kind, m, k, n, L = cell[0]; calls = cell[1]
            if kind != "read" or L != 5:
                continue
            db = (bill(m, k, n, 6) - bill(m, k, n, 5)) * calls
            du = {v: (user(m, k, n, 6, c) - user(m, k, n, 5, c)) * calls for v, c in VAR.items()}
            rows.append(dict(shape=(m, k, n), calls=calls, bill5=cell[4], user5=ucpu[(kind, m, k, n, 5)], db=db, du=du))
        nets.append(dict(rank=r["rank"], total=r["kept_bill"], rows=rows))
print(f"{len(nets)} warm nets (E-46 s5A+s5B); per net: read classes, depth-5 read bill share, all-reads-at-6 deltas")
for v in ("R", "P"):
    allb = [100 * sum(x["db"] for x in n["rows"]) / n["total"] for n in nets]
    allu = [sum(x["du"][v] for x in n["rows"]) for n in nets]
    share = [100 * sum(x["bill5"] for x in n["rows"]) / n["total"] for n in nets]
    print(f"variant {v}: reads at depth 5 = {st.mean(share):.1f}% of the bill; ALL reads 5->6: bill {st.mean(allb):+.3f}% per net, user CPU {st.mean(allu):+.2f} s "
          f"(platform kernel {1.2*st.mean(allu):+.2f}..{1.5*st.mean(allu):+.2f} s)")
    pool = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for n in nets:
        for x in n["rows"]:
            p = pool[x["shape"]]; p[0] += x["db"] / n["total"] / len(nets); p[1] += x["du"][v] / len(nets); p[2] += x["calls"]
    ranked = sorted(pool.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1e-9))      # most negative bill per second first
    cb = cu = 0.0; marks = (0.25, 0.5, 0.75, 1.0); mi = 0; n_cls = len(ranked)
    for i, (shape, (b, u, c)) in enumerate(ranked):
        cb += b; cu += u
        if (i + 1) / n_cls >= marks[mi]:
            rate = -cb * 100 / max(cu, 1e-9)
            print(f"   best {int(100*marks[mi])}% of read classes ({i+1}): bill {100*cb:+.3f}%, user {cu:+.2f} s, platform {1.2*cu:+.2f}..{1.5*cu:+.2f} s; {rate:.3f}% bill per user-s")
            mi += 1
            if mi == len(marks): break
