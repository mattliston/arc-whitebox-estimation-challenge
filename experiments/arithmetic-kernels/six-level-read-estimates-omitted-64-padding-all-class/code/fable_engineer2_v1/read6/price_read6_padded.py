#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-R6P): six-level READ price WITH PADDING. A six-level split needs m, k, n divisible by 64; most big reads have
m or n = odd multiples of 32 (e.g. 928 x 8128 x 992). Price as the push5p6 rule prices its padded pushes: depth-6 bill on the dims padded
to multiples of 64, plus the zero-padded operand copies (m'k' + k'n' billed), plus the design extras of the q=1 packet design (level-5
slot copies 7x3x343x(m'k'+k'n')/1024; +227 requests per read at 0.41 ms). User CPU from the in-situ model at the padded dims.
Compared per class with the unpadded paper price. Per warm net (E-46 s5A+s5B). usage: price_read6_padded.py"""
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
P = lambda x: (x + 63) // 64 * 64
nets = []
for f in ("depth_shadow_s5A.json", "depth_shadow_s5B.json"):
    for r in json.load(open(ENG / f))[1:]:
        nets.append((r["kept_bill"], [(tuple(c[0][1:4]), c[1], c[4] / c[1]) for c in r["cells"] if c[0][0] == "read" and c[0][4] == 5]))
pool = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
for tot, rows in nets:
    for (m, k, n), calls, b5meas in rows:
        mp, kp, npd = P(m), P(k), P(n); padded = (mp, kp, npd) != (m, k, n)
        b6 = bill(mp, kp, npd, 6) + ((mp * kp + kp * npd) if padded else 0) + 7 * 3 * 343 * (mp * kp + kp * npd) / 1024
        u6 = user(mp, kp, npd, 6, CR) - user(m, k, n, 5, CR)
        p = pool[(m, k, n)]; w = calls / len(nets)
        p[0] += w; p[1] += (bill(m, k, n, 6) - bill(m, k, n, 5)) * w / tot; p[2] += (b6 - bill(m, k, n, 5)) * w / tot; p[3] += u6 * w
        p[4] = 1.0 if padded else 0.0
cls = list(pool.items())
npad = sum(1 for _, v in cls if v[4]); print(f"{len(cls)} read classes at depth 5; {npad} need padding to reach six levels ({100*npad/len(cls):.0f}%)")
paper = sum(v[1] for _, v in cls); real = sum(v[2] for _, v in cls)
print(f"ALL reads: unpadded paper bill {100*paper:+.3f}% -> padded design bill {100*real:+.3f}% per net")
good = sorted([(k, v) for k, v in cls if v[2] < 0], key=lambda kv: kv[1][2] / max(kv[1][3] + 227 * kv[1][0] * 0.41e-3 / 1.35, 1e-9))
print(f"classes that still SAVE bill after padding + design copies: {len(good)} of {len(cls)}")
cb = cu = cc = 0.0
for i, (shape, v) in enumerate(good):
    cb += v[2]; cu += v[3]; cc += v[0]
    if i + 1 in (3, 5, 10, len(good)):
        plat = (1.2 * cu + 227 * cc * 0.41e-3, 1.5 * cu + 227 * cc * 0.41e-3)
        print(f"  best {i+1:2d}: {cc:.1f} calls/net, bill {100*cb:+.3f}%, user {cu:+.2f} s, platform wall {plat[0]:+.2f}..{plat[1]:+.2f} s, rate {100*-cb/max(plat[1],1e-9):.3f}%/s; e.g. {[s for s, _ in good[:3]]}")
