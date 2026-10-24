#!/usr/bin/env python3
"""TAG fable-engineer (E-30): write a compute-budgeted depth-rule fragment (v3 = v2 + padded six-level pushes; v1/v2 generators kept).
The fragment = depthrule_fragment_v0p.py (five-level push layout in 4096 packets, depth-6 squares, _DR_FORCE hook) + a new _rule_depth:
  push (1024-row capacity): depth = argmin_L  bill(L) + MU * t(L)   over L in {4, 5} (+6 when m == 1024 and k, n divide 64)
  square with bank depth 5 and all dims dividing 64 (not a W5 grant): 6 if bill(6) + MU_SQ*t(6) < bill(5) + MU_SQ*t(5)
  everything else (reads, other squares, W5 grants): the bank rule, unchanged.
bill(L) = the bank rule's formula x the kernel's fitted bill ratio; t(L) = fitted compute model (bench numpy seconds, 1 thread):
  t = c0 + c1*m*k*S*packets + c2*k*n*S + c3*m*n*S + c4*7^L*packets + c5*2mkn(7/8)^L + REQ_S*requests,  S = sum_{j<=L} (7/4)^j.
MU, MU_SQ in flops of bill per second of compute. Shape-only, deterministic, pure Python (no array work, no timing at run time).
usage: make_depthrule_fragment_v3.py <depth_bench.json> <mu Gflop/s> <mu_sq Gflop/s> <out fragment.py>"""
import sys, json, collections
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
HERE = Path(__file__).resolve().parent
bench = json.load(open(sys.argv[1])); MU = float(sys.argv[2]) * 1e9; MU_SQ = float(sys.argv[3]) * 1e9; OUT = Path(sys.argv[4])
assert not OUT.exists(), f"refusing to overwrite {OUT}"
S = lambda L: sum((7 / 4) ** j for j in range(1, L + 1))
def formula(m, k, n, L):
    packets = (4096, n - 4096) if L >= 5 and n > 4096 else (n,)
    g = (7 / 4) ** L
    return sum(2 * m * k * nc * (7 / 8) ** L - m * nc * g + (7 / 3) * (m * k + k * nc + m * nc) * (g - 1) for nc in packets)
def feats(m, k, n, L):
    s = S(L); p = 2 if (L >= 5 and n > 4096) else 1
    return [1.0, m * k * s * p, k * n * s, m * n * s, 7 ** L * p, 2 * m * k * n * (7 / 8) ** L]
coef, ratio, req = {}, {}, {}
groups = collections.defaultdict(list)
for r in bench:
    groups[(r["kind"], r["depth"])].append(r)
for key in (("push", 4), ("push", 5), ("push", 6), ("sq", 5), ("sq", 6)):
    rows = groups[key]
    X = np.array([feats(r["m"], r["k"], r["n"], r["depth"]) for r in rows]); y = np.array([r["np_med"] for r in rows])
    sc = X.max(axis=0) + 1e-30
    c, _ = nnls(X / sc, y); coef[key] = tuple(float(v) for v in c / sc)
    ratio[key] = float(np.median([r["bill"] / formula(r["m"], r["k"], r["n"], r["depth"]) for r in rows]))
    req[key] = int(np.median([r["req"] / (2 if (key[1] >= 5 and r["n"] > 4096) else 1) for r in rows]))
base = (HERE / "depthrule_fragment_v0q.py").read_text().replace("# END FABLE-ENGINEER DEPTH RULE v0q\n", "")
head = base.replace("# BEGIN FABLE-ENGINEER DEPTH RULE v0q (E-30, 2026-09-23; = v0p + padded six-level pushes)", "# BEGIN FABLE-ENGINEER DEPTH RULE v3 (E-30, 2026-09-23; = v2 + padded six-level pushes)", 1)
rule = f'''
# v3: the depth of a push (a 1024-row push whose k or n is not a multiple of 64 may take six levels, zero-padded, priced with its pad copies), and of a depth-5 square whose dims divide 64, minimizes bill + MU * compute seconds (fitted model:
# fable_engineer_v1/make_depthrule_fragment_v2.py on {Path(sys.argv[1]).name}; seconds = in-process numpy time, 1 thread).
# MU = {MU/1e9:g} Gflop of bill per compute-second (pushes), {MU_SQ/1e9:g} (squares).
_DR_MU = {MU!r}
_DR_MU_SQ = {MU_SQ!r}
_DR_REQ_S = 0.00015
_DR_COEF = {coef!r}
_DR_RATIO = {ratio!r}
_DR_REQ = {req!r}
_DR_CACHE = {{}}


def _dr_value(kind, m, k, n, L, mu):
    packets = (4096, n - 4096) if kind == 'push' and L >= 5 and n > 4096 else (n,)
    g = (7 / 4) ** L
    bill = sum(2 * m * k * nc * (7 / 8) ** L - m * nc * g + (7 / 3) * (m * k + k * nc + m * nc) * (g - 1) for nc in packets)
    s = sum((7 / 4) ** j for j in range(1, L + 1))
    c = _DR_COEF[(kind, L)]
    seconds = (c[0] + c[1] * m * k * s * len(packets) + c[2] * k * n * s + c[3] * m * n * s + c[4] * 7 ** L * len(packets)
               + c[5] * 2 * m * k * n * (7 / 8) ** L + _DR_REQ_S * _DR_REQ[(kind, L)] * len(packets))
    return _DR_RATIO[(kind, L)] * bill + mu * seconds


def _rule_depth(kind, m, k, n, maximum=6):
    forced = _DR_FORCE.get((kind, m, k, n))
    granted = kind == 'sq' and _W5_PORT_SCOPE is not None and _W5_PORT_SCOPE['grant'] is not None
    if forced is not None and not granted:
        return forced
    bank = _DR_BANK_RULE(kind, m, k, n, maximum)
    if granted or kind == 'read':
        return bank
    key = (kind, m, k, n, bank)
    found = _DR_CACHE.get(key)
    if found is not None:
        return found
    choice = bank
    if kind == 'push' and bank != 0 and m <= 1024 and k <= 1024 and m % 32 == k % 32 == n % 32 == 0:
        if m == 1024 and m % 64 == k % 64 == n % 64 == 0:
            options = (4, 5, 6)
        elif m == 1024:
            options = (4, 5, 7)   # 7 = six levels on the zero-padded product
        else:
            options = (4, 5)
        def value(L):
            if L != 7:
                return _dr_value('push', m, k, n, L, _DR_MU)
            kp, npad = (k + 63) // 64 * 64, (n + 63) // 64 * 64
            return _dr_value('push', m, kp, npad, 6, _DR_MU) + m * k + k * n + _DR_MU * (m * kp + kp * npad) / 2e9
        choice = min(options, key=value)
        choice = 6 if choice == 7 else choice
    elif kind == 'sq' and bank == 5 and m % 64 == k % 64 == n % 64 == 0:
        choice = 6 if _dr_value('sq', m, k, n, 6, _DR_MU_SQ) < _dr_value('sq', m, k, n, 5, _DR_MU_SQ) else 5
    _DR_CACHE[key] = choice
    return choice
# END FABLE-ENGINEER DEPTH RULE v3
'''
OUT.write_text(head + rule)
print(OUT, "coef", coef, "ratio", ratio, "req", req)
