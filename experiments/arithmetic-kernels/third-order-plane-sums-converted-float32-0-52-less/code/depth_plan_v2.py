#!/usr/bin/env python3
"""TAG fable-engineer (E-28, v2: five-level pushes run in 4096 packets like six-level ones; v1 kept): compute model + greedy depth plan from depth_bench.py timings and a product_profile.py census.
Time model per (kind, depth L), fitted by non-negative least squares on the bench medians (numpy seconds):
    t = c0 + c1*m*k*S(L) + c2*k*n*S(L) + c3*m*n*S(L) + c4*7^L + c5*leaf_flops(L)
S(L) = sum_{j=1..L} (7/4)^j (transform elements per operand element); 7^L = leaf count (per-leaf overhead); leaf_flops = 2mkn(7/8)^L.
Bill per (shape, L): the bank rule's formula times a per-(kind, L) ratio fitted on the bench's exact bills.
Plan: for exchange rate mu (Gflop per compute-second), every non-W5 product takes argmin_L bill(L) + mu*t(L) over its allowed depths;
report predicted d(bill) and d(time) vs the census's current depths (the census's own in-situ seconds anchor the current depth).
usage: depth_plan.py <depth_bench.json> <product_profile.json>"""
import sys, json, collections
import numpy as np
from scipy.optimize import nnls
bench = json.load(open(sys.argv[1])); prof = json.load(open(sys.argv[2]))
S = lambda L: sum((7 / 4) ** j for j in range(1, L + 1))
def formula(m, k, n, L):
    if L == 0:
        return 2 * m * k * n - m * n
    packets = (4096, n - 4096) if L >= 5 and n > 4096 else (n,)
    g = (7 / 4) ** L
    return sum(2 * m * k * nc * (7 / 8) ** L - m * nc * g + (7 / 3) * (m * k + k * nc + m * nc) * (g - 1) for nc in packets)
def feats(m, k, n, L):
    s = S(L); packets = 2 if (L >= 5 and n > 4096) else 1
    return [1.0, m * k * s * packets, k * n * s, m * n * s, 7 ** L * packets, 2 * m * k * n * (7 / 8) ** L]
models = {}; ratios = {}
groups = collections.defaultdict(list)
for r in bench:
    groups[(r["kind"], r["depth"])].append(r)
print("time model fits (kind, L): n, rel rms, coefficients")
for key, rows in sorted(groups.items()):
    X = np.array([feats(r["m"], r["k"], r["n"], r["depth"]) for r in rows]); y = np.array([r["np_med"] for r in rows])
    scale = X.max(axis=0) + 1e-30
    coef, _ = nnls(X / scale, y); coef = coef / scale
    pred = X @ coef; rel = np.sqrt(np.mean(((pred - y) / y) ** 2)) if len(rows) > 1 else 0.0
    models[key] = coef
    ratios[key] = float(np.median([r["bill"] / formula(r["m"], r["k"], r["n"], r["depth"]) for r in rows]))
    print(f"  {key}: n={len(rows)} relrms={rel:.3f} bill/formula={ratios[key]:.4f}")
def t_model(kind, m, k, n, L):
    return float(np.dot(models[(kind, L)], feats(m, k, n, L)))
def bill_model(kind, m, k, n, L):
    return formula(m, k, n, L) * ratios[(kind, L)]
def allowed(kind, m, k, n):
    if kind == "push":
        return [4, 5] + ([6] if m == 1024 and m % 64 == k % 64 == n % 64 == 0 else [])
    if kind == "read":
        return [4, 5]
    return [4, 5] + ([6] if m % 64 == k % 64 == n % 64 == 0 else [])
for rec in prof:
    bill_tot = rec["util"] * 2 ** 41
    items = []
    for key, cnt, s, g, b, ops in rec["products"]:
        kind, m, k, n, dep, path = key
        if kind not in ("push", "read", "sq") or "w5" in path or not isinstance(dep, int) or (kind, dep) not in models:
            continue
        opts = [L for L in allowed(kind, m, k, n) if (kind, L) in models]
        if dep not in opts:
            continue
        items.append((kind, m, k, n, dep, cnt, s, b, opts))
    print(f"\nrank {rec['rank']}: {len(items)} product classes, in-situ seconds {sum(i[6] for i in items):.2f}, bill share {sum(i[7] for i in items)/bill_tot*100:.1f}%")
    for mu in (0.0, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        dt = db = 0.0; moves = collections.Counter()
        for kind, m, k, n, dep, cnt, s, b, opts in items:
            best = min(opts, key=lambda L: bill_model(kind, m, k, n, L) / 1e9 + mu * t_model(kind, m, k, n, L))
            if best != dep:
                dt += cnt * (t_model(kind, m, k, n, best) - t_model(kind, m, k, n, dep))
                db += cnt * (bill_model(kind, m, k, n, best) - bill_model(kind, m, k, n, dep))
                moves[(kind, dep, best)] += cnt
        print(f"  mu {mu:4.2f} Gflop/s: d(bill) {db/bill_tot*100:+6.2f}%  d(time) {dt:+6.2f} s (bench units)  moves {dict(moves)}")
