#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PF3 = fit_platform_v2 with a CLEAN elementwise feature): billed 'other' flops mix data-moving elementwise
work with views that are billed but ~free in time (reshape, getitem...). Features per file (warm r100/r37 means, platfit/optime_*.json):
G = matmul+einsum+dot+tensordot flops; X = billed flops of data-moving non-GEMM ops (everything except VIEW ops); C = calls.
Models through the origin: (B') K = a*G + b*X; (C') K = a*G + b*X + c*C. Fitted on S2..S6b, must predict S7; LOO over all five;
then predicts candidates from the fit on all five. usage: fit_platform_v3.py [candidate_tags...]"""
import json, sys, statistics as st
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; E = H.parent; F = E.parent / "fable_fire_v1"
RUNS = [("S2", F / "eval_331947.json"), ("S4", F / "eval_331961.json"), ("S5e", F / "eval_331992.json"), ("S6b", F / "eval_332005.json"),
        ("S7", E / "telemetry" / "eval_332026.json")]
GEMM = {"matmul", "dot", "tensordot", "einsum", "inner", "outer", "vdot"}
VIEW = {"reshape", "getitem", "transpose", "swapaxes", "moveaxis", "expand_dims", "squeeze", "unstack", "split", "ravel", "view", "broadcast_to", "diagonal", "T"}
def feats(tag):
    d = json.load(open(H / f"optime_{tag}.json")); warm = [r for r in d if r["rank"] in (100, 37)]
    g = st.mean(sum(v[2] for k, v in r["ops"].items() if k in GEMM) for r in warm)
    x = st.mean(sum(v[2] for k, v in r["ops"].items() if k not in GEMM and k not in VIEW) for r in warm)
    c = st.mean(sum(v[0] for v in r["ops"].values()) for r in warm)
    return np.array([g, x, c])
tele = {tag: {m["mlp_index"]: m["telemetry"]["flopscope_server_kernel_wall_s"] for m in json.load(open(p))["evaluation"]["results"]["per_mlp"] if m.get("telemetry")} for tag, p in RUNS}
common = sorted(set.intersection(*(set(v) for v in tele.values())) - set(range(5)))
X = np.array([feats(t) for t, _ in RUNS]); K = np.array([st.median(tele[t][i] for i in common) for t, _ in RUNS]); names = [t for t, _ in RUNS]
print(f"platform kernel median over {len(common)} common indices; features warm r100+r37")
for n, f, k in zip(names, X, K):
    print(f"  {n:4s} GEMM {f[0]/1e9:6.1f}  elementwise(data-moving) {f[1]/1e9:6.2f} Gflop  calls {f[2]:6.0f}  | kernel {k:6.2f} s")
for lab, cols in (("B' GEMM+elementwise", [0, 1]), ("C' GEMM+elementwise+calls", [0, 1, 2])):
    A = X[:, cols]; coef = np.linalg.lstsq(A[:4], K[:4], rcond=None)[0]; p7 = float(A[4] @ coef)
    loo = []
    for j in range(5):
        idx = [i for i in range(5) if i != j]; cj = np.linalg.lstsq(A[idx], K[idx], rcond=None)[0]; loo.append(100 * (float(A[j] @ cj) / K[j] - 1))
    call = np.linalg.lstsq(A, K, rcond=None)[0]
    share = [call[i] * A[4, i] / K[4] for i in range(len(cols))]
    print(f"{lab:26s} fit S2..S6b -> S7 {p7:6.2f} vs {K[4]:6.2f} ({100*(p7/K[4]-1):+.2f}%); LOO " + " ".join(f"{n}:{e:+.1f}%" for n, e in zip(names, loo))
          + f"; all-five coef {', '.join(f'{c:.3g}' for c in call)}; S7 kernel shares {', '.join(f'{100*s:.0f}%' for s in share)}")
    for tag in sys.argv[1:]:
        a = feats(tag)[cols]; p = float(a @ call)
        print(f"    PREDICT {tag}: {p:6.2f} s ({100*(p/K[4]-1):+.2f}% vs S7's measured kernel)")
