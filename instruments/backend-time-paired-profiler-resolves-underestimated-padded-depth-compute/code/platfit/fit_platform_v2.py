#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-PF2 = fit_platform.py on DETERMINISTIC features; local op TIMES under load vary +-30% between runs, so
they cannot resolve a 3-point effect). Per fired run: platform kernel = median over MLP indices common to all runs (indices 5+);
local features from the in-process op census (platfit/optime_<tag>.json, warm nets r100 and r37): G = GEMM billed flops
(matmul/einsum/dot/tensordot), N = all other billed flops, C = op calls. Models through the origin: (A) K = s*(G+N), the lead's
ns-per-kflop model; (B) K = a*G + b*N; (C) K = a*G + b*N + c*C. Each model is fitted on the runs BEFORE S7 and must predict S7;
leave-one-out errors over all five runs are printed too. usage: fit_platform_v2.py [candidate_tags...]"""
import json, sys, statistics as st
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; E = H.parent; F = E.parent / "fable_fire_v1"
RUNS = [("S2", F / "eval_331947.json"), ("S4", F / "eval_331961.json"), ("S5e", F / "eval_331992.json"), ("S6b", F / "eval_332005.json"),
        ("S7", E / "telemetry" / "eval_332026.json")]
GEMM = {"matmul", "dot", "tensordot", "einsum", "inner", "outer", "vdot"}
def feats(tag):
    d = json.load(open(H / f"optime_{tag}.json")); warm = [r for r in d if r["rank"] in (100, 37)]
    g = st.mean(sum(v[2] for k, v in r["ops"].items() if k in GEMM) for r in warm)
    n = st.mean(sum(v[2] for k, v in r["ops"].items() if k not in GEMM) for r in warm)
    c = st.mean(sum(v[0] for v in r["ops"].values()) for r in warm)
    t = st.mean(sum(v[1] for v in r["ops"].values()) for r in warm)
    tg = st.mean(sum(v[1] for k, v in r["ops"].items() if k in GEMM) for r in warm)
    return np.array([g, n, c]), t, tg
tele = {tag: {m["mlp_index"]: (m["telemetry"]["flopscope_server_kernel_wall_s"], m["telemetry"]["flops_used"])
              for m in json.load(open(p))["evaluation"]["results"]["per_mlp"] if m.get("telemetry")} for tag, p in RUNS}
common = sorted(set.intersection(*(set(v) for v in tele.values())) - set(range(5)))
print(f"platform kernel: median over {len(common)} common MLP indices (5..{max(common)}); local features: warm r100 + r37 means")
X, K, names = [], [], []
for tag, _ in RUNS:
    f, t, tg = feats(tag); k = st.median(tele[tag][i][0] for i in common); nsk = st.median(1e12 * tele[tag][i][0] / tele[tag][i][1] for i in common)
    X.append(f); K.append(k); names.append(tag)
    print(f"{tag:4s} GEMM {f[0]/1e9:7.1f} Gflop  other {f[1]/1e9:7.1f} Gflop  calls {f[2]:7.0f}  local backend {t:5.1f} s (GEMM {100*tg/t:4.1f}%)  "
          f"| platform kernel {k:6.2f} s, {nsk:5.1f} ns/kflop")
X = np.array(X); K = np.array(K)
MODELS = {"A one ratio": lambda Z: (Z[:, 0] + Z[:, 1])[:, None], "B GEMM+other": lambda Z: Z[:, :2], "C GEMM+other+calls": lambda Z: Z}
for lab, f in MODELS.items():
    A = f(X); coef = np.linalg.lstsq(A[:4], K[:4], rcond=None)[0]; p7 = float(A[4] @ coef)
    loo = []
    for j in range(5):
        idx = [i for i in range(5) if i != j]; cj = np.linalg.lstsq(A[idx], K[idx], rcond=None)[0]; loo.append(100 * (float(A[j] @ cj) / K[j] - 1))
    print(f"{lab:20s} fit on S2..S6b predicts S7 {p7:6.2f} s vs actual {K[4]:6.2f} ({100*(p7/K[4]-1):+.2f}%); LOO errors " + " ".join(f"{n}:{e:+.1f}%" for n, e in zip(names, loo)))
    for tag in sys.argv[1:]:
        fc, _, _ = feats(tag); a = f(fc[None, :])[0]; c5 = np.linalg.lstsq(A, K, rcond=None)[0]
        print(f"    PREDICT {tag}: {float(a @ c5):6.2f} s ({100*(float(a @ c5)/K[4]-1):+.2f}% vs S7's measured kernel)")
