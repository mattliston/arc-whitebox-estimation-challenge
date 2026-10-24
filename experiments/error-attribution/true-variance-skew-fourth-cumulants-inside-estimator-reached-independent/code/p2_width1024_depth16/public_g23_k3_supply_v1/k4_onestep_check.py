#!/usr/bin/env python3
"""Where is the k4 one-step deficit? From TRUTH inputs at layer l (32M m,v,k3,k4; 8M C, planes), predict the
next-layer per-unit k4 with (a) [D] univariate Edgeworth-4 post k4 pushed (sum W^4 k4post), (b) + [P] the pair
planes K31post/K22post from the engine's fourth_emission_1024 (verbatim), i.e. everything the card/engine carry
at the plane level, and compare the rms ratio to the truth k4(l+1). If (a)+(b) under-scale like the card, the
missing content is the DISTINCT fourth-order born words. Usage: k4_onestep_check.py <rank>"""
import json, sys, os
from math import erf, factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; G19 = HERE.parent / "public_g19_attribution_v1"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(G19))
from k3gen import hermite_cols
from run_attribution import edg2_relu_mean
_src = (G19 / "run_tensor_recursion_v4.py").read_text()
_a = _src.index("def columns("); _b = _src.index("def _big_pair_gemm(")
_ns = {"np": np, "factorial": factorial, "_erf": np.vectorize(erf), "os": os, "sys": sys, "HERE": G19}
exec(compile(_src[_a:_b], "engine_verbatim", "exec"), _ns)
fourth_emission_1024 = _ns["fourth_emission_1024"]
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
T32 = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz"); T8 = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz"); Pl = np.load(G19 / f"mps_planes_rank{rank:04d}.npz")
st = np.load(G19 / f"states_rank{rank:04d}.npz")
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
def univariate_k4post(m, v, k3, k4):
    c1, c2, c3, sd, pdf, cdf = hermite_cols(m, v)
    v_ = np.maximum(v, 1e-30); al = m / sd; ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, 34): He.append(ma * He[-1] - (k - 1) * He[-2])
    raw1 = c1[0]; raw2 = c2[0]; raw3 = c3[0]
    raw4 = (m ** 4 + 6 * m * m * v_ + 3 * v_ * v_) * cdf + m * sd * (m * m + 5 * v_) * pdf
    c4 = [raw4, 4 * sd * raw3, 12 * v_ * raw2, 24 * v_ * sd * raw1, 24 * v_ * v_ * cdf] + [24 * v_ * v_ * He[q - 5] * pdf for q in range(5, 30)]
    c4 = np.stack(c4)
    g3, g4 = k3 / sd ** 3, k4 / sd ** 4
    corr = lambda c: (g3 / 6) * c[3] + (g4 / 24) * c[4] + (g3 * g3 / 72) * c[6]
    r1 = edg2_relu_mean(m, v, k3, k4); r2 = c2[0] + corr(c2); r3 = c3[0] + corr(c3); r4 = c4[0] + corr(c4)
    return r4 - 4 * r3 * r1 - 3 * r2 ** 2 + 12 * r2 * r1 ** 2 - 6 * r1 ** 4
print("L+1 | truth k4 rms | ratio: [D] only | [D]+[P] | card (one-step, truth-forced) | corr [D]+[P]")
out = {}
for l in range(2, 15):
    m, v = T32["mc_pre_m"][l].astype(float), T32["mc_pre_v"][l].astype(float)
    k3, k4 = T32["mc_pre_k3"][l].astype(float), T32["mc_pre_k4"][l].astype(float)
    C = T8["mc_pre_cov"][l].astype(float)
    K21 = Pl[f"K21_{l}"].astype(float).copy(); np.fill_diagonal(K21, k3); K22 = Pl[f"K22_{l}"].astype(float); K31 = Pl[f"K31_{l}"].astype(float)
    Wd = np.asarray(W[l + 1], dtype=float)
    kt = T32["mc_pre_k4"][l + 1].astype(float)
    k4p = univariate_k4post(m, v, k3, k4)
    D = (Wd ** 4).T @ k4p
    k4d, K22p, K31p = fourth_emission_1024(m, v, k3, k4, C, K21, K22, K31)
    P = 4.0 * np.einsum("hi,li,hl->i", Wd ** 3, Wd, K31p, optimize=True) + 3.0 * np.einsum("hi,li,hl->i", Wd ** 2, Wd ** 2, K22p, optimize=True)
    r = lambda x: float(np.sqrt((x ** 2).mean()) / np.sqrt((kt ** 2).mean()))
    kc = np.load(HERE.parent / "public_g20_emission_retrofit_v1" / "k4dump_tfvk_retro-35.npz")[f"k4_{l + 1}"] if rank == 35 and l + 1 <= 14 else None
    row = {"truth_rms": float(np.sqrt((kt ** 2).mean())), "ratio_D": r(D), "ratio_DP": r(D + P), "ratio_card_onestep": (r(kc) if kc is not None else None), "corr_DP": float(np.corrcoef(D + P, kt)[0, 1]), "rel_err_DP": float(np.linalg.norm(D + P - kt) / np.linalg.norm(kt))}
    out[l + 1] = row
    print(l + 1, f"{row['truth_rms']:.3e} | {row['ratio_D']:.3f} | {row['ratio_DP']:.3f} | {row['ratio_card_onestep']} | {row['corr_DP']:.3f} (rel err {row['rel_err_DP']:.3f})", flush=True)
(HERE / f"K4_ONESTEP_rank{rank:04d}.json").write_text(json.dumps(out, indent=1) + "\n")
