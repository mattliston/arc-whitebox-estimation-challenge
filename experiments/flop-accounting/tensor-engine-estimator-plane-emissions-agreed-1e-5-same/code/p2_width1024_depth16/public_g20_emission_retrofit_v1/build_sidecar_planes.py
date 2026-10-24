#!/usr/bin/env python3
"""TAG g20-sidecar: direct post-hoc sidecar supply, NO carry.
Per layer L (pre state = card's own dumped host state + dense host planes):
  E_L = emission_engine(card state) - emission_card(card state)
      = run_tensor_recursion_v4.emission_K21(m,v,k3,k4,C,K21,K22,K31)
        - estimator_retro._full_relu_reset(...)[2]  (the card's Q24 post-K21 target)
  pushed ONE layer as a plane-slice delta (the certified _push_cubic form, three slice
  orientations, no distinct-entry carry) -> Delta pre_{L+1}.
Writes public_g19_attribution_v1/tensor_planes_rank{rank}_sidecar.npz with
pre_{L+1} = host_k21[L] + Delta_{L+1} (pre_1 = host_k21[0]); prints diagnostics vs truth.
Usage: (venv) build_sidecar_planes.py <rank>"""
import importlib.util as ilu, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3")
rank = int(sys.argv[1])
sys.path.insert(0, str(G19))
# the engine module imports torch at top level (absent in the card venv): exec the
# two pure-numpy functions verbatim from the engine source (no transcription)
import os
from math import erf, factorial
_src = (G19 / "run_tensor_recursion_v4.py").read_text()
_a = _src.index("def columns("); _b = _src.index("def fourth_emission_1024(")
_ns = {"np": np, "factorial": factorial, "_erf": np.vectorize(erf), "os": os, "sys": sys, "HERE": G19}
exec(compile(_src[_a:_b], "engine_emission_verbatim", "exec"), _ns)
emission_K21 = _ns["emission_K21"]
import flopscope as flops
import flopscope.numpy as fnp
spec = ilu.spec_from_file_location("retrocard", HERE / "estimator_retro.py")
mod = ilu.module_from_spec(spec); sys.modules["retrocard"] = mod; spec.loader.exec_module(mod)
manifest = json.loads((STORE / "manifest.json").read_text())
entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
st = np.load(G19 / f"states_rank{rank:04d}.npz")
h = np.load(G19 / f"host_cov_rank{rank:04d}_base.npz")
tpp = G19 / f"mps_planes_rank{rank:04d}.npz"
tp = np.load(tpp) if tpp.exists() else None
n = 1024
D = np.eye(n, dtype=bool)
hermite_context = mod._hermite_context()
fusion_context = mod._fusion_context(n)
diagonal_context = fnp.asarray(np.eye(n, dtype=np.float32), dtype=fnp.bool_)
out = {"pre_1": h["k21"][0].astype(np.float32)}
diag = {}
def push_slices(E, Wl):
    """certified plane-slice push (estimator_cap._push_cubic form), diag(E)=0"""
    W2 = Wl * Wl
    tr = E @ Wl
    K = 2.0 * (Wl * tr).T @ Wl + W2.T @ tr
    np.fill_diagonal(K, np.sum(3.0 * W2 * tr, axis=0))
    return K
t0 = time.time()
with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=3600.0, quiet=True):
    for L in range(1, 14):
        m = st["host_m"][L - 1].astype(np.float64); v = st["host_v"][L - 1].astype(np.float64)
        k3 = st["host_k3"][L - 1].astype(np.float64); k4 = st["host_k4"][L - 1].astype(np.float64)
        C = h["cov"][L - 1].astype(np.float64); K21 = h["k21"][L - 1].astype(np.float64).copy()
        K22 = h["k22"][L - 1].astype(np.float64); K31 = h["k31"][L - 1].astype(np.float64)
        if np.abs(np.diag(K21)).max() < 1e-12:
            np.fill_diagonal(K21, k3)
        # engine emission on the card's state
        P_eng = emission_K21(m, v, k3, k4, C, K21, K22, K31)
        # card's own post target on the same inputs
        coef, graw, sigma, alpha, pdf, sigma2, sigma3 = mod._q38_coefficients(
            fnp.asarray(m), fnp.asarray(v), hermite_context)
        res = mod._full_relu_reset(fnp.asarray(m), fnp.asarray(C), fnp.asarray(K21),
                                   fnp.asarray(k4), fnp.asarray(K31), fnp.asarray(K22),
                                   coef, graw, sigma, sigma2, sigma3, alpha, pdf,
                                   diagonal_context, fusion_context)
        P_card = np.array(np.asarray(res[2], dtype=np.float64))
        E = np.asarray(P_eng, dtype=np.float64) - P_card
        np.fill_diagonal(E, 0.0)
        Wl = np.asarray(W[L + 1], dtype=np.float64)
        dpre = push_slices(E, Wl)
        out[f"pre_{L + 1}"] = (h["k21"][L].astype(np.float64) + dpre).astype(np.float32)
        row = {"E_off_rms": float(np.sqrt((E[~D] ** 2).mean())),
               "Pcard_off_rms": float(np.sqrt((P_card[~D] ** 2).mean())),
               "dpre_off_rms": float(np.sqrt((dpre[~D] ** 2).mean()))}
        if tp is not None and f"K21_{L + 1}" in tp.files:
            gap = tp[f"K21_{L + 1}"].astype(np.float64) - h["k21"][L].astype(np.float64)
            row["gap_off_rms"] = float(np.sqrt((gap[~D] ** 2).mean()))
            row["corr_dpre_gap"] = float(np.corrcoef(dpre[~D], gap[~D])[0, 1])
            row["beta_dpre_gap"] = float(np.dot(dpre[~D], gap[~D]) / np.dot(dpre[~D], dpre[~D]))
        diag[L + 1] = row
        print(L + 1, json.dumps({k: round(x, 5) for k, x in row.items()}), round(time.time() - t0, 1), "s", flush=True)
np.savez(G19 / f"tensor_planes_rank{rank:04d}_sidecar.npz", **out)
(HERE / f"SIDECAR_DIAG_rank{rank:04d}.json").write_text(json.dumps(diag, indent=1) + "\n")
print("wrote", G19 / f"tensor_planes_rank{rank:04d}_sidecar.npz")
