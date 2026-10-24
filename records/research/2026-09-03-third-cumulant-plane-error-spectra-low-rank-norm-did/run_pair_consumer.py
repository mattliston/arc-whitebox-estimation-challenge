#!/usr/bin/env python3
"""G21 pair CONSUMER chain (replaces the tangent's linearized injection/transport).
Per layer l: card base state (host m, v, C, k3, k4, planes); SUPPLY = truth K21 plane
(PAIRS=1: + truth K22/K31; UV=oracle: + truth k3/k4).
modes:  lin   = the tangent's first-order injection + linear D transport (parity check)
        edge  = nonlinear injection: postcov(C; truth cum) - postcov(C; card cum); linear D transport
        exact = full nonlinear propagation: core = postcov(C+eps; truth cum) - postcov(C; card cum)
Saves dvhat_rank{rank}_pc_{mode}{tag}.npz into public_g19_attribution_v1 (consumed by
consumption_betas.py with DV_SUFFIX). Usage: run_pair_consumer.py <rank> <mode>"""
import json, os, sys, time
from math import factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "public_g19_attribution_v1"))
from pairmap import post_cov, G19
from run_tangent_v2 import c1_std_factors
rank = int(sys.argv[1]); mode = sys.argv[2]
PAIRS = os.environ.get("PAIRS") == "1"; UV = os.environ.get("UV", "off")
ORDER = int(os.environ.get("ORDER", "2"))
tag = ("_pairs" if PAIRS else "") + ("_uv" if UV == "oracle" else "") + (f"_o{ORDER}" if ORDER != 2 else "")
if mode == "truthC":
    tag += "_w" + os.environ.get("CWIN", "2-14").replace("-", "")
st = np.load(G19 / f"states_rank{rank:04d}.npz"); z = np.load(G19 / f"host_cov_rank{rank:04d}_base.npz")
T = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz"); Pl = np.load(G19 / f"mps_planes_rank{rank:04d}.npz")
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
eps = np.zeros((1024, 1024)); DV = {}; rows = {}
t0 = time.time()
for l in range(1, 15):
    i = l - 1
    m, v = st["host_m"][i].astype(float), st["host_v"][i].astype(float)
    k3, k4 = st["host_k3"][i].astype(float), st["host_k4"][i].astype(float)
    C = z["cov"][i].astype(float)
    K21 = z["k21"][i].astype(float).copy(); np.fill_diagonal(K21, k3)
    K22 = z["k22"][i].astype(float); K31 = z["k31"][i].astype(float)
    have = l in Pl["layers"]
    tK21 = Pl[f"K21_{l}"].astype(float).copy() if have else K21.copy()
    if os.environ.get("SUPPLY_K21"):   # G29: an alternative K21 supply file (keys K21_l), e.g. rank-limited oracle corrections
        _sup = np.load(os.environ["SUPPLY_K21"])
        if f"K21_{l}" in _sup.files:
            tK21 = _sup[f"K21_{l}"].astype(float).copy(); have = True
    sk3, sk4 = (T["mc_pre_k3"][l].astype(float), T["mc_pre_k4"][l].astype(float)) if UV == "oracle" else (k3, k4)
    np.fill_diagonal(tK21, sk3)
    tK22, tK31 = (Pl[f"K22_{l}"].astype(float), Pl[f"K31_{l}"].astype(float)) if (PAIRS and have) else (K22, K31)
    if os.environ.get("SUPPLY_K22K31"):   # G29: deployable fourth-order planes inside the consumer (keys K22_l / K31_l; missing key -> the card's own)
        _s4 = np.load(os.environ["SUPPLY_K22K31"])
        tK22 = _s4[f"K22_{l}"].astype(float) if f"K22_{l}" in _s4.files else K22
        tK31 = _s4[f"K31_{l}"].astype(float) if f"K31_{l}" in _s4.files else K31
    if os.environ.get("SUPPLY_K22K31"):   # G29: alternative fourth-order plane supply (keys K22_l / K31_l; missing key -> the card's own)
        _s4 = np.load(os.environ["SUPPLY_K22K31"])
        tK22 = _s4[f"K22_{l}"].astype(float) if f"K22_{l}" in _s4.files else K22
        tK31 = _s4[f"K31_{l}"].astype(float) if f"K31_{l}" in _s4.files else K31
    if mode == "lin":
        c1, e, sd = c1_std_factors(m, v)
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
        D = np.zeros_like(rho); rp = np.ones_like(rho)
        for t in range(1, 9):
            if t > 1: rp = rp * rho
            D += t * rp / factorial(t) * np.outer(c1[t], c1[t])
        D = D / np.outer(sd, sd)
        def Gker(a, b, tmax=8):
            acc = np.zeros_like(rho); rp2 = np.ones_like(rho)
            for t in range(tmax + 1):
                if t > 0: rp2 = rp2 * rho
                acc += rp2 / factorial(t) * np.outer(c1[a + t] / sd ** a, c1[b + t] / sd ** b)
            return acc
        dK21 = tK21 - K21
        inj = (dK21 / 2.0) * Gker(2, 1) + (dK21.T / 2.0) * Gker(1, 2) if have else np.zeros_like(rho)
        if PAIRS and have:
            dK22 = tK22 - K22; dK31 = tK31 - K31; np.fill_diagonal(dK22, 0.0); np.fill_diagonal(dK31, 0.0)
            inj = inj + (dK22 / 4.0) * Gker(2, 2) + (dK31 / 6.0) * Gker(3, 1) + (dK31.T / 6.0) * Gker(1, 3)
        core = D * eps + inj
    elif mode == "edge":
        c1, e, sd = c1_std_factors(m, v)
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
        D = np.zeros_like(rho); rp = np.ones_like(rho)
        for t in range(1, 9):
            if t > 1: rp = rp * rho
            D += t * rp / factorial(t) * np.outer(c1[t], c1[t])
        D = D / np.outer(sd, sd)
        inj = post_cov(m, v, C, sk3, sk4, tK21, tK22, tK31, ORDER) - post_cov(m, v, C, k3, k4, K21, K22, K31, ORDER)
        core = D * eps + inj
    elif mode == "exact":
        Ce = C + eps
        ve = np.maximum(np.diag(Ce), 1e-30)
        core = post_cov(m, ve, Ce, sk3, sk4, tK21, tK22, tK31, ORDER) - post_cov(m, v, C, k3, k4, K21, K22, K31, ORDER)
    elif mode == "truthC":
        # covariance channel: replace the card's pre-covariance by the TRUTH pre-covariance
        # (off-diagonal only; v, m, cumulants stay card) at layers in CWIN, no accumulation
        lo, hi = [int(x) for x in os.environ.get("CWIN", "2-14").split("-")]
        if lo <= l <= hi:
            tC = T["mc_pre_cov"][l].astype(float).copy(); np.fill_diagonal(tC, v)
            core = post_cov(m, v, tC, k3, k4, K21, K22, K31, ORDER) - post_cov(m, v, C, k3, k4, K21, K22, K31, ORDER)
        else:
            core = np.zeros((1024, 1024))
    else:
        raise SystemExit("mode?")
    Wd = np.asarray(W[l + 1], dtype=float)
    eps = Wd.T @ core @ Wd
    dv_hat = np.diag(eps).copy()
    dv_true = T["mc_pre_v"][l + 1].astype(float) - st["host_v"][l].astype(float)
    cc = float(np.corrcoef(dv_hat, dv_true)[0, 1]); beta = float(np.dot(dv_hat, dv_true) / (np.dot(dv_hat, dv_hat) + 1e-300))
    DV[l + 1] = dv_hat
    rows[l + 1] = {"dv_corr": round(cc, 4), "dv_beta": round(beta, 4), "dv_rms": float(np.sqrt((dv_hat ** 2).mean())), "dv_true_rms": float(np.sqrt((dv_true ** 2).mean()))}
    print(l + 1, json.dumps(rows[l + 1]), round(time.time() - t0, 1), "s", flush=True)
sfx = f"_pc_{mode}{tag}"
sfx = sfx + os.environ.get("OUT_TAG", "")
np.savez(G19 / f"dvhat_rank{rank:04d}{sfx}.npz", **{f"L{k}": v for k, v in DV.items()})
(HERE / f"PAIR_CONSUMER_rank{rank:04d}{sfx}.json").write_text(json.dumps(rows, indent=1) + "\n")
print("saved", sfx)
