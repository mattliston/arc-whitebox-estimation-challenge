#!/usr/bin/env python3
"""Where is the card's K31 post target short? Layer-1 inputs (Gaussian pre: k3=k4=0, planes 0): run the card's
own _full_relu_reset (retro copy, flopscope) and compare its k31/k22 (and the raw pair moments m31/m21/m11/m22)
against the Mehler-series reference (engine's fourth_emission machinery at tmax 6 and 16) on the same inputs."""
import importlib.util as ilu, json, sys, os
from math import erf, factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; G19 = HERE.parent / "public_g19_attribution_v1"; G20 = HERE.parent / "public_g20_emission_retrofit_v1"
sys.path.insert(0, str(G19))
import flopscope as flops, flopscope.numpy as fnp
spec = ilu.spec_from_file_location("retrocard", G20 / "estimator_retro.py"); mod = ilu.module_from_spec(spec); sys.modules["retrocard"] = mod; spec.loader.exec_module(mod)
_src = (G19 / "run_tensor_recursion_v4_g24fix.py").read_text(); _a = _src.index("def columns("); _b = _src.index("def _big_pair_gemm(")
src = _src[_a:_b].replace("def series(colsA, colsB, a, b, tmax=16):", "def series(colsA, colsB, a, b, tmax=TMAX):")
def make(tmax):
    ns = {"np": np, "factorial": factorial, "_erf": np.vectorize(erf), "os": os, "sys": sys, "HERE": G19, "TMAX": tmax}; exec(compile(src, "eng", "exec"), ns); return ns
e6, e16 = make(6), make(16)
T32 = np.load(G19 / f"mps_truth_rank0035_P32M.npz"); h = np.load(G19 / "host_cov_rank0035_base.npz")
n = 1024; off = ~np.eye(n, dtype=bool)
hermite_context = mod._hermite_context(); fusion_context = mod._fusion_context(n); diagonal_context = fnp.asarray(np.eye(n, dtype=np.float32), dtype=fnp.bool_)
for L in [1, 5]:
    m, v = T32["mc_pre_m"][L].astype(float), T32["mc_pre_v"][L].astype(float)
    k3, k4 = (T32["mc_pre_k3"][L].astype(float), T32["mc_pre_k4"][L].astype(float)) if L > 1 else (np.zeros(n), np.zeros(n))
    C = h["cov"][L - 1].astype(float)
    K21 = (h["k21"][L - 1].astype(float).copy() if L > 1 else np.zeros((n, n))); np.fill_diagonal(K21, k3)
    K22 = h["k22"][L - 1].astype(float) if L > 1 else np.zeros((n, n)); K31 = h["k31"][L - 1].astype(float) if L > 1 else np.zeros((n, n))
    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=3600.0, quiet=True):
        coef, graw, sigma, alpha, pdf, sigma2, sigma3 = mod._q38_coefficients(fnp.asarray(m), fnp.asarray(v), hermite_context)
        res = mod._full_relu_reset(fnp.asarray(m), fnp.asarray(C), fnp.asarray(K21), fnp.asarray(k4), fnp.asarray(K31), fnp.asarray(K22), coef, graw, sigma, sigma2, sigma3, alpha, pdf, diagonal_context, fusion_context)
        card_k21, card_k4, card_k31, card_k22 = (np.array(np.asarray(res[i], dtype=np.float64)) for i in (2, 3, 4, 5))
        # raw pair moments as the card computes them (repeat the internal calls)
        inv = mod._inverse_sigma_context(sigma); ctx = mod._q24_context(coef, sigma, fnp.asarray(C), fusion_context, inv)
        weights = mod._edgeworth_weights(fnp.asarray(k3), fnp.asarray(K21), fnp.asarray(k4), fnp.asarray(K31), fnp.asarray(K22), fusion_context)
        m11, m21, m31, m22 = (np.array(np.asarray(x, dtype=np.float64)) for x in mod._four_pair_edgeworth(ctx, weights, fusion_context[12:14]))
    for lab, e in (("tmax6", e6), ("tmax16", e16)):
        c1, c2, sd, pdf0, cdf0 = e["columns"](m, v); rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
        # engine-style raw moments (Gaussian part + first-order channels via fourth_emission internals): recompute M31/M22/M21/M11
        k4e, K22e, K31e = e["fourth_emission_1024"](m, v, k3, k4, C, K21, K22, K31)
        r = lambda a, b: (float(np.sqrt((a[off] ** 2).mean()) / np.sqrt((b[off] ** 2).mean())), float(np.corrcoef(a[off], b[off])[0, 1]))
        print(f"L{L} [{lab}] card k31 / engine K31post: ratio {r(card_k31, K31e)[0]:.3f} corr {r(card_k31, K31e)[1]:.3f} | k22: ratio {r(card_k22, K22e)[0]:.3f} corr {r(card_k22, K22e)[1]:.3f} | k4 diag ratio {np.sqrt((card_k4**2).mean())/np.sqrt((k4e**2).mean()):.3f}")
    # Gaussian-series raw moments for comparison with the card's m31 (tmax 16)
    c1, c2, sd, pdf0, cdf0 = e16["columns"](m, v); rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
    v_ = np.maximum(v, 1e-30); al = m / sd; ma = -al; He = [np.ones_like(al), ma]
    for k in range(2, 32): He.append(ma * He[-1] - (k - 1) * He[-2])
    raw1 = c1[0]; raw2 = (m * m + v_) * cdf0 + m * sd * pdf0; raw3 = (m ** 3 + 3 * m * v_) * cdf0 + (m * m + 2 * v_) * sd * pdf0
    c3 = np.stack([raw3, 3 * sd * raw2, 6 * v_ * raw1, 6 * v_ * sd * cdf0] + [6 * v_ * sd * He[q - 4] * pdf0 for q in range(4, 30)])
    def ser(A, B, tmax):
        acc = np.zeros_like(rho); rp = np.ones_like(rho)
        for t in range(tmax + 1):
            if t > 0: rp = rp * rho
            acc += rp / factorial(t) * np.outer(A[t], B[t])
        return acc
    for tm in (6, 16):
        M31g = ser(c3, c1, tm); M11g = ser(c1, c1, tm)
        print(f"L{L} card m31 vs Gaussian-series m31 (tmax {tm}): ratio {r(m31, M31g)[0]:.4f} corr {r(m31, M31g)[1]:.4f} | m11: ratio {r(m11, M11g)[0]:.4f}")
