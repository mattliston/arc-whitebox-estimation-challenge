#!/usr/bin/env python3
"""Is the card's m31 missing its Edgeworth corrections? Compare the card's raw pair moments (m11, m21, m31, m22) at
layer 5 with (a) the Gaussian-only Mehler series and (b) Gaussian + first-order Edgeworth dM (engine form), each
channel's contribution; report (card - Gaussian)/dM per moment."""
import importlib.util as ilu, json, sys, os
from math import erf, factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; G19 = HERE.parent / "public_g19_attribution_v1"; G20 = HERE.parent / "public_g20_emission_retrofit_v1"
sys.path.insert(0, str(G19)); sys.path.insert(0, str(HERE.parent / "public_g21_pair_consumer_v1"))
import flopscope as flops, flopscope.numpy as fnp
from pairmap import series as pseries, dM as pdM, columns as pcolumns
spec = ilu.spec_from_file_location("retrocard", G20 / "estimator_retro.py"); mod = ilu.module_from_spec(spec); sys.modules["retrocard"] = mod; spec.loader.exec_module(mod)
T32 = np.load(G19 / "mps_truth_rank0035_P32M.npz"); h = np.load(G19 / "host_cov_rank0035_base.npz")
n = 1024; off = ~np.eye(n, dtype=bool)
hermite_context = mod._hermite_context(); fusion_context = mod._fusion_context(n); diagonal_context = fnp.asarray(np.eye(n, dtype=np.float32), dtype=fnp.bool_)
L = int(sys.argv[1]) if len(sys.argv) > 1 else 5
m, v = T32["mc_pre_m"][L].astype(float), T32["mc_pre_v"][L].astype(float); k3, k4 = T32["mc_pre_k3"][L].astype(float), T32["mc_pre_k4"][L].astype(float)
C = h["cov"][L - 1].astype(float); K21 = h["k21"][L - 1].astype(float).copy(); np.fill_diagonal(K21, k3); K22 = h["k22"][L - 1].astype(float); K31 = h["k31"][L - 1].astype(float)
with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=3600.0, quiet=True):
    coef, graw, sigma, alpha, pdf, sigma2, sigma3 = mod._q38_coefficients(fnp.asarray(m), fnp.asarray(v), hermite_context)
    inv = mod._inverse_sigma_context(sigma); ctx = mod._q24_context(coef, sigma, fnp.asarray(C), fusion_context, inv)
    weights = mod._edgeworth_weights(fnp.asarray(k3), fnp.asarray(K21), fnp.asarray(k4), fnp.asarray(K31), fnp.asarray(K22), fusion_context)
    wts = np.array(np.asarray(weights, dtype=np.float64))
    m11, m21, m31, m22 = (np.array(np.asarray(x, dtype=np.float64)) for x in mod._four_pair_edgeworth(ctx, weights, fusion_context[12:14]))
    # Gaussian-only: zero weights except channel 0
    w0 = fnp.asarray(np.concatenate([np.ones((1, n, n)), np.zeros((16, n, n))]), dtype=fnp.float64)
    g11, g21, g31, g22 = (np.array(np.asarray(x, dtype=np.float64)) for x in mod._four_pair_edgeworth(ctx, w0, fusion_context[12:14]))
c1, c2, sd, pdf0, cdf0 = pcolumns(m, v); rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
v_ = np.maximum(v, 1e-30); al = m / sd; ma = -al; He = [np.ones_like(al), ma]
for k in range(2, 34): He.append(ma * He[-1] - (k - 1) * He[-2])
raw1 = c1[0]; raw2 = (m * m + v_) * cdf0 + m * sd * pdf0; raw3 = (m ** 3 + 3 * m * v_) * cdf0 + (m * m + 2 * v_) * sd * pdf0
c3 = np.stack([raw3, 3 * sd * raw2, 6 * v_ * raw1, 6 * v_ * sd * cdf0] + [6 * v_ * sd * He[q - 4] * pdf0 for q in range(4, 30)])
S = lambda A, B, a, b: pseries(A, B, a, b, rho, sd, 20)
G = {"m11": S(c1, c1, 0, 0), "m21": S(c2, c1, 0, 0), "m31": S(c3, c1, 0, 0), "m22": S(c2, c2, 0, 0)}
D = {"m11": pdM(c1, c1, rho, sd, k3, k4, K21, K22, K31), "m21": pdM(c2, c1, rho, sd, k3, k4, K21, K22, K31), "m31": pdM(c3, c1, rho, sd, k3, k4, K21, K22, K31), "m22": pdM(c2, c2, rho, sd, k3, k4, K21, K22, K31)}
card = {"m11": m11, "m21": m21, "m31": m31, "m22": m22}; cg = {"m11": g11, "m21": g21, "m31": g31, "m22": g22}
print(f"L{L}: per moment — |dM|/|Gauss| (engine first-order size) | card's own correction (card − card_Gaussian) / engine dM: ratio, corr | card_Gaussian vs series Gaussian ratio")
for k in card:
    cc = (card[k] - cg[k])[off]; dd = D[k][off]
    print(k, f"{np.linalg.norm(dd)/np.linalg.norm(G[k][off]):.4f} | {np.dot(cc, dd)/np.dot(dd, dd):.3f}, {np.corrcoef(cc, dd)[0,1]:.3f} | {np.linalg.norm(cg[k][off])/np.linalg.norm(G[k][off]):.4f}")
# which channels carry m31's correction in the card: contribution of each weight channel
print("channel weights rms (card, 17 channels):", np.round(np.sqrt((wts ** 2).mean(axis=(1, 2))), 5))
