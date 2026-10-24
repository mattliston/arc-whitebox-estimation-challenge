#!/usr/bin/env python3
"""TAG fable-wall-repair (W-R): structure of the card's pre-activation third-order plane error at chosen layers. Records the card's K21
(pre plane) and Cpre through the head hook in a normal closed-loop run, compares with the MPS 8M truth planes: scale (regression slope of
truth on card), off-diagonal relative error before/after the best scalar rescale, SVD energy of the error in its top-k singular directions,
and the share of the error explained by a rank-1 (a_i b_j) fit and by the covariance-shaped ansatz E ~ (u_i C_ij + C_ij w_j).
usage: plane_error_structure.py <cand> <rank> <planes.npz> <layers>"""
import sys, json, types, importlib.util
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, planes = Path(sys.argv[1]).resolve(), int(sys.argv[2]), np.load(sys.argv[3]); LAYERS = [int(x) for x in sys.argv[4].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("pes_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
rec = {}
orig_pb = mod._predict_binary
def pb(mlp_, hook=None, **kw):
    def h(l, m, C, K21, aux, w):
        if l in LAYERS: rec[l] = dict(K21=np.asarray(K21, dtype=np.float64), Cpre=np.asarray(aux["pre_cov"], dtype=np.float64), s=np.asarray(aux["sigma"], dtype=np.float64), g=np.asarray(aux["gate"], dtype=np.float64), pdf=np.asarray(aux["pdf"], dtype=np.float64), k3=np.asarray(aux["pre_k3"], dtype=np.float64))
        return hook(l, m, C, K21, aux, w)
    return orig_pb(mlp_, hook=h, **kw)
mod._predict_binary = pb
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().predict(MLP([W[l] for l in range(16)], f"pes-{rank}"), 2**41)
off = ~np.eye(1024, dtype=bool)
for l in LAYERS:
    K = rec[l]["K21"]; T = planes[f"K21_{l}"].astype(np.float64); C = rec[l]["Cpre"]; s = rec[l]["s"]
    e = (K - T)[off]; t = T[off]; k = K[off]
    slope = float(np.dot(k, t) / np.dot(k, k)); rel = np.sqrt(np.sum(e ** 2) / np.sum(t ** 2)); rel_scaled = np.sqrt(np.sum((slope * k - t) ** 2) / np.sum(t ** 2))
    corr = float(np.dot(k, t) / np.sqrt(np.dot(k, k) * np.dot(t, t)))
    E = (K - T).copy(); np.fill_diagonal(E, 0.0)
    U, S, Vt = np.linalg.svd(E, full_matrices=False); en = np.cumsum(S ** 2) / np.sum(S ** 2)
    Tt = T.copy(); np.fill_diagonal(Tt, 0.0); Ut, St, _ = np.linalg.svd(Tt, full_matrices=False); ent = np.cumsum(St ** 2) / np.sum(St ** 2)
    # normalised planes (by sigma^2 sigma): where is the error concentrated?
    Kn = K / (s[:, None] ** 2 * s[None, :]); Tn = T / (s[:, None] ** 2 * s[None, :]); En = (Kn - Tn); np.fill_diagonal(En, 0)
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0)
    # covariance-shaped ansatz: E_n ~ (a_i + b_j) rho_ij  (weighted least squares in a, b via 2048x2048 normal equations)
    R2 = rho ** 2; tgt = En * rho   # sum_ij (E - (a_i+b_j) rho)^2 -> normal eqs: diag(rowsum R2) a + R2 b = rowsum(E rho); R2^T a + diag(colsum R2) b = colsum(E rho)
    Mx = np.block([[np.diag(R2.sum(1)), R2], [R2.T, np.diag(R2.sum(0))]]); rhs = np.concatenate([tgt.sum(1), tgt.sum(0)])
    ab = np.linalg.lstsq(Mx, rhs, rcond=None)[0]; a, b = ab[:1024], ab[1024:]; fit = (a[:, None] + b[None, :]) * rho; resid = En - fit; np.fill_diagonal(resid, 0)
    print(f"L{l}: K21 off-diag: corr(card,truth) {corr:.3f} | slope truth/card {slope:.3f} | rel err {rel:.3f} -> after scalar rescale {rel_scaled:.3f} | error SVD energy top1/8/32/128: {en[0]:.2f}/{en[7]:.2f}/{en[31]:.2f}/{en[127]:.2f} (truth plane: {ent[0]:.2f}/{ent[7]:.2f}/{ent[31]:.2f}/{ent[127]:.2f}) | rho-ansatz explains {100*(1-np.sum(resid[off]**2)/np.sum(En[off]**2)):.1f}% of the normalised error")
