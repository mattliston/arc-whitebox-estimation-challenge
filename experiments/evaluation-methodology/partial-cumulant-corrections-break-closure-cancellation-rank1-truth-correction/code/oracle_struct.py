#!/usr/bin/env python3
"""TAG fable-wall-repair (W-S): STRUCTURED corrections of the card's late-layer supplies, each testable for attainability:
  k21r<k>   : card K21 + the top-k SVD components of (truth - card) off-diagonal plane  (oracle for the low-rank part of the plane error)
  k21scale  : card K21 off-diagonal scaled by the truth/card regression slope (oracle scalar)
  k4ridge   : per-unit k4 := card k4 + ridge prediction of the k4 error from PRE features (fit on other nets' rows + 8M truths; no oracle)
  modes may be combined with '+'.  usage: oracle_struct.py <cand> <rank> <planes.npz> <layers> <mode> [fit_ranks e.g. 100,150,200]"""
import sys, json, types, importlib.util, math
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
G19 = P2 / "public_g19_attribution_v1"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2]); planes = np.load(sys.argv[3]) if sys.argv[3] != "-" else None; LAYERS = {int(x) for x in sys.argv[4].split(",")}; MODES = sys.argv[5].split("+")
FIT = [int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "100,150,200").split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("os_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
COLS = [3, 4, 5, 6, 10, 12, 14]   # alpha, log s - mean, gate, pdf, k3/(v s), k4/v^2, sqrt(mean rho^2): all PRE quantities
def expand(F):
    n = F.shape[1]; quad = np.stack([F[:, i] * F[:, j] for i in range(n) for j in range(i, n)], 1); return np.concatenate([F, quad, np.ones((len(F), 1))], 1)
ridge = None
if any(m == "k4ridge" for m in MODES):
    Xs = []; Ys = []
    for r in FIT:
        rows = np.load(HERE / "rows_c2" / f"rows_rank{r:04d}.npz"); tr = np.load(G19 / f"mps_truth_rank{r:04d}_P8M.npz"); x, sig = rows["x"].astype(np.float64), rows["sigma"]
        for l in range(4, 16):
            v = sig[l] ** 2; k4c = x[l, :, 12] * v * v
            Xs.append(np.concatenate([x[l][:, COLS], np.full((1024, 1), l / 15.0)], 1)); Ys.append((tr["mc_pre_k4"][l] - k4c) / (v * v))
    Xs = np.concatenate(Xs); Ys = np.concatenate(Ys); mu_f, sd_f = Xs.mean(0), Xs.std(0) + 1e-9; A = expand((Xs - mu_f) / sd_f)
    w = np.linalg.solve(A.T @ A + 1e-2 * len(A) * np.eye(A.shape[1]), A.T @ Ys); ridge = (mu_f, sd_f, w)
    print(f"k4 ridge fitted on nets {FIT}: in-sample R2 {1 - np.mean((A @ w - Ys) ** 2) / np.var(Ys):.3f}")
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mlp = MLP([W[l] for l in range(16)], f"os-{rank}")
layer = [0]; notes = []
def wrap(fn):
    def w_(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = layer[0]; layer[0] += 1
        if l in LAYERS:
            K = np.asarray(K21, dtype=np.float64).copy(); k4n = np.asarray(k4, dtype=np.float64).copy(); k22 = np.asarray(k22_override, dtype=np.float64).copy()
            Cn = np.asarray(C, dtype=np.float64); v = np.maximum(np.diag(Cn), 1e-10); s = np.sqrt(v); mu_ = np.asarray(mu, dtype=np.float64)
            for m in MODES:
                if m.startswith("k21r") or m == "k21scale":
                    T = planes[f"K21_{l}"].astype(np.float64); E = T - K; np.fill_diagonal(E, 0.0)
                    if m == "k21scale":
                        off = ~np.eye(1024, dtype=bool); kk = K[off]; slope = float(np.dot(kk, T[off]) / np.dot(kk, kk)); Kd = np.diag(K).copy(); K = K * slope; np.fill_diagonal(K, Kd); notes.append(f"L{l} slope {slope:.3f}")
                    else:
                        kk = int(m[4:]); U, S, Vt = np.linalg.svd(E, full_matrices=False); corr = (U[:, :kk] * S[:kk]) @ Vt[:kk]; np.fill_diagonal(corr, 0.0); K = K + corr
                        notes.append(f"L{l} rank-{kk} share of plane error {np.sum(S[:kk]**2)/np.sum(S**2):.2f}")
                elif m == "k4ridge":
                    alpha = mu_ / s; g = 0.5 * (1 + np.vectorize(math.erf)(alpha / math.sqrt(2))); pdf = np.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)
                    k3 = np.diag(K); rho = Cn / np.outer(s, s); np.fill_diagonal(rho, 0.0)
                    F = np.stack([alpha, np.log(s) - np.mean(np.log(s)), g, pdf, k3 / (v * s), k4n / (v * v), np.sqrt(np.mean(rho ** 2, axis=1)), np.full(1024, l / 15.0)], 1)
                    mu_f, sd_f, w = ridge; pred = expand((F - mu_f) / sd_f) @ w; k4new = k4n + pred * v * v
                    notes.append(f"L{l} k4 ridge mean |corr|/|k4| {np.mean(np.abs(pred * v * v)) / np.mean(np.abs(k4n)):.3f}")
                    k4n = k4new; np.fill_diagonal(k22, k4n)
            return fn(mu, C, fnp.asarray(K), fnp.asarray(k4n), fnp.asarray(k22), metric)
        return fn(mu, C, K21, k4, k22_override, metric)
    return w_
for name in ("_nonlinear_first_gaussian", "_nonlinear_pair_batch", "_nonlinear_terminal_live", "nonlinear"):
    setattr(mod, name, wrap(getattr(mod, name)))
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    out = np.asarray(mod.Estimator().predict(mlp, 2**41), dtype=np.float64)
mse = ((out - truth) ** 2).mean(axis=1)
print(f"r{rank} {'+'.join(MODES)} at layers {sorted(LAYERS)}: final MSE {mse[15]:.3e} | " + " ".join(f"L{l}:{mse[l]:.2e}" for l in range(11, 16)) + " | " + "; ".join(notes[:6]))
