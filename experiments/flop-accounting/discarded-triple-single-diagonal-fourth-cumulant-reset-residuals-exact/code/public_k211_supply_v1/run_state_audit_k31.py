#!/usr/bin/env python3
"""TAG k31-supply-audit-v1: thread the OTHER two discarded reset residuals
(k31 off-diagonal, k4 diagonal) through the push, on top of the K22 supply.
Arms: A = +R31, B = +R31+k4diag. Reads the k4 ratio ladder vs the deep MC
oracle (baseline K22-only: 0.94 at L6 decaying to 0.82 at L15).
Push chains (exact multiplicities):
  R31[i,j]=cum(x_i,x_i,x_i,x_j), diag zeroed:
    k4_a  += 4 sum_ij W_ia^3 W_ja R31[i,j]
    k31ab += (W3^T R31 W) + 3 (W2 o R31W)^T W
    k22ab += 2 (M + M^T), M = W2^T (W o R31W)
  d_i = k4 diagonal residual:
    k4_a += W4^T d;  k31ab += (W3 o d)^T W;  k22ab += W2^T (d o W2)
  (k31/k22 add diagonals zeroed: diagonal mass lives in the k4 channel)"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ORACLE = P2 / "public_oracle_kappa_probe_v1"


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def instrument(mod, use_k4diag):
    state = {"K22resid": None, "R31": None, "k4d": None, "W": None,
             "k3": [], "k4": []}
    orig_stars = mod._reset_stars
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic
    orig_ue = mod._univariate_edgeworth

    def tap_ue(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,
               inverse_context=None):
        state["k3"].append(np.asarray(k3, dtype=np.float64).view(np.ndarray).copy())
        state["k4"].append(np.asarray(k4, dtype=np.float64).view(np.ndarray).copy())
        return orig_ue(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2,
                       sigma3, inverse_context)

    def stars_wrap(k21_target, cp_post_k21, k4, k31, k22, covariance, variance,
                   covariance2, variance_outer, quartic_provider_planes=None,
                   diagonal_context=None):
        result = orig_stars(k21_target, cp_post_k21, k4, k31, k22, covariance,
                            variance, covariance2, variance_outer,
                            quartic_provider_planes, diagonal_context)
        k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22, quartic_provider_planes)
        arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
        sym = 0.5 * (arr + arr.T)
        var = np.asarray(variance, dtype=np.float64).view(np.ndarray)
        cov = np.asarray(covariance, dtype=np.float64).view(np.ndarray)
        pairing = np.outer(var, var) + 2.0 * cov * cov
        np.fill_diagonal(pairing, 0.0)
        np.fill_diagonal(sym, 0.0)
        state["K22resid"] = sym - float(result[2]) * pairing
        r31 = np.asarray(k31r, dtype=np.float64).view(np.ndarray).copy()
        np.fill_diagonal(r31, 0.0)
        state["R31"] = r31
        state["k4d"] = np.asarray(k4r, dtype=np.float64).view(np.ndarray).copy()
        return result

    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance, pre_variance,
                  diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0], dtype=np.float64).view(np.ndarray)
        return orig_push(star_h, pairing_alpha, weight_context, pre_covariance,
                         pre_variance, diagonal_context, fusion_context)

    def assemble_wrap(host_planes, provider_planes):
        k4, k31, k22 = orig_assemble(host_planes, provider_planes)
        if state["K22resid"] is not None and state["W"] is not None:
            W = state["W"]; W2 = W * W; W3 = W2 * W
            K = state["K22resid"]
            KW2 = K @ W2
            S = 3.0 * np.einsum("ia,ia->a", W2, KW2)
            k4a = S.copy()
            W2TK = W2.T @ K
            k31a = 3.0 * ((W2TK * W.T) @ W)
            k22a = W2.T @ KW2
            # R31 channel
            R = state["R31"]
            RW = R @ W
            k4a += 4.0 * np.einsum("ia,ia->a", W3, RW)
            k31a += W3.T @ RW + 3.0 * ((W2 * RW).T @ W)
            M = W2.T @ (W * RW)
            k22a += 2.0 * (M + M.T)
            if use_k4diag:
                d = state["k4d"]
                W4 = W2 * W2
                k4a += W4.T @ d
                k31a += (W3 * d[:, None]).T @ W
                k22a += W2.T @ (d[:, None] * W2)
            np.fill_diagonal(k31a, 0.0)
            np.fill_diagonal(k22a, 0.0)
            k4 = k4 + mod.fnp.asarray(k4a, dtype=mod.fnp.float64)
            k31 = k31 + mod.fnp.asarray(k31a, dtype=mod.fnp.float64)
            k22 = k22 + mod.fnp.asarray(k22a, dtype=mod.fnp.float64)
        return k4, k31, k22

    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap
    mod._univariate_edgeworth = tap_ue
    return state


def main():
    rank = 35
    oracle = np.load(ORACLE / f"mc_oracle_deep_rank{rank:04d}.npz")
    ok3, ok4 = oracle["k3"], oracle["k4"]
    for arm, use_k4diag in (("R31", False), ("R31+k4diag", True)):
        mod = load(f"k31aud_{arm}_{rank}", BASE)
        state = instrument(mod, use_k4diag)
        manifest = json.loads((STORE / "manifest.json").read_text())
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"k31-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**44, wall_time_limit_s=900.0, quiet=True):
                mod.Estimator().predict(mlp, 2**44)
        r4 = []
        for L in range(2, 16):
            m4 = state["k4"][L - 2]
            o4 = ok4[L - 1]
            m4_ = np.abs(o4) > np.quantile(np.abs(o4), 0.5)
            r4.append(round(float(np.median(m4[m4_] / o4[m4_])), 3))
        print(json.dumps({"rank": rank, "arm": arm, "k4_ratio_L2_15": r4}), flush=True)


if __name__ == "__main__":
    main()
