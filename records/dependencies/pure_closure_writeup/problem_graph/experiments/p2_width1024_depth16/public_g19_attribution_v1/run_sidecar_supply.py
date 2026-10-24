#!/usr/bin/env python3
"""TAG g19b-sidecar-supply-v1: score the K22-supply sidecar as the
(k3,k4)-error estimator. Two forwards per row on the SAME estimator file:
(1) base host - tap state + native tape gates Phi; (2) supplied host (the
banked K22-residual supply monkeypatch) - tap state. Then offline:
delta_hat_l = Edg2(host m,v, supplied k3,k4) - Edg2(host state), transported
r_l = Phi_l (r_{l-1} W_l) + delta_hat_l, correction added at the output,
scored against the saved MC truth. Also the supplied-v variant and the
supplied-state-quality ladder. Usage: run_sidecar_supply.py <rank>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EXTENDED = False

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean, load_host, tap_host_state


def instrument_supply(mod):
    """The banked full-rank K22-residual supply (verbatim pattern from
    public_k211_supply_v1/run_k22full_raw.py, the -60 percent receipt)."""
    state = {"K22resid": None, "W": None}
    orig_reset_stars = mod._reset_stars
    orig_assemble = mod._assemble_quartic_provider
    orig_push = mod._push_quartic

    def stars_wrap(k21_target, cp_post_k21, k4, k31, k22, covariance, variance,
                   covariance2, variance_outer, quartic_provider_planes=None,
                   diagonal_context=None):
        result = orig_reset_stars(k21_target, cp_post_k21, k4, k31, k22,
                                  covariance, variance, covariance2,
                                  variance_outer, quartic_provider_planes,
                                  diagonal_context)
        k4r, k31r, k22r = mod._quartic_residual(k4, k31, k22,
                                                quartic_provider_planes)
        arr = np.asarray(k22r, dtype=np.float64).view(np.ndarray)
        sym = 0.5 * (arr + arr.T)
        var = np.asarray(variance, dtype=np.float64).view(np.ndarray)
        cov = np.asarray(covariance, dtype=np.float64).view(np.ndarray)
        pairing = np.outer(var, var) + 2.0 * cov * cov
        np.fill_diagonal(pairing, 0.0)
        np.fill_diagonal(sym, 0.0)
        state["K22resid"] = sym - float(result[2]) * pairing
        if EXTENDED:
            r31 = np.asarray(k31r, dtype=np.float64).view(np.ndarray).copy()
            np.fill_diagonal(r31, 0.0)
            state["R31"] = r31
            state["k4d"] = np.asarray(k4r, dtype=np.float64).view(np.ndarray).copy()
        return result

    def push_wrap(star_h, pairing_alpha, weight_context, pre_covariance,
                  pre_variance, diagonal_context, fusion_context):
        state["W"] = np.asarray(weight_context[0],
                                dtype=np.float64).view(np.ndarray)
        return orig_push(star_h, pairing_alpha, weight_context, pre_covariance,
                         pre_variance, diagonal_context, fusion_context)

    def assemble_wrap(host_planes, provider_planes):
        k4, k31, k22 = orig_assemble(host_planes, provider_planes)
        if state["K22resid"] is not None and state["W"] is not None:
            mod._K_LOG.append(state["K22resid"].copy())
            W = state["W"]
            W2 = W * W
            K = state["K22resid"]
            KW2 = K @ W2
            S = 3.0 * np.einsum("ia,ia->a", W2, KW2)
            k4 = k4 + mod.fnp.asarray(S, dtype=mod.fnp.float64)
            W2TK = W2.T @ K
            k31_add = 3.0 * ((W2TK * W.T) @ W)
            k22_add = W2.T @ KW2
            if EXTENDED and state.get("R31") is not None:
                W3 = W2 * W
                R = state["R31"]
                RW = R @ W
                S = S + 4.0 * np.einsum("ia,ia->a", W3, RW)
                k4 = k4 + mod.fnp.asarray(
                    4.0 * np.einsum("ia,ia->a", W3, RW), dtype=mod.fnp.float64)
                k31_add = k31_add + W3.T @ RW + 3.0 * ((W2 * RW).T @ W)
                M = W2.T @ (W * RW)
                k22_add = k22_add + 2.0 * (M + M.T)
                dvec = state["k4d"]
                W4 = W2 * W2
                k4 = k4 + mod.fnp.asarray(W4.T @ dvec, dtype=mod.fnp.float64)
                k31_add = k31_add + (W3 * dvec[:, None]).T @ W
                k22_add = k22_add + W2.T @ (dvec[:, None] * W2)
            np.fill_diagonal(k31_add, 0.0)
            np.fill_diagonal(k22_add, 0.0)
            k31 = k31 + mod.fnp.asarray(k31_add, dtype=mod.fnp.float64)
            k22 = k22 + mod.fnp.asarray(k22_add, dtype=mod.fnp.float64)
        return k4, k31, k22

    mod._reset_stars = stars_wrap
    mod._push_quartic = push_wrap
    mod._assemble_quartic_provider = assemble_wrap


def one_forward(rank, with_supply, tag):
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                      allow_pickle=False)
    mod = load_host(tag)
    mod._K_LOG = []
    states = []
    tap_host_state(mod, states)
    if with_supply:
        instrument_supply(mod)
    mlp = MLP(width=1024, depth=16,
              weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=tag)
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=600.0, quiet=True):
            host_v, raw_v, _ = mod.predict_with_native_tape(mlp)
    return (np.asarray(host_v, dtype=np.float64),
            np.asarray(raw_v, dtype=np.float64), states,
            [np.asarray(weights[l], dtype=np.float64) for l in range(16)],
            mod._K_LOG)


def main():
    global EXTENDED
    rank = int(sys.argv[1])
    EXTENDED = len(sys.argv) > 2 and sys.argv[2] == "ext"
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    mc_k3, mc_k4, mc_v = d["mc_pre_k3"], d["mc_pre_k4"], d["mc_pre_v"]
    mc_post, mc_se, host_post_saved = d["mc_post"], d["mc_se"], d["host_post"]

    host_post, raw20, hstates, W, _ = one_forward(rank, False, f"sb_b_{rank}")
    assert float(np.max(np.abs(host_post - host_post_saved))) < 1e-6
    _sup_post, _r2, sstates, _, K_log = one_forward(rank, True, f"sb_s_{rank}")
    np.savez(HERE / f"sidecar_K_rank{rank:04d}.npz",
             K=np.stack(K_log).astype(np.float32))
    np.savez(HERE / f"sidecar_states_rank{rank:04d}.npz",
             s_m=np.stack([t["m"] for t in sstates]),
             s_v=np.stack([t["v"] for t in sstates]),
             s_k3=np.stack([t["k3"] for t in sstates]),
             s_k4=np.stack([t["k4"] for t in sstates]),
             raw20=raw20)

    e_true = mc_post - host_post
    quality = {}
    for l in (5, 11, 15):
        i = l - 1
        quality[str(l)] = {
            "k4_rel_base": float(np.sqrt(np.mean((hstates[i]["k4"] - mc_k4[l]) ** 2)
                                         / np.mean(mc_k4[l] ** 2))),
            "k4_rel_supplied": float(np.sqrt(np.mean((sstates[i]["k4"] - mc_k4[l]) ** 2)
                                             / np.mean(mc_k4[l] ** 2))),
            "k3_rel_base": float(np.sqrt(np.mean((hstates[i]["k3"] - mc_k3[l]) ** 2)
                                         / np.mean(mc_k3[l] ** 2))),
            "k3_rel_supplied": float(np.sqrt(np.mean((sstates[i]["k3"] - mc_k3[l]) ** 2)
                                             / np.mean(mc_k3[l] ** 2))),
        }

    def score(deltas, name):
        r = np.zeros((16, 1024))
        for l in range(1, 16):
            r[l] = raw20[l, :, 5] * (r[l - 1] @ W[l]) + deltas[l]
        num16 = float(np.sum((e_true[15] - r[15]) ** 2))
        den16 = float(np.sum(e_true[15] ** 2))
        noise16 = float(np.sum(mc_se[15] ** 2))
        corrected_mse = num16 / 1024.0
        return {name: {
            "residual_over_error_L16": num16 / den16,
            "recovered_fraction_nc": 1.0 - max(num16 - noise16, 0.0)
                                           / max(den16 - noise16, 1e-30),
            "host_L16_mse": den16 / 1024.0,
            "corrected_L16_mse": corrected_mse,
        }}

    out = {"rank": rank, "supplied_state_quality": quality}
    for use_v, name in ((False, "sidecar_k3k4"), (True, "sidecar_k3k4v")):
        deltas = np.zeros((16, 1024))
        for l in range(1, 16):
            i = l - 1
            base = edg2_relu_mean(hstates[i]["m"], hstates[i]["v"],
                                  hstates[i]["k3"], hstates[i]["k4"])
            v_use = sstates[i]["v"] if use_v else hstates[i]["v"]
            deltas[l] = edg2_relu_mean(hstates[i]["m"], v_use,
                                       sstates[i]["k3"], sstates[i]["k4"]) - base
        out.update(score(deltas, name))
    (HERE / f"SIDECAR_SUPPLY_rank{rank:04d}{'_ext' if EXTENDED else ''}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
