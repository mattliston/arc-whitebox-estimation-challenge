"""ORACLE-FINAL-SENSITIVITY-V1: which input order pays at the output.

TAG: public_oracle_kappa_probe_v1/analyze_final_sensitivity.py
Inputs : mc_oracle_final_rank{R}.npz (L16 oracle pre stats + post_mean),
         final_tap_rank{R}.npz (supplied-host L16 consumed m/v/k3/k4)
Output : FINAL_SENSITIVITY_RESULT.json

At the final consumption site (no downstream feedback), decompose the model's
L16 post-mean error dE = f(model inputs) - mc_post_mean into per-order
contributions c_x = (df/dx) * (model_x - oracle_x) for x in (mean, var, k3, k4),
with f the model's OWN univariate Edgeworth map (module _q38_coefficients +
_univariate_edgeworth, evaluated under a BudgetContext) and df/dx central
finite differences at the model's consumed point.  The remainder
dE - sum(c_x) contains the map's own truncation error (the intrinsic
Edgeworth limit given exact inputs), linearization error, and MC noise.

Attribution: share_x = mean(c_x * dE) / mean(dE^2) (signed projection; sums
with the remainder share to 1); also RMS of each c_x, the linearization check,
and each channel's MC noise floor |df/dx|*se_x.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
BASE = PKG.parent / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
RANKS = tuple(int(a) for a in sys.argv[1:]) or (35, 201)
ORDERS = ("mean", "var", "k3", "k4")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def load_module():
    spec = importlib.util.spec_from_file_location("final_sensitivity_map", BASE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_map(module):
    fnp = module.fnp
    hermite_context = module._hermite_context()

    def f(mean, var, k3, k4):
        m = fnp.asarray(mean, dtype=fnp.float64)
        v = fnp.asarray(var, dtype=fnp.float64)
        (coefficients, raw, sigma, alpha, pdf, sigma2, sigma3) = module._q38_coefficients(
            m, v, hermite_context
        )
        out = module._univariate_edgeworth(
            coefficients, raw, sigma, alpha, pdf,
            fnp.asarray(k3, dtype=fnp.float64), fnp.asarray(k4, dtype=fnp.float64),
            sigma2, sigma3,
        )[0]
        return np.asarray(out, dtype=np.float64).view(np.ndarray).copy()

    return f


def central_grad(f, point, order_index, step):
    args_plus = [p.copy() for p in point]
    args_minus = [p.copy() for p in point]
    args_plus[order_index] = args_plus[order_index] + step
    args_minus[order_index] = args_minus[order_index] - step
    return (f(*args_plus) - f(*args_minus)) / (2.0 * step)


def main():
    module = load_module()
    per_rank = {}
    pooled = {"dE_sq": 0.0, "n": 0}
    pooled_c = {x: 0.0 for x in ORDERS}
    pooled_proj = {x: 0.0 for x in ORDERS}
    with threadpool_limits(limits=2):
        with module.flops.BudgetContext(
            flop_budget=2**40, wall_time_limit_s=300, quiet=True,
            namespace="final_sensitivity_map",
        ):
            f = make_map(module)
            for rank in RANKS:
                mc = np.load(PKG / f"mc_oracle_final_rank{rank:04d}.npz")
                tap = np.load(PKG / f"final_tap_rank{rank:04d}.npz")
                model_point = [
                    tap["model_pre_mean"].astype(np.float64),
                    tap["model_pre_var"].astype(np.float64),
                    tap["model_k3"].astype(np.float64),
                    tap["model_k4"].astype(np.float64),
                ]
                oracle_point = [
                    mc["pre_mean"].astype(np.float64),
                    mc["pre_var"].astype(np.float64),
                    mc["k3"].astype(np.float64),
                    mc["k4"].astype(np.float64),
                ]
                oracle_se = [mc["se_pre_mean"], mc["se_pre_var"], mc["se_k3"], mc["se_k4"]]

                model_out = f(*model_point)
                dE = model_out - mc["post_mean"].astype(np.float64)

                grads, contribs = {}, {}
                for idx, name in enumerate(ORDERS):
                    scale = rms(model_point[idx])
                    step = 1e-4 * scale + 1e-4 * np.abs(model_point[idx])
                    grads[name] = central_grad(f, model_point, idx, step)
                    contribs[name] = grads[name] * (model_point[idx] - oracle_point[idx])

                total_c = sum(contribs.values())
                oracle_out = f(*oracle_point)
                map_error_at_oracle_inputs = oracle_out - mc["post_mean"].astype(np.float64)
                lin_check = model_out - oracle_out - total_c

                # The MC reference noise sits inside dE; correct the share
                # denominator for it, floored at 20% of the raw value so the
                # correction can never collapse the denominator.
                dE_sq_raw = float(np.mean(dE ** 2))
                se_sq = float(np.mean(mc["se_post_mean"].astype(np.float64) ** 2))
                dE_sq = max(dE_sq_raw - se_sq, 0.2 * dE_sq_raw)
                entry = {
                    "dE_rms": rms(dE),
                    "dE_mse_raw": dE_sq_raw,
                    "dE_mse_noise_corrected": dE_sq,
                    "denominator_note": "shares use the noise-corrected MSE (floored at 20% of raw)",
                    "map_error_at_oracle_inputs_rms": rms(map_error_at_oracle_inputs),
                    "linearization_residual_rms": rms(lin_check),
                    "mc_se_rms_post_mean": rms(mc["se_post_mean"]),
                    "orders": {},
                }
                remainder = 1.0
                for name in ORDERS:
                    share = float(np.mean(contribs[name] * dE) / dE_sq)
                    remainder -= share
                    entry["orders"][name] = {
                        "delta_rel_rms": rms(np.asarray(model_point[ORDERS.index(name)])
                                             - oracle_point[ORDERS.index(name)])
                                          / rms(oracle_point[ORDERS.index(name)]),
                        "contribution_rms": rms(contribs[name]),
                        "share_of_dE_mse": round(share, 4),
                        "noise_floor_rms": rms(grads[name] * oracle_se[ORDERS.index(name)]),
                    }
                entry["remainder_share_map_plus_nonlinear_plus_noise"] = round(remainder, 4)
                per_rank[str(rank)] = entry
                pooled["dE_sq"] += dE_sq
                pooled["n"] += 1
                for name in ORDERS:
                    pooled_c[name] += float(np.mean(contribs[name] ** 2))
                    pooled_proj[name] += float(np.mean(contribs[name] * dE))
                print(f"[final_sensitivity] rank={rank} dE_rms={rms(dE):.3e} "
                      f"shares=" + ",".join(
                          f"{n}:{entry['orders'][n]['share_of_dE_mse']}" for n in ORDERS),
                      flush=True)

    pooled_shares = {name: round(pooled_proj[name] / pooled["dE_sq"], 4) for name in ORDERS}
    dominant = max(pooled_shares, key=lambda n: pooled_shares[n])
    result = {
        "tag": "public_oracle_kappa_probe_v1/final_sensitivity",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "At the final (L16) consumption site, which input-order error actually pays in the output mean error?",
        "method": "central finite-difference sensitivities of the model's own univariate Edgeworth map at the model's consumed point; contributions c_x = (df/dx)(model_x - oracle_x); signed projection shares of mean(dE^2); remainder = map truncation + nonlinearity + MC noise",
        "host": "workspace-v4 + K22-residual supply (run_state_audit_deep instrument verbatim; L16 q38 call identified by univariate-call count = 14)",
        "oracle": "mc_oracle_final_rank*.npz, 8M samples, independent seed offset +13",
        "ranks": list(RANKS),
        "per_rank": per_rank,
        "pooled_shares_of_dE_mse": pooled_shares,
        "pooled_contribution_rms": {name: float(np.sqrt(pooled_c[name] / pooled["n"]))
                                    for name in ORDERS},
        "pooled_dE_rms": float(np.sqrt(pooled["dE_sq"] / pooled["n"])),
        "dominant_order": dominant,
        "reading": "share_of_dE_mse is signed: positive = this channel's input error pushes the output in the direction of the observed error; the remainder share is the map's own limit given exact inputs (plus nonlinearity and MC noise)",
    }
    out = PKG / "FINAL_SENSITIVITY_RESULT.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[final_sensitivity] wrote {out}")
    print(json.dumps({
        "pooled_shares": pooled_shares,
        "dominant_order": dominant,
        "per_rank_summary": {r: {"dE_rms": per_rank[r]["dE_rms"],
                                 "map_error_at_oracle_inputs_rms": per_rank[r]["map_error_at_oracle_inputs_rms"],
                                 "remainder_share": per_rank[r]["remainder_share_map_plus_nonlinear_plus_noise"]}
                             for r in per_rank},
    }, indent=1))


if __name__ == "__main__":
    main()
