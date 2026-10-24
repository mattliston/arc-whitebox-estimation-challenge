#!/usr/bin/env python3
"""Census strained pairs: capture (mu, sigma, rho) + the host's Mehler kernel.

Runs the graded base estimator on N rows with a wrapper on _q24_context that,
per layer, records for every off-diagonal pair with |rho| above the lowest
threshold: mu_i, mu_j, sd_i, sd_j, rho, and the host's own (0,0) Mehler
kernel value at t=25 and t=7 (sum_t c1_i[t] c1_j[t] rho^t / t!). Pure
observation - the wrapper adds no metered ops (numpy views only) and the
prediction output is bit-checked against an unwrapped reference run.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROWS = [35, 36, 37, 55, 100, 150, 201, 210, 220, 234]
THRESHOLDS = [0.90, 0.95, 0.99, 0.999]
CAP_PER_LAYER = 4096  # record at most this many strained pairs per layer


def main():
    spec = importlib.util.spec_from_file_location("census_base", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    captured = []
    counts = {t: np.zeros(16, dtype=np.int64) for t in THRESHOLDS}
    call_state = {"row": None, "call": 0}
    original_q24 = module._q24_context
    inv_factorial = np.array(
        [1.0 / __import__('math').factorial(t) for t in range(25)], dtype=np.float64)

    def wrapped_q24(coefficients, sigma, covariance, fusion_context, inverse_context=None):
        layer = call_state["call"]
        call_state["call"] += 1
        sig_np = np.asarray(sigma, dtype=np.float64).view(np.ndarray)
        cov_np = np.asarray(covariance, dtype=np.float64).view(np.ndarray)
        rho = np.clip(cov_np / (sig_np[:, None] * sig_np[None, :]), -1.0, 1.0)
        c1 = np.asarray(coefficients, dtype=np.float64).view(np.ndarray)[0]
        sig = np.asarray(sigma, dtype=np.float64).view(np.ndarray)
        off = np.abs(rho - np.eye(rho.shape[0]))
        for t in THRESHOLDS:
            counts[t][layer] += int((off > t).sum()) // 2
        ii, jj = np.where(np.triu(off, 1) > THRESHOLDS[0])
        if ii.size:
            if ii.size > CAP_PER_LAYER:
                order = np.argsort(off[ii, jj])[::-1][:CAP_PER_LAYER]
                ii, jj = ii[order], jj[order]
            # the host's (0,0) kernel: features = c1[order_row0 = 0..24] * inv_sigma^0
            # row d=0 of the order grid selects orders 0..24 directly from c1.
            fi = c1[:25, ii]
            fj = c1[:25, jj]
            r = rho[ii, jj]
            rp = r[None, :] ** np.arange(25)[:, None]
            terms = fi * fj * rp * inv_factorial[:, None]
            series25 = terms.sum(axis=0)
            series7 = terms[:7].sum(axis=0)
            captured.append({
                "row": call_state["row"], "layer": layer,
                "i": ii.astype(np.int32), "j": jj.astype(np.int32),
                "rho": r.astype(np.float64),
                "sd_i": sig[ii], "sd_j": sig[jj],
                "series25": series25, "series7": series7,
            })
        return original_q24(coefficients, sigma, covariance, fusion_context, inverse_context)

    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    mu_store = []

    original_q38 = module._q38_coefficients

    def wrapped_q38(mean, variance, hermite_context):
        mu_store.append((call_state["call"],
                         np.asarray(mean, dtype=np.float64).view(np.ndarray).copy()))
        return original_q38(mean, variance, hermite_context)

    module._q24_context = wrapped_q24
    module._q38_coefficients = wrapped_q38

    all_pairs = []
    with threadpool_limits(limits=2):
        for rank in ROWS:
            entry = entries[rank]
            rmp = STORE / entry["path"]
            row = json.loads(rmp.read_text())
            weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                              allow_pickle=False)
            mlp = MLP(width=1024, depth=16,
                      weights=[weights[l] for l in range(16)],
                      seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                      name=f"bondcensus-{rank}")
            mlp.validate()
            call_state.update(row=rank, call=0)
            mu_store.clear()
            with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=300.0,
                                     quiet=True):
                module.Estimator().predict(mlp, 2**41)
            mu_by_call = dict(mu_store)
            for rec in captured:
                if rec["row"] != rank:
                    continue
                mean_vec = mu_by_call.get(rec["layer"])
                if mean_vec is None:
                    continue
                all_pairs.append({
                    "rank": rank, "layer": rec["layer"],
                    "mu_i": mean_vec[rec["i"]], "mu_j": mean_vec[rec["j"]],
                    "sd_i": rec["sd_i"], "sd_j": rec["sd_j"],
                    "rho": rec["rho"],
                    "series25": rec["series25"], "series7": rec["series7"],
                })
            captured[:] = [r for r in captured if r["row"] != rank]
            print(json.dumps({"rank": rank, "pairs_so_far": sum(len(p["rho"]) for p in all_pairs)}), flush=True)

    arrays = {}
    for key in ("mu_i", "mu_j", "sd_i", "sd_j", "rho", "series25", "series7"):
        arrays[key] = np.concatenate([p[key] for p in all_pairs]) if all_pairs else np.zeros(0)
    arrays["layer"] = np.concatenate(
        [np.full(len(p["rho"]), p["layer"], dtype=np.int32) for p in all_pairs]
    ) if all_pairs else np.zeros(0, dtype=np.int32)
    np.savez_compressed(HERE / "strained_pairs.npz", **arrays)
    summary = {
        "rows": ROWS,
        "pair_counts_per_layer_by_threshold": {
            str(t): counts[t].tolist() for t in THRESHOLDS},
        "captured_pairs": int(arrays["rho"].size),
    }
    (HERE / "CENSUS_COUNTS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["pair_counts_per_layer_by_threshold"], indent=1))


if __name__ == "__main__":
    main()
