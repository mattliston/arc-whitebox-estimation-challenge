"""ORACLE-DEEP-ANALYZE-V1: supplied-host state fidelity, layers 2-15.

TAG: public_oracle_kappa_probe_v1/analyze_deep.py
Inputs : mc_oracle_deep_rank{R}.npz (8M, 15 layers), deep_audit_rank{R}.npz
Output : ORACLE_DEEP_RESULT.json

Ratio convention (from public_k211_supply_v1/run_state_audit.py): masked
median over units with |oracle| above its per-layer median (top-half units),
of consumed/oracle; unmasked medians included for continuity with
ORACLE_KAPPA_RESULT.json.  consumed[L-2] pairs oracle index L-1.

Verdict rule (deep layers 7-15, on the pooled masked k4/k3 ratios):
  K3_DEEP_DRIFT            if k3 falls below 0.85 deep and its drop from its
                           L2-6 mean exceeds k4's drop;
  DEEP_K4_DECAYS           if k4 at L15 is more than 0.10 below its L6 value,
                           or k4 sustains < 0.80 for two consecutive deep
                           layers (first such layer named);
  DEEP_STATE_HOLDS_MAP_LIMITED otherwise (ratios hold ~flat to L15, so the
                           remaining born error is the map's own order limit).
"""

import json
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
RANKS = (35, 201)
LAYERS = tuple(range(2, 16))


def masked_median_ratio(model, oracle):
    mask = np.abs(oracle) > np.quantile(np.abs(oracle), 0.5)
    return float(np.median(model[mask] / oracle[mask]))


def unmasked_median_ratio(model, oracle):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = model / oracle
    return float(np.median(ratio[np.isfinite(ratio)]))


def pooled_masked(models, oracles):
    chunks = []
    for model, oracle in zip(models, oracles):
        mask = np.abs(oracle) > np.quantile(np.abs(oracle), 0.5)
        chunks.append(model[mask] / oracle[mask])
    return float(np.median(np.concatenate(chunks)))


def main():
    mc = {rank: np.load(PKG / f"mc_oracle_deep_rank{rank:04d}.npz") for rank in RANKS}
    audit = {rank: np.load(PKG / f"deep_audit_rank{rank:04d}.npz") for rank in RANKS}
    for rank in RANKS:
        if int(mc[rank]["n_samples"]) < 8_000_000 or mc[rank]["k4"].shape[0] != 15:
            raise AssertionError(f"deep oracle for rank {rank} wrong shape/samples")

    tables = {"k3": {}, "k4": {}}
    noise = {"k3": {}, "k4": {}}
    for channel in ("k3", "k4"):
        for rank in RANKS:
            per = {}
            for layer in LAYERS:
                model = audit[rank][f"{channel}_in"][layer - 2]
                oracle = mc[rank][channel][layer - 1]
                per[str(layer)] = {
                    "masked_median": round(masked_median_ratio(model, oracle), 4),
                    "unmasked_median": round(unmasked_median_ratio(model, oracle), 4),
                }
            tables[channel][str(rank)] = per
        pooled = {}
        for layer in LAYERS:
            pooled[str(layer)] = round(pooled_masked(
                [audit[rank][f"{channel}_in"][layer - 2] for rank in RANKS],
                [mc[rank][channel][layer - 1] for rank in RANKS],
            ), 4)
        tables[channel]["pooled_masked"] = pooled
        for layer in LAYERS:
            noise[channel][str(layer)] = round(float(np.mean([
                np.sqrt(np.mean(mc[rank][f"se_{channel}"][layer - 1] ** 2))
                / np.sqrt(np.mean(mc[rank][channel][layer - 1] ** 2))
                for rank in RANKS
            ])), 4)

    k3p = {int(l): v for l, v in tables["k3"]["pooled_masked"].items()}
    k4p = {int(l): v for l, v in tables["k4"]["pooled_masked"].items()}
    k3_shallow = float(np.mean([k3p[l] for l in range(2, 7)]))
    k4_shallow = float(np.mean([k4p[l] for l in range(2, 7)]))
    k3_deep_min = min(k3p[l] for l in range(7, 16))
    k4_deep_min = min(k4p[l] for l in range(7, 16))
    k3_drop = k3_shallow - k3_deep_min
    k4_drop = k4_shallow - k4_deep_min

    decay_layer = None
    for layer in range(7, 15):
        if k4p[layer] < 0.80 and k4p[layer + 1] < 0.80:
            decay_layer = layer
            break

    if k3_deep_min < 0.85 and k3_drop > k4_drop:
        verdict = "K3_DEEP_DRIFT"
    elif (k4p[15] < k4p[6] - 0.10) or decay_layer is not None:
        verdict = "DEEP_K4_DECAYS"
        if decay_layer is None:
            for layer in range(7, 16):
                if k4p[layer] < k4p[6] - 0.05:
                    decay_layer = layer
                    break
    else:
        verdict = "DEEP_STATE_HOLDS_MAP_LIMITED"

    result = {
        "tag": "public_oracle_kappa_probe_v1/deep",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does the K22-residual supply hold the k3/k4 state to layer 15, or does it lose hold past L6 (and on which channel)?",
        "host": "workspace-v4 estimator + K22-residual supply (public_k211_supply_v1/run_state_audit.py instrument(), used verbatim; predict path Estimator().predict)",
        "oracle": "8M-sample MC, 15 layers, ranks 35 and 201 (mc_oracle_deep_rank*.npz)",
        "ranks": list(RANKS),
        "ratio_convention": "median of consumed/oracle over units with |oracle| above its per-layer median (masked); unmasked medians alongside",
        "k3_median_unit_ratio_by_layer": tables["k3"],
        "k4_median_unit_ratio_by_layer": tables["k4"],
        "oracle_noise_floor_rel_by_layer": noise,
        "summary": {
            "k3_shallow_mean_L2_6": round(k3_shallow, 4),
            "k4_shallow_mean_L2_6": round(k4_shallow, 4),
            "k3_deep_min_L7_15": round(k3_deep_min, 4),
            "k4_deep_min_L7_15": round(k4_deep_min, 4),
            "k4_L6": k4p[6], "k4_L15": k4p[15],
            "k3_L6": k3p[6], "k3_L15": k3p[15],
            "first_sustained_k4_decay_layer": decay_layer,
        },
        "verdict": verdict,
        "verdict_rule": "K3_DEEP_DRIFT if pooled masked k3 < 0.85 deep with a larger drop than k4; DEEP_K4_DECAYS if k4(L15) < k4(L6) - 0.10 or k4 < 0.80 sustained 2 layers (first layer named); else DEEP_STATE_HOLDS_MAP_LIMITED",
    }
    out = PKG / "ORACLE_DEEP_RESULT.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[analyze_deep] wrote {out}")
    print(json.dumps({
        "verdict": verdict,
        "summary": result["summary"],
        "k3_pooled_masked": tables["k3"]["pooled_masked"],
        "k4_pooled_masked": tables["k4"]["pooled_masked"],
        "noise_floors": noise,
    }, indent=1))


if __name__ == "__main__":
    main()
