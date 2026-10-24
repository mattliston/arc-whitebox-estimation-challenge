"""ORACLE-SECOND-ORDER-ANALYZE-V1: supplied-host 2nd-order state vs MC, L2-15.

TAG: public_oracle_kappa_probe_v1/analyze_second_order.py
Inputs : mc_oracle_deep_rank{R}.npz (pre_mean/pre_var + SEs), mc_pairs_rank{R}.npz
         (2000-pair pre-activation covariance + SEs), second_order_rank{R}.npz
         (model taps).  Ranks 35, 201; model index L-2 pairs oracle index L-1.
Output : ORACLE_SECOND_ORDER_RESULT.json

Per channel (pre_mean, pre_var, pre_cov_pairs), per layer L2-15, per rank:
relative RMS error, masked median unit ratio (units/pairs with |oracle| above
its median), and the MC noise floor; noise-corrected relative error
nc = sqrt(max(rel_rms^2 - floor^2, 0)).

Verdict: SECOND_ORDER_EXACT if no channel has nc > 0.02 at any layer; else the
channel with the largest worst-layer nc names the verdict (MEAN_DRIFTS |
VAR_DRIFTS | COV_DRIFTS) and every channel's tripped layers are listed.
"""

import json
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
RANKS = (35, 201)
LAYERS = tuple(range(2, 16))
NC_TRIP = 0.02


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def masked_median_ratio(model, oracle):
    mask = np.abs(oracle) > np.quantile(np.abs(oracle), 0.5)
    return float(np.median(model[mask] / oracle[mask]))


def main():
    mc = {rank: np.load(PKG / f"mc_oracle_deep_rank{rank:04d}.npz") for rank in RANKS}
    pairs = {rank: np.load(PKG / f"mc_pairs_rank{rank:04d}.npz") for rank in RANKS}
    model = {rank: np.load(PKG / f"second_order_rank{rank:04d}.npz") for rank in RANKS}
    for rank in RANKS:
        if not (np.array_equal(model[rank]["pairs_i"], pairs[rank]["pairs_i"])
                and np.array_equal(model[rank]["pairs_j"], pairs[rank]["pairs_j"])):
            raise AssertionError(f"rank {rank}: pair lists differ between MC and model taps")

    def channel_arrays(rank, channel, layer):
        if channel == "pre_mean":
            return (model[rank]["pre_mean"][layer - 2], mc[rank]["pre_mean"][layer - 1],
                    mc[rank]["se_pre_mean"][layer - 1])
        if channel == "pre_var":
            return (model[rank]["pre_var"][layer - 2], mc[rank]["pre_var"][layer - 1],
                    mc[rank]["se_pre_var"][layer - 1])
        return (model[rank]["pre_cov_pairs"][layer - 2], pairs[rank]["cov"][layer - 1],
                pairs[rank]["se_cov"][layer - 1])

    tables = {}
    worst = {}
    trips = {}
    for channel in ("pre_mean", "pre_var", "pre_cov_pairs"):
        tables[channel] = {}
        nc_by_layer = {}
        for rank in RANKS:
            per = {}
            for layer in LAYERS:
                m, o, se = channel_arrays(rank, channel, layer)
                rel = rms(m - o) / rms(o)
                floor = rms(se) / rms(o)
                nc = float(np.sqrt(max(rel ** 2 - floor ** 2, 0.0)))
                per[str(layer)] = {
                    "rel_rms_err": round(rel, 5),
                    "noise_floor_rel": round(floor, 5),
                    "rel_rms_err_noise_corrected": round(nc, 5),
                    "masked_median_ratio": round(masked_median_ratio(m, o), 5),
                }
                nc_by_layer.setdefault(layer, []).append(nc)
            tables[channel][str(rank)] = per
        pooled_nc = {layer: float(np.sqrt(np.mean(np.array(v) ** 2)))
                     for layer, v in nc_by_layer.items()}
        tables[channel]["pooled_nc_rel"] = {str(l): round(v, 5) for l, v in pooled_nc.items()}
        worst_layer = max(pooled_nc, key=pooled_nc.get)
        worst[channel] = (worst_layer, pooled_nc[worst_layer])
        trips[channel] = sorted(l for l, v in pooled_nc.items() if v > NC_TRIP)

    if all(not v for v in trips.values()):
        verdict = "SECOND_ORDER_EXACT"
    else:
        dominant = max((c for c in trips if trips[c]), key=lambda c: worst[c][1])
        verdict = {"pre_mean": "MEAN_DRIFTS", "pre_var": "VAR_DRIFTS",
                   "pre_cov_pairs": "COV_DRIFTS"}[dominant]

    result = {
        "tag": "public_oracle_kappa_probe_v1/second_order",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Is the supplied host's second-order state (pre-activation mean, variance, pair covariance) exact at depth, or does a channel drift?",
        "host": "workspace-v4 estimator + K22-residual supply (run_state_audit_deep.py instrument() verbatim; second-order capture added here via a _full_relu_reset wrapper since that file taps only k3/k4)",
        "oracle": "8M 15-layer MC (mc_oracle_deep_rank*.npz) for mean/var; independent 8M pass at 2000 fixed unit pairs (mc_pairs_rank*.npz, layer-1 gate vs exact (W1^T W1)_ij) for covariance",
        "ranks": list(RANKS),
        "pairs": {"count": 2000, "seed": 20260825, "convention": "i<j uniform distinct pairs, pairs_spec.py"},
        "tables": tables,
        "worst_pooled_nc_rel": {c: {"layer": worst[c][0], "value": round(worst[c][1], 5)} for c in worst},
        "tripped_layers_nc_above_2pct": {c: trips[c] for c in trips},
        "verdict": verdict,
        "verdict_rule": f"SECOND_ORDER_EXACT if no channel's pooled noise-corrected relative RMS error exceeds {NC_TRIP} at any layer L2-15; otherwise the channel with the largest worst-layer value names the verdict",
    }
    out = PKG / "ORACLE_SECOND_ORDER_RESULT.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[analyze_second_order] wrote {out}")
    print(json.dumps({
        "verdict": verdict,
        "worst_pooled_nc_rel": result["worst_pooled_nc_rel"],
        "tripped_layers": result["tripped_layers_nc_above_2pct"],
        "pooled_nc_rel": {c: tables[c]["pooled_nc_rel"] for c in tables},
    }, indent=1))


if __name__ == "__main__":
    main()
