"""ORACLE-KAPPA-VARIANTS-ANALYZE-V1: name the lost kappa4 channel.

TAG: public_oracle_kappa_probe_v1/analyze_variants.py
Inputs : mc_oracle_rank{R}.npz (8M oracle), probe_rank{R}.npz (base captures),
         variant_{cap8192,k22r}_rank{R}.npz (variant captures)
Output : ORACLE_KAPPA_VARIANTS.json

Compares k4 median unit ratios (model/oracle, layers 2-6) of the base
workspace-v4 estimator against (a) the same estimator with _CP4_CAP=8192 (no
CP4 truncation through layer 6) and (b) the k22r rank-4 k22-residual-carrier
estimator (condrepair_v3 lineage).  Layer-6 recovery fraction per variant =
(ratio_variant - ratio_base) / (1 - ratio_base).  Verdict names the dominant
lost channel: CP4_DROPS | GENERIC_K4_UNCARRIED | MIXED.
"""

import json
import time
from pathlib import Path

import numpy as np

PKG = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_oracle_kappa_probe_v1")
RANKS = (35, 100, 201)
LAYERS = (2, 3, 4, 5, 6)


def unit_ratios(model, oracle):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(model, dtype=np.float64) / np.asarray(oracle, dtype=np.float64)
    return ratio[np.isfinite(ratio)]


def ratio_tables(k4_by_rank):
    """k4_by_rank[rank] = array [14+, width] of consumed k4 per call; layer l = call l-1."""
    per_rank = {}
    pooled = {}
    for layer in LAYERS:
        chunks = []
        for rank in RANKS:
            mc = np.load(PKG / f"mc_oracle_rank{rank:04d}.npz")
            ratios = unit_ratios(k4_by_rank[rank][layer - 2], mc["k4"][layer - 1])
            per_rank.setdefault(str(rank), {})[str(layer)] = float(np.median(ratios))
            chunks.append(ratios)
        pooled[str(layer)] = float(np.median(np.concatenate(chunks)))
    return per_rank, pooled


def recovery(pooled_var, pooled_base):
    out = {}
    for layer in LAYERS:
        base = pooled_base[str(layer)]
        var = pooled_var[str(layer)]
        deficit = 1.0 - base
        out[str(layer)] = float((var - base) / deficit) if deficit > 0.05 else None
    return out


def main():
    base_k4 = {rank: np.load(PKG / f"probe_rank{rank:04d}.npz")["base_k4"] for rank in RANKS}
    cap_k4 = {rank: np.load(PKG / f"variant_cap8192_rank{rank:04d}.npz")["k4_in"] for rank in RANKS}
    k22_k4 = {rank: np.load(PKG / f"variant_k22r_rank{rank:04d}.npz")["k4_in"] for rank in RANKS}

    base_per_rank, base_pooled = ratio_tables(base_k4)
    cap_per_rank, cap_pooled = ratio_tables(cap_k4)
    k22_per_rank, k22_pooled = ratio_tables(k22_k4)

    # cap8192 shares every code path with base until the first CP4 truncation
    # (reset 3 = the layer-4 state), so its consumed k4 at layers 2-3 must match.
    identity_check = {}
    for layer in (2, 3):
        rel = []
        for rank in RANKS:
            b = base_k4[rank][layer - 2]
            c = cap_k4[rank][layer - 2]
            rel.append(float(np.max(np.abs(c - b)) / max(float(np.max(np.abs(b))), 1e-300)))
        identity_check[f"layer{layer}_max_rel_diff_cap_vs_base"] = max(rel)
    k22_lineage_check = {}
    for layer in (2, 3):
        rel = []
        for rank in RANKS:
            b = base_k4[rank][layer - 2]
            c = k22_k4[rank][layer - 2]
            rel.append(float(np.max(np.abs(c - b)) / max(float(np.max(np.abs(b))), 1e-300)))
        k22_lineage_check[f"layer{layer}_max_rel_diff_k22r_vs_base"] = max(rel)

    rec_cap = recovery(cap_pooled, base_pooled)
    rec_k22 = recovery(k22_pooled, base_pooled)
    rec_cap_l6 = rec_cap["6"]
    rec_k22_l6 = rec_k22["6"]

    if rec_cap_l6 is not None and rec_cap_l6 >= 0.5 and rec_cap_l6 >= 2.0 * max(rec_k22_l6 or 0.0, 0.0):
        verdict = "CP4_DROPS"
    elif rec_k22_l6 is not None and rec_k22_l6 >= 0.5 and rec_k22_l6 >= 2.0 * max(rec_cap_l6 or 0.0, 0.0):
        verdict = "GENERIC_K4_UNCARRIED"
    else:
        verdict = "MIXED"

    meta = {}
    for name in ("cap8192", "k22r"):
        for rank in RANKS:
            v = np.load(PKG / f"variant_{name}_rank{rank:04d}.npz")
            meta[f"{name}_rank{rank}"] = {
                "wall_s": float(v["wall_s"]), "flops_used": int(v["flops_used"]),
                "cp4_cap": int(v["cp4_cap"]),
            }

    result = {
        "tag": "public_oracle_kappa_probe_v1/variants",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Is the ~7%/layer kappa4 push loss the CP4 energy-truncation drops, or generic kappa4 content the carriers never hold?",
        "oracle": "same 8M MC files as ORACLE_KAPPA_RESULT.json (networks unchanged)",
        "variants": {
            "base": "workspace-v4 estimator, _CP4_CAP=2048 (ratios from probe_rank*.npz captures)",
            "cap8192": "same source, _CP4_CAP=8192 (one-line copy estimator_cap8192.py; no CP4 truncation through layer 6, max candidate rank 5120)",
            "k22r": "public_k22_carrier_ab_probe_v1/estimator_k22r.py, rank-4 k22-residual carrier, condrepair_v3 lineage",
        },
        "k4_median_unit_ratio_by_layer": {
            "base": {"pooled": base_pooled, "per_rank": base_per_rank},
            "cap8192": {"pooled": cap_pooled, "per_rank": cap_per_rank},
            "k22r": {"pooled": k22_pooled, "per_rank": k22_per_rank},
        },
        "recovery_fraction_by_layer": {
            "definition": "(ratio_variant - ratio_base) / (1 - ratio_base), pooled; None where the base deficit < 0.05",
            "cap8192": rec_cap,
            "k22r": rec_k22,
        },
        "layer6_recovery": {"cap8192": rec_cap_l6, "k22r": rec_k22_l6},
        "identity_checks": {
            "cap8192_vs_base_pre_truncation": identity_check,
            "k22r_vs_base_lineage": k22_lineage_check,
            "note": "cap8192 must match base at layers 2-3 (first truncation is reset 3 -> layer-4 state); k22r may differ per lineage",
        },
        "run_meta": meta,
        "verdict": verdict,
        "verdict_rule": "on pooled layer-6 recovery: CP4_DROPS if cap8192 >= 0.5 and >= 2x k22r; GENERIC_K4_UNCARRIED if k22r >= 0.5 and >= 2x cap8192; else MIXED",
    }
    out = PKG / "ORACLE_KAPPA_VARIANTS.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[analyze_variants] wrote {out}")
    print(json.dumps({
        "verdict": verdict,
        "k4_pooled": {"base": base_pooled, "cap8192": cap_pooled, "k22r": k22_pooled},
        "recovery": {"cap8192": rec_cap, "k22r": rec_k22},
        "identity_checks": result["identity_checks"],
    }, indent=1))


if __name__ == "__main__":
    main()
