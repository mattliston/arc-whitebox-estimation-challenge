#!/usr/bin/env python3
"""TAG ue5-probe-v1: univariate order-3 companion (He7 k3k4/144 + He9 k3^3/1296)
patched into the DEPLOYMENT supply card; paired vs on-disk per-row card raw.
Deriver's L5 verdict: shell pair terms exact, harm = order mismatch; this tests
the marginal half of the match at ~zero flop cost. Rows 35/100/160/220."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
CARD = P2 / "public_supply_card_v1/estimator.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
ROWS = [35, 100, 160, 220]


def load(tag, p):
    spec = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def patch_ue5(mod, lam=1.0):
    fnp = mod.fnp

    def ue5(coefficients, raw, sigma, alpha, pdf, k3, k4, sigma2, sigma3,
            inverse_context=None):
        mass, raw1, raw2, raw3, raw4 = raw
        if inverse_context is None:
            inverse_context = mod._inverse_sigma_context(sigma)
        _inverse_sigma, _inv1, _inv2, inv3, inv4, inv6 = inverse_context
        k3_squared = k3 * k3
        c1, c2, c3 = coefficients[0], coefficients[1], coefficients[2]
        sigma4 = sigma2 * sigma2
        c4_3 = 24.0 * sigma3 * raw1
        c4_4 = 24.0 * sigma4 * mass
        c4_6 = -24.0 * sigma4 * alpha * pdf
        alpha2 = alpha * alpha
        c4_7 = 24.0 * sigma4 * (alpha2 - 1.0) * pdf
        c4_9 = 24.0 * sigma4 * (alpha2 * alpha2 - 6.0 * alpha2 + 3.0) * pdf
        inv7 = inv3 * inv4
        inv9 = inv3 * inv6
        standardized = fnp.stack(
            (
                k3 * inv3 / 6.0,
                k4 * inv4 / 24.0,
                k3_squared * inv6 / 72.0,
                lam * k3 * k4 * inv7 / 144.0,
                lam * k3_squared * k3 * inv9 / 1296.0,
            ),
            axis=0,
        )
        coefficient_table = fnp.reshape(
            fnp.stack(
                (
                    c1[0], c2[0], c3[0], raw4,
                    c1[3], c2[3], c3[3], c4_3,
                    c1[4], c2[4], c3[4], c4_4,
                    c1[6], c2[6], c3[6], c4_6,
                    c1[7], c2[7], c3[7], c4_7,
                    c1[9], c2[9], c3[9], c4_9,
                ),
                axis=0,
            ),
            (6, 4, int(sigma.shape[0])),
        )
        correction_terms = coefficient_table[1:] * standardized[:, None, :]
        corrected = coefficient_table[0]
        for t in range(5):
            corrected = corrected + correction_terms[t]
        corrected1 = fnp.maximum(corrected[0], 0.0)
        corrected2 = fnp.maximum(
            corrected[1],
            corrected1 * corrected1 + mod._VAR_FLOOR,
        )
        return corrected1, corrected2, corrected[2], corrected[3]

    mod._univariate_edgeworth = ue5


def main():
    base = {}
    for part in (0, 1):
        base.update(json.loads(
            (P2 / f"public_supply_card_v1/card_raw200_part{part}.json").read_text()))
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)
    rel = []
    for rank in ROWS:
        mod = load(f"ue5_{rank}", CARD)
        patch_ue5(mod)
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"ue5-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True) as ctx:
                pred = np.asarray(mod.Estimator().predict(mlp, 2**43), dtype=np.float64)
        assert pred.shape == (16, 1024), pred.shape
        mse = float(np.mean((targets[rank][15] - pred[15]) ** 2))
        b = base[str(rank)]["mse"]
        rel.append(mse / b - 1.0)
        print(json.dumps({"rank": rank, "card": round(b * 1e8, 4),
                          "ue5": round(mse * 1e8, 4),
                          "rel": round(mse / b - 1.0, 4)}), flush=True)
    print(json.dumps({"mean_rel_change": round(float(np.mean(rel)), 4),
                      "wins": int(sum(r < 0 for r in rel)), "n": len(rel)}))


if __name__ == "__main__":
    main()
