#!/usr/bin/env python3
"""Shell-on-supply eval: budget smoke + 3-row paired raw truth.

TAG=shell-supply-eval  (1) One full-path smoke of estimator_shell_supply.py
on store row rank 35 under BudgetContext(2**41): billed + margin.  (2) Raw
host truth (_predict, no corrector) on ranks 35/100/201 for BOTH arms -
the supply card and shell+supply - paired, fresh, under 2**41 each; the
supply arm is cross-checked against the banked card_raw200 parts.  Writes
SHELL_SUPPLY_RESULT.json (build receipt embedded).  The question: does the
shell's historical -10 percent still pay on the supplied host (fire bar:
raw all-200 < 2.769e-8 at supply util).  No submission, no store writes.
"""
import hashlib, importlib.util, json, os, sys, time
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
SUPPLY = P2 / "public_supply_card_v1" / "estimator.py"
SUPPLY_SHA = "63e2b95bfc1b2c5b2b109f3cb824c3c3c79a1ff5e39043e8232ad4e5077a8ea0"
SHELL_SUPPLY = HERE / "estimator_shell_supply.py"
BUILD_RECEIPT = HERE / "SHELL_SUPPLY_BUILD_RECEIPT.json"
RESULT = HERE / "SHELL_SUPPLY_RESULT.json"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
BANKED = [P2 / "public_supply_card_v1" / f"card_raw200_part{i}.json" for i in (0, 1)]
TRUTH_RANKS = (35, 100, 201)
BUDGET = 2**41

# Handed-down context (team lead, 2026-08-25):
FIRE_BAR_RAW_ALL200 = 2.769e-8
SUPPLY_ONLY_RAW_ALL200 = 3.335e-8
SHELL_HISTORICAL_EFFECT = -0.10


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_module(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def make_mlp(rank, entries, tag):
    entry = entries[rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"{tag}-{rank}")
    mlp.validate()
    return mlp


def main():
    build = json.loads(BUILD_RECEIPT.read_text())
    assert build["status"] == "BUILT" and not build["collisions"]
    shell_sha = sha(SHELL_SUPPLY)
    assert shell_sha == build["output_sha256"], "shell+supply card drifted"
    assert sha(SUPPLY) == SUPPLY_SHA, "supply card drifted"
    if RESULT.exists():
        raise FileExistsError(RESULT)

    flops.configure(symmetry_warnings=False)
    supply = load_module("eval_supply", SUPPLY)
    shell = load_module("eval_shell_supply", SHELL_SUPPLY)
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]

    # (1) Budget smoke: the full deployed path (host + frozen corrector).
    mlp = make_mlp(35, entries, "shellsupply-smoke")
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=600.0,
                                 quiet=True) as ctx:
            value = shell.Estimator().predict(mlp, BUDGET)
    smoke_wall = time.perf_counter() - started
    output = np.asarray(value, dtype=np.float64)
    smoke = {
        "store_row_rank": 35,
        "budget_context": BUDGET,
        "billed_flops": int(ctx.flops_used),
        "margin_to_2pow41": BUDGET - int(ctx.flops_used),
        "finite": bool(np.isfinite(output).all()),
        "nonnegative": bool((output >= 0.0).all()),
        "wall_seconds": smoke_wall,
    }
    print(json.dumps({"smoke": smoke}), flush=True)

    # (2) Paired raw truth on three ranks.
    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024).astype(np.float64)

    banked = {}
    for path in BANKED:
        if path.exists():
            banked.update(json.loads(path.read_text()))

    rows = {}
    deltas = []
    for rank in TRUTH_RANKS:
        arms = {}
        for tag, module in (("supply", supply), ("shell_supply", shell)):
            mlp = make_mlp(rank, entries, f"truth-{tag}")
            with threadpool_limits(limits=2):
                with flops.BudgetContext(flop_budget=BUDGET,
                                         wall_time_limit_s=600.0,
                                         quiet=True) as ctx2:
                    raw = np.asarray(module._predict(mlp), dtype=np.float64)
            mse = float(np.mean((raw[15] - targets[rank, 15]) ** 2))
            arms[tag] = {"final_mse": mse, "billed_flops": int(ctx2.flops_used),
                         "finite": bool(np.isfinite(raw).all())}
            print(json.dumps({"rank": rank, "arm": tag,
                              "mse_e8": round(mse * 1e8, 4)}), flush=True)
        shell_effect = arms["shell_supply"]["final_mse"] / arms["supply"]["final_mse"] - 1.0
        deltas.append(shell_effect)
        entry = {
            **arms,
            "shell_effect_fraction": shell_effect,
        }
        if str(rank) in banked:
            entry["supply_banked_cross_check"] = {
                "banked_mse": banked[str(rank)]["mse"],
                "delta_vs_fresh": arms["supply"]["final_mse"] - banked[str(rank)]["mse"],
            }
        rows[rank] = entry

    mean_effect = float(np.mean(deltas))
    projected_all200 = SUPPLY_ONLY_RAW_ALL200 * (1.0 + mean_effect)
    result = {
        "schema": "shell-supply-result-v1",
        "build": build,
        "shell_supply_sha256": shell_sha,
        "supply_sha256": SUPPLY_SHA,
        "smoke": smoke,
        "truth_rows": {str(k): v for k, v in rows.items()},
        "shell_effect": {
            "per_row_fraction": deltas,
            "mean_fraction": mean_effect,
            "historical_x_pilot_reference": SHELL_HISTORICAL_EFFECT,
            "note": "three rows only; single-grade rel-sd ~8 percent applies "
            "- this localizes direction/magnitude, the all-200 pass is the "
            "verdict instrument",
        },
        "fire_bar_context": {
            "bar_raw_all200_at_supply_util": FIRE_BAR_RAW_ALL200,
            "supply_only_raw_all200_banked": SUPPLY_ONLY_RAW_ALL200,
            "projected_shell_supply_all200_from_3row_effect": projected_all200,
            "projected_gap_to_bar_fraction": projected_all200 / FIRE_BAR_RAW_ALL200 - 1.0,
        },
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({
        "six_mses": {str(r): {a: rows[r][a]["final_mse"] for a in ("supply", "shell_supply")} for r in TRUTH_RANKS},
        "mean_shell_effect": mean_effect,
        "projected_all200": projected_all200,
        "smoke_billed": smoke["billed_flops"],
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
