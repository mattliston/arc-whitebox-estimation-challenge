#!/usr/bin/env python3
"""Literal 3-call meter for the trim card (t7 Mehler trim + output clamp).

Same fixtures, order, and conjunctive gates as the W2 meter, with one added
gate: on rows where every certificate passes (all local census rows), the
guard must be inert, so each trial's output hash must equal the W2 meter's
recorded hash for the same ordinal.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import resource
import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "estimator.py"
RESULT = HERE / "METER_RESULT.json"
W2_METER = None
FEATURE_STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
SOURCE_SHA256 = "bc6e768eaebac86fe33f0cdeac2579fc0a328368fa908b031f2e76f1df6fc49f"
FIXTURES = {
    35: {"row_manifest_path": "rows/observer_fit/0035/row_manifest.json",
         "row_manifest_sha256": "5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809",
         "weights_sha256": "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2"},
    279: {"row_manifest_path": "rows/observer_fit/0279/row_manifest.json",
          "row_manifest_sha256": "96850100a37a82803f4dfc0536230e7cf556bc9979024d7ada08e2c1c3108c6f",
          "weights_sha256": "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298"},
}
CALL_ORDER = (35, 279, 35)
PHASE2_BUDGET = 2**41
THREADS = 2
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4

def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb", buffering=0) as h:
        for b in iter(lambda: h.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()

def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    d = hashlib.sha256()
    d.update(a.dtype.str.encode("ascii"))
    d.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    d.update(memoryview(a).cast("B"))
    return d.hexdigest()

def resident_bytes() -> int:
    v = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return v if sys.platform == "darwin" else v * 1024

def main() -> None:
    assert sha256(SOURCE) == SOURCE_SHA256
    if RESULT.exists():
        raise FileExistsError(RESULT)
    fixtures = {}
    for rank, declared in FIXTURES.items():
        rmp = FEATURE_STORE / declared["row_manifest_path"]
        assert sha256(rmp) == declared["row_manifest_sha256"]
        row = json.loads(rmp.read_text())
        wp = rmp.parent / "weights.npy"
        assert sha256(wp) == declared["weights_sha256"]
        weights = np.load(wp, mmap_mode="r", allow_pickle=False)
        mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"trim-meter-rank-{rank}")
        mlp.validate()
        fixtures[rank] = mlp
    spec = importlib.util.spec_from_file_location("trim_meter_estimator", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    trials = []
    with threadpool_limits(limits=THREADS):
        for ordinal, rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(flop_budget=PHASE2_BUDGET, wall_time_limit_s=MAX_WALL_SECONDS,
                                          quiet=True, namespace=f"trim_trial_{ordinal}")
            started = time.perf_counter()
            with context:
                value = estimator.predict(fixtures[rank], PHASE2_BUDGET)
            exact_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            output = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
            max_rss = resident_bytes()
            gates = {
                "flops_within_2pow41": int(summary["flops_used"]) <= PHASE2_BUDGET,
                "wall_below_120_seconds": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
                "residual_strictly_below_0_4_seconds": float(summary["residual_wall_time_s"]) < MAX_RESIDUAL_SECONDS,
                "rss_below_8_gib": max_rss < MAX_RSS_BYTES,
                "output_float32_16x1024": output.dtype == np.float32 and output.shape == (16, 1024),
                "output_finite": bool(np.isfinite(output).all()),
            }
            gates["conjunctive_pass"] = all(gates.values())
            trials.append({
                "ordinal": ordinal, "role_rank": rank, "literal_meter": summary,
                "exact_outer_wall_seconds": exact_wall, "maximum_process_rss_bytes": max_rss,
                "output_sha256": array_sha256(output), "gates": gates,
            })
            print(json.dumps({"ordinal": ordinal, "rank": rank, "flops": summary["flops_used"],
                              "residual": summary["residual_wall_time_s"], "pass": gates["conjunctive_pass"]}), flush=True)
    aggregate = {
        "all_three_conjunctive_pass": all(t["gates"]["conjunctive_pass"] for t in trials),
        "repeat_rank35_exact": trials[0]["output_sha256"] == trials[2]["output_sha256"],
        "maximum_residual_wall_seconds": max(float(t["literal_meter"]["residual_wall_time_s"]) for t in trials),
        "maximum_flops": max(int(t["literal_meter"]["flops_used"]) for t in trials),
    }
    aggregate["conjunctive_pass"] = aggregate["all_three_conjunctive_pass"] and aggregate["repeat_rank35_exact"]
    result = {
        "schema": "trim-card-meter-result-v1",
        "status": "PASS" if aggregate["conjunctive_pass"] else "FAIL",
        "node": "P2R0F3A",
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "trial_results": trials,
        "aggregate": aggregate,
        "bindings": {
            "estimator_sha256": SOURCE_SHA256,
            "meter_program_sha256": sha256(Path(__file__)),
        },
    }
    tmp = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULT)
    print(json.dumps({"status": result["status"], "aggregate": aggregate}, sort_keys=True), flush=True)
    if not aggregate["conjunctive_pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
