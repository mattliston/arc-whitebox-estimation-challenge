"""Inert candidate-only fixed14 acquisition, then sealed matched-baseline scoring.

The parent is the authenticated, unchanged archived response-P panel. No
estimator is executed in --score. Neither mode trains, repairs arrays, changes
geometry, selects a variant, submits, or overwrites any previous artifact.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE / "pair_polynomial_f32_panel_v1"
ARCHIVED = HERE / "response_rows_panel_v1/response"
DEV_ROOT = HERE / "pair_polynomial_f32_fixed_dev_v1"
SCHEMA = "pair_polynomial_f32_independent14_v1"
PARENT = "candidate_response_rows_current_runtime_v1.py"
CANDIDATE = "candidate_response_pair_polynomial_f32_v1.py"
HELPER = "check_response_rows_adjusted_v1.py"
OLD_PANEL = "check_response_rows_panel_v1.py"
DEV_CHECKER = "check_pair_polynomial_f32_dev_v1.py"
PUBLIC = (35, 100, 150, 200, 250, 300, 400, 500)
VIRGIN = (3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
RANKS = PUBLIC + VIRGIN
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PINS = {
    PARENT: "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    CANDIDATE: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    OLD_PANEL: "6fcf38e55173a6053c51d536dc9791a0c925a84cd4164d39d159120c0d33acc6",
    DEV_CHECKER: "6ec901abff457e06fd028cd3901fd726b3e70bc9f2dcd3a41feb90530a324b02",
    "PAIR_POLYNOMIAL_F32_PANEL_V1.md": "88ee8e52c596041f5c63b9a0def51a5993f8438a77009075b04cba1aa762a289",
    "pair_polynomial_f32_fixed_dev_v1/score/result.json": "fc89efe6cd9117dcfb86a31469cfc356d61b5f70397338641d0357c99e534b81",
    "pair_polynomial_f32_fixed_dev_v1/parent/result.json": "be287320fe18095a58d8a1902a5b72b9b2f74e2feda176bfa8d25eff508ae50f",
    "pair_polynomial_f32_fixed_dev_v1/candidate/result.json": "11a66c5a85cabde9c23bca95dba81e0585d0ede999db09ac26c0d538a988a08a",
    "response_rows_panel_v1/response/result.json": "8b8433a13d2448f785f468cb67d2f793745495ffc632d76408e5a0e4f33e2de4",
    "response_rows_panel_v1/score/result.json": "a31981681305de1828c48d8b6e8830c87ed979789c7d600b949217814f479cdd",
}
OUTPUT_VALIDATION = (
    "Concrete served and materialized values are both checked after context exit; "
    "only shape/dtype and official materialization occur inside. No deferred "
    "estimator work, harness FLOP charge, cast, clamp, or other repair."
)


def load_module(name, alias):
    path = HERE / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError("Frozen helper changed: " + name)
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load inert helper: " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    return (
        load_module(HELPER, "fp32_panel_common"),
        load_module(OLD_PANEL, "fp32_panel_archived"),
        load_module(DEV_CHECKER, "fp32_panel_development"),
    )


def prerequisites(h, panel, dev):
    h.require(panel.PUBLIC == PUBLIC and panel.VIRGIN == VIRGIN and panel.RANKS == RANKS,
              "The frozen panel sequence changed")
    source_set = {name: h.binding(HERE / name) for name in (*PINS, Path(__file__).name)}
    h.require(all(source_set[name]["sha256"] == expected for name, expected in PINS.items()),
              "Source or prerequisite pin mismatch")
    baseline = panel.verify_role(h, "response")
    parent_dev = dev.verify_role(h, "parent")
    candidate_dev = dev.verify_role(h, "candidate")
    dev_score = h.load_json(DEV_ROOT / "score/result.json")
    old_score = h.load_json(HERE / "response_rows_panel_v1/score/result.json")
    h.require(dev_score["qualification_passed"] and dev_score["hashes_unchanged"]
              and dev_score["advances_to_independent_14_panel"]
              and dev_score["scientific_status"] == "dev_gate_pass", "No passing development gate")
    h.require(old_score["qualification_passed"] and old_score["hashes_unchanged"]
              and old_score["role_candidates"]["response"]["sha256"] == PINS[PARENT],
              "Archived matched truth-frame receipt is invalid")
    h.require(baseline["versions"] == parent_dev["versions"] == candidate_dev["versions"],
              "Archived and current native versions differ")
    h.require(baseline["candidate_sha256"] == parent_dev["candidate_sha256"] == PINS[PARENT]
              and candidate_dev["candidate_sha256"] == PINS[CANDIDATE], "Wrong baseline or candidate")
    for call in (0, 1):
        a, b = baseline["trials"][call], parent_dev["trials"][call]
        h.require(a["rank"] == b["rank"] == RANKS[call]
                  and a["flops"] == b["flops"]
                  and a["output_bytes_sha256"] == b["output_bytes_sha256"],
                  "Fresh development parent does not corroborate archived panel bytes/bills")
    for name, record in (("baseline", baseline), ("dev_parent", parent_dev), ("dev_candidate", candidate_dev)):
        for key, value in record["source_bindings"].items():
            source_set[f"{name}_source/{key}"] = value
    for prefix, directory in (("baseline", ARCHIVED), ("dev_parent", DEV_ROOT / "parent"),
                              ("dev_candidate", DEV_ROOT / "candidate")):
        for name in ("started.json", "result.json", "completion.json"):
            source_set[f"{prefix}_seal/{name}"] = h.binding(directory / name)
    h.check_bindings(source_set)
    return source_set, baseline, candidate_dev, old_score


def load_output(h, np, directory, record, call):
    trial = record["trials"][call]
    name = trial["output_artifact"]
    path = directory / name
    h.require(Path(record["artifacts"][name]["path"]).resolve() == path.resolve(),
              "Reference output escaped its sealed directory")
    h.require(h.digest(path) == record["artifacts"][name]["sha256"], "Output file digest differs")
    value = np.load(path, allow_pickle=False)
    h.assert_output(np, value)
    h.require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == trial["output_bytes_sha256"],
              "Served-byte digest differs")
    return value


def valid_trial(h, trial, call):
    h.require(trial["call"] == call and trial["rank"] == RANKS[call]
              and 0 < trial["flops"] <= h.BUDGET
              and 0 < trial["context_wall_s"] < 120 and 0 < trial["meter_wall_s"] < 120
              and 0 <= trial["native_residual_s"] < math.inf
              and 0 <= trial["flopscope_backend_s"] < math.inf
              and 0 <= trial["flopscope_overhead_s"] < math.inf
              and 0 < trial["peak_rss_bytes"] < h.RSS_LIMIT
              and trial["includes_first_call_initialization"] == (call == 0)
              and trial["served_materialized_byte_exact"], "Native output/budget/resource gate failed")


def acquire(h, panel, dev, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable and "anaconda3" not in sys.prefix,
              "Only OFFPY data_gate/.venv is allowed; never AP/threadpoolctl")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    h.require(nice >= 10 and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
              "Launch with nice>=10 and PYTHONDONTWRITEBYTECODE=1")
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(env == dict.fromkeys(BLAS_ENV, "1"), "Set all BLAS controls1 before imports")
    source_set, baseline, dev_candidate, _ = prerequisites(h, panel, dev)
    fixture_set, rows = panel.fixtures(h)
    h.require(fixture_set == baseline["fixture_bindings"]
              and [row["rank"] for row in rows] == list(RANKS), "Archived fixture/frame mismatch")
    versions = {key: importlib.metadata.version(key) for key in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions == baseline["versions"] == dev_candidate["versions"], "Native versions changed")
    h.require(os.getpid() not in (baseline["pid"], dev_candidate["pid"]), "A fresh candidate process is required")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, role="candidate", pid=os.getpid(), utc=h.utc(),
                   executable=sys.executable, nice=nice, versions=versions, blas_env=env,
                   sequence=list(RANKS), source_bindings=source_set, fixture_bindings=fixture_set,
                   candidate=CANDIDATE, candidate_sha256=PINS[CANDIDATE], truth_files_accessed=False,
                   instrumented=False, one_estimator_instance=True, no_warmup=True, no_reset=True,
                   initialization_once_in_call0=True, development_initialization_flops=1383913666,
                   native_wall_limit_s=120, flop_budget=h.BUDGET, rss_limit_bytes=h.RSS_LIMIT,
                   output_validation=OUTPUT_VALIDATION, baseline_reused=True)
    start_sha = h.write_json(directory / "started.json", started)
    trials, artifacts, comparisons = [], {}, []
    result = dict(schema=SCHEMA, role="candidate", pid=os.getpid(), completed=False,
                  qualification_passed=False, started=started, started_sha256=start_sha,
                  versions=versions, blas_env=env, source_bindings=source_set,
                  fixture_bindings=fixture_set, candidate=CANDIDATE, candidate_sha256=PINS[CANDIDATE],
                  truth_files_accessed=False, instrumented=False, baseline_reused=True,
                  output_validation=OUTPUT_VALIDATION, production_residual_validated=False)
    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("fixed_fp32_panel_candidate", HERE / CANDIDATE)
        h.require(spec is not None and spec.loader is not None, "Candidate import specification failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["candidate_import_wall_s"] = time.perf_counter() - t0
        h.require(h.digest(HERE / CANDIDATE) == PINS[CANDIDATE], "Candidate changed during import")
        estimator = None
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(pool["num_threads"] == 1 for pool in pools if pool.get("user_api") == "blas"),
                      "Reported BLAS thread count differs from1")
            result["threadpools"] = pools
            for call, row in enumerate(rows):
                rank = row["rank"]
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16, 1024, 1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"fp32poly-panel-{rank}")
                began = h.utc(); t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16, 1024), "Concrete served float32 output required")
                    returned = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter() - t0
                # Validate BOTH concrete values outside the estimator meter.
                original = np.asarray(served)
                out = np.asarray(returned).copy()
                h.assert_output(np, original); h.assert_output(np, out)
                same = original.tobytes(order="C") == out.tobytes(order="C")
                h.require(same, "Official materialization changed the concrete served bytes")
                name = f"call_{call:02d}_rank{rank}_output.npy"
                artifacts[name] = h.save_npy(np, directory / name, out)
                trial = dict(call=call, rank=rank, began_utc=began, ended_utc=h.utc(),
                             flops=int(ctx.flops_used), context_wall_s=elapsed,
                             meter_wall_s=float(ctx.wall_time_s),
                             flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                             flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                             native_residual_s=float(ctx.residual_wall_time_s), peak_rss_bytes=h.peak_rss_bytes(),
                             output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                             includes_first_call_initialization=call == 0, served_materialized_byte_exact=same)
                trials.append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); artifacts[bill] = h.binding(directory / bill)
                print(json.dumps(dict(role="candidate", **trial)), flush=True)
                valid_trial(h, trial, call)
                ref = load_output(h, np, ARCHIVED, baseline, call)
                error = np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64)), axis=1)
                comparison = dict(call=call, rank=rank, layer0_byte_exact=out[0].tobytes() == ref[0].tobytes(),
                                  per_layer_max_abs=error.tolist(), numerical_passed=bool((error[1:] <= 5e-6).all()))
                h.require(comparison["layer0_byte_exact"] and comparison["numerical_passed"],
                          "Fixed layer0/numerical trajectory gate failed")
                if call < 2:
                    d = load_output(h, np, DEV_ROOT / "candidate", dev_candidate, call)
                    comparison["development_output_and_bill_exact"] = (
                        out.tobytes() == d.tobytes() and trial["flops"] == dev_candidate["trials"][call]["flops"])
                    h.require(comparison["development_output_and_bill_exact"], "First two development calls changed")
                    del d
                comparisons.append(comparison)
                del weights, mlp, served, returned, original, out, ref, error
        h.check_bindings(source_set); h.check_bindings(fixture_set); h.check_bindings(artifacts)
        h.check_bindings(baseline["artifacts"]); h.check_bindings(dev_candidate["artifacts"])
        prediction_seal = dict(schema=SCHEMA, sealed_utc=h.utc(), source_bindings=source_set,
                               fixture_bindings=fixture_set, trials=trials, artifacts=artifacts,
                               comparisons=comparisons, truth_files_accessed=False,
                               candidate_sha256=PINS[CANDIDATE], baseline_result_sha256=PINS["response_rows_panel_v1/response/result.json"])
        prediction_sha = h.write_json(directory / "predictions_and_bills_sealed.json", prediction_seal)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                      compatibility_passed=True, predictions_and_bills_seal_sha256=prediction_sha)
    except BaseException:
        result.update(error=traceback.format_exc())
    result.update(trials=trials, artifacts=artifacts, comparisons=comparisons, completed_utc=h.utc())
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, role="candidate", completed_utc=h.utc(),
                 qualification_passed=result["qualification_passed"], result_sha256=result_sha, truth_files_accessed=False))
    print(json.dumps(dict(qualification_passed=result["qualification_passed"],
                          result_sha256=result_sha, error=result.get("error"))), flush=True)
    return 0 if result["qualification_passed"] else 1


def verify_acquisition(h, panel, dev):
    directory = ROOT / "candidate"
    source_set, baseline, dev_candidate, old_score = prerequisites(h, panel, dev)
    complete = h.load_json(directory / "completion.json")
    h.require(complete["schema"] == SCHEMA and complete["role"] == "candidate"
              and complete["qualification_passed"] and not complete["truth_files_accessed"]
              and complete["result_sha256"] == h.digest(directory / "result.json"), "Missing qualified candidate seal")
    record = h.load_json(directory / "result.json")
    h.require(record["schema"] == SCHEMA and record["role"] == "candidate" and record["completed"]
              and record["qualification_passed"] and record["hashes_unchanged"] and record["compatibility_passed"]
              and not record["truth_files_accessed"] and not record["instrumented"] and record["baseline_reused"]
              and record["source_bindings"] == source_set and record["candidate_sha256"] == PINS[CANDIDATE],
              "Invalid candidate acquisition")
    start = h.load_json(directory / "started.json")
    h.require(start == record["started"] and h.digest(directory / "started.json") == record["started_sha256"]
              and start["pid"] == record["pid"] and start["nice"] >= 10 and start["sequence"] == list(RANKS)
              and start["source_bindings"] == source_set and start["fixture_bindings"] == record["fixture_bindings"]
              and start["versions"] == record["versions"] == baseline["versions"] == dev_candidate["versions"]
              and start["blas_env"] == record["blas_env"] == dict.fromkeys(BLAS_ENV, "1")
              and start["output_validation"] == record["output_validation"] == OUTPUT_VALIDATION,
              "Acquisition header/frame changed")
    fixtures, rows = panel.fixtures(h)
    h.require(record["fixture_bindings"] == fixtures == baseline["fixture_bindings"]
              and [row["rank"] for row in rows] == list(RANKS), "Panel fixtures changed")
    h.require([trial["rank"] for trial in record["trials"]] == list(RANKS)
              and len(record["comparisons"]) == len(RANKS), "Incomplete fixed panel")
    expected = set()
    for call, trial in enumerate(record["trials"]):
        valid_trial(h, trial, call)
        name = f"call_{call:02d}_rank{RANKS[call]}_output.npy"
        bill = f"call_{call:02d}_bill.json"; expected.update((name, bill))
        h.require(trial["output_artifact"] == name and h.load_json(directory / bill) == trial, "Per-call bill changed")
        comparison = record["comparisons"][call]
        h.require(comparison["call"] == call and comparison["rank"] == RANKS[call]
                  and comparison["layer0_byte_exact"] and comparison["numerical_passed"]
                  and len(comparison["per_layer_max_abs"]) == 16
                  and all(0 <= value <= 5e-6 for value in comparison["per_layer_max_abs"][1:]),
                  "Trajectory qualification missing")
        if call < 2:
            h.require(comparison["development_output_and_bill_exact"]
                      and trial["flops"] == dev_candidate["trials"][call]["flops"]
                      and trial["output_bytes_sha256"] == dev_candidate["trials"][call]["output_bytes_sha256"],
                      "Development equality changed")
    h.require(set(record["artifacts"]) == expected, "All28 outputs/bills must be sealed")
    for name, item in record["artifacts"].items():
        h.require(Path(item["path"]).resolve() == (directory / name).resolve(), "Artifact path escaped panel")
    h.check_bindings(record["artifacts"]); h.check_bindings(fixtures)
    path = directory / "predictions_and_bills_sealed.json"
    h.require(h.digest(path) == record["predictions_and_bills_seal_sha256"], "Prediction seal changed")
    seal = h.load_json(path)
    h.require(seal["schema"] == SCHEMA and not seal["truth_files_accessed"]
              and seal["source_bindings"] == source_set and seal["fixture_bindings"] == fixtures
              and seal["trials"] == record["trials"] and seal["artifacts"] == record["artifacts"]
              and seal["comparisons"] == record["comparisons"], "Incomplete prediction seal")
    return record, baseline, old_score


def score(h, panel, dev, directory):
    candidate, baseline, old_score = verify_acquisition(h, panel, dev)
    import numpy as np
    h.require(importlib.metadata.version("numpy") == candidate["versions"]["numpy"], "Scoring NumPy changed")
    records = {"parent": baseline, "candidate": candidate}
    paths = {"parent": ARCHIVED, "candidate": ROOT / "candidate"}
    predictions = {role: [load_output(h, np, paths[role], record, call) for call in range(len(RANKS))]
                   for role, record in records.items()}
    role_seals = {f"{role}/{name}": h.binding(paths[role] / name) for role in records
                  for name in ("started.json", "result.json", "completion.json")}
    role_seals["candidate/predictions_and_bills_sealed.json"] = h.binding(ROOT / "candidate/predictions_and_bills_sealed.json")
    before = h.write_json(directory / "both_roles_verified_before_truth.json", dict(
        schema=SCHEMA, utc=h.utc(), role_seals=role_seals, source_bindings=candidate["source_bindings"],
        fixture_bindings=candidate["fixture_bindings"], ranks=list(RANKS), truth_files_accessed=False))
    truth_time = h.utc()
    h.write_json(directory / "truth_access_started.json", dict(utc=truth_time, prior_seal_sha256=before))
    truth_bindings, per_role = {}, {role: [] for role in records}
    for call, rank in enumerate(RANKS):
        path = panel.G19 / (f"mps_truth_rank{rank:04d}_indep4M.npz" if rank in PUBLIC else f"fresh_{rank}_truth4M.npz")
        truth_bindings[str(rank)] = h.binding(path)
        expected = old_score["truth_bindings"][str(rank)]
        h.require(truth_bindings[str(rank)] == expected, "Independent4M truth frame differs from archived baseline")
        with np.load(path, allow_pickle=False) as archive:
            mean = archive["mc_post"][15].astype(np.float64)
            se = archive["mc_se"][15].astype(np.float64)
        h.require(mean.shape == se.shape == (1024,) and bool(np.isfinite(mean).all())
                  and bool(np.isfinite(se).all()) and bool((se >= 0).all()), "Invalid fixed panel truth")
        noise = float(np.mean(se * se))
        for role, record in records.items():
            trial = record["trials"][call]
            raw = float(np.mean((predictions[role][call][15].astype(np.float64) - mean) ** 2))
            nc = raw - noise
            multiplier = max(.1, trial["flops"] / h.BUDGET)
            item = dict(call=call, rank=rank, flops=trial["flops"], multiplier=multiplier,
                        includes_first_call_initialization=call == 0, raw_observed=raw,
                        mc_se_squared_mean=noise, raw_noise_corrected=nc,
                        adjusted_observed=raw * multiplier, adjusted_noise_corrected=nc * multiplier,
                        output_bytes_sha256=trial["output_bytes_sha256"])
            if role == "parent":
                old = old_score["per_role"]["response"][call]
                h.require(old["rank"] == rank and all(item[key] == old[key] for key in (
                    "flops", "raw_observed", "mc_se_squared_mean", "raw_noise_corrected",
                    "multiplier", "adjusted_observed", "adjusted_noise_corrected")),
                    "Recomputed parent score differs from the sealed same-frame baseline")
            per_role[role].append(item)
    summaries = {}
    for role, rows in per_role.items():
        summaries[role] = {}
        for name, ranks in (("public8", PUBLIC), ("virgin6", VIRGIN), ("combined14", RANKS)):
            subset = [row for row in rows if row["rank"] in ranks]
            fields = ("raw_observed", "raw_noise_corrected", "mc_se_squared_mean", "flops",
                      "adjusted_observed", "adjusted_noise_corrected")
            summaries[role][name] = dict(n=len(subset), **{
                "mean_" + key: sum(row[key] for row in subset) / len(subset) for key in fields})
    ratios, gates = {}, {}
    for group in ("public8", "virgin6", "combined14"):
        denominator = summaries["parent"][group]["mean_adjusted_noise_corrected"]
        h.require(math.isfinite(denominator) and denominator > 0, "Invalid baseline adjusted denominator")
        ratios[group] = summaries["candidate"][group]["mean_adjusted_noise_corrected"] / denominator
        h.require(math.isfinite(ratios[group]), "Nonfinite adjusted ratio")
        if group != "combined14":
            gates[group] = ratios[group] <= .995
    h.check_bindings(role_seals); h.check_bindings(truth_bindings)
    for record in records.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    retain = all(gates.values())
    result = dict(schema=SCHEMA, qualification_passed=True, hashes_unchanged=True, completed_utc=h.utc(),
                  baseline_reused=True, baseline_recomputed_exact=True, role_seals=role_seals,
                  source_bindings=candidate["source_bindings"], truth_bindings=truth_bindings,
                  both_roles_verified_before_truth_sha256=before, truth_access_started_utc=truth_time,
                  per_role=per_role, summaries=summaries, adjusted_candidate_over_parent=ratios,
                  group_gain_gate=gates, retain_local_component=retain,
                  scientific_status="panel_component_pass" if retain else "fixed_partition_closed",
                  aggregation="Mean per-net (MSE-mean(mc_se15^2))*max(.1,actual completeF/2^41); no clipping; initialization once in call0",
                  frame="Existing independent4M public8/virgin6 panel; not unused heldout or official scoring",
                  native_timing_only=True, archived_runtime_not_contemporaneous=True,
                  production_residual_validated=False, official_score=False, milestone_achieved=False,
                  goal_achieved=False, candidate=CANDIDATE, candidate_sha256=PINS[CANDIDATE])
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, completed_utc=h.utc(), qualification_passed=True,
                 result_sha256=result_sha, retain_local_component=retain))
    print(json.dumps(dict(summaries=summaries, adjusted_candidate_over_parent=ratios,
                          retain_local_component=retain, result_sha256=result_sha)), flush=True)
    return 0  # Scientific continuation must use retain_local_component, never exit status alone.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--acquire", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    h, panel, dev = modules()
    if args.acquire:
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Begin with a new fixed acquisition")
    directory = ROOT / ("candidate" if args.acquire else "score")
    directory.mkdir(exist_ok=False)
    try:
        return acquire(h, panel, dev, directory) if args.acquire else score(h, panel, dev, directory)
    except BaseException:
        h.write_json(directory / "preflight_or_score_failure.json", dict(
            schema=SCHEMA, utc=h.utc(), qualification_passed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
