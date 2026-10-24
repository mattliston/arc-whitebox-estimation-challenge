"""INERT exact terminal-cleanup panel: --acquire, then separate --score.

The only candidate is the source-only build qualified by the fixed synthetic
gate. Reuse the sealed current0b84 parent panel, not another head/frame. One
fresh OFFPY worker serves14 fixed MLPs with initialization once. No fitting,
diagnostic wrapper, array repair, threshold, retry, or submission is provided.
All candidate outputs/bills are sealed before independent4M truth access.
The three required SHA arguments pin provenance, never numerical choices.
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
NEW = HERE.parent
ROOT = HERE / "panel"
ARCHIVED = NEW / "pair_polynomial_f32_panel_v1/candidate"
ARCHIVED_SCORE = NEW / "pair_polynomial_f32_panel_v1/score"
SCHEMA = "terminal_dead_consumers_fixed14_v1"
PUBLIC = (35, 100, 150, 200, 250, 300, 400, 500)
VIRGIN = (3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
RANKS = PUBLIC + VIRGIN
PARENT = "candidate_response_pair_polynomial_f32_v1.py"
PARENT_SHA = "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96"
PREVIOUS = "check_pair_polynomial_f32_panel_v1.py"
PREVIOUS_SHA = "9f17e45aaa3e968f5eaca236115e6d060bb2dfc70e450478e1282e5d4ed7588f"
PARENT_RESULT_SHA = "72b41dbc14cd9b98f823b8f3b0aa9116839d5181264a8b597c8f3f47b835e595"
PARENT_SCORE_SHA = "7fc755056d504aa827eda0a2d6141bc9f64cf2eb3a1c0739fdd6b7bd4f34c5e3"
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
OUTPUT_VALIDATION = (
    "Original concrete served and official materialized values are both checked "
    "after context exit. Shape/dtype assertion and official materialization occur "
    "inside. No deferred estimator work, cast, clamp, or repair; harness scans "
    "and reference-byte checks are outside the estimator FLOP boundary."
)


def sha_arg(value):
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("Expected a lowercase SHA256")
    return value


def modules():
    path = NEW / PREVIOUS
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREVIOUS_SHA:
        raise RuntimeError("Frozen prior panel helper changed")
    spec = importlib.util.spec_from_file_location("terminal_dead_parent_panel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load inert prior panel verifier")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    h, panel, dev = previous.modules()
    h.require(h.digest(path) == PREVIOUS_SHA, "Prior verifier changed during import")
    return h, panel, dev, previous


def prerequisites(h, panel, dev, previous, args):
    h.require(previous.PUBLIC == panel.PUBLIC == PUBLIC
              and previous.VIRGIN == panel.VIRGIN == VIRGIN
              and previous.RANKS == panel.RANKS == RANKS, "Fixed14 sequence changed")
    pins = {"build_sha256": args.build_sha256, "synthetic_sha256": args.synthetic_sha256,
            "protocol_sha256": args.protocol_sha256}
    files = {"build": HERE / "build.json", "synthetic": HERE / "synthetic/result.json",
             "synthetic_completion": HERE / "synthetic/completion.json",
             "protocol": HERE / "PROTOCOL.md", "builder": HERE / "source.py",
             "synthetic_checker": HERE / "check_synthetic.py", "candidate": HERE / "estimator.py",
             "panel_checker": Path(__file__), "parent": NEW / PARENT,
             "prior_panel_checker": NEW / PREVIOUS,
             "parent_score": ARCHIVED_SCORE / "result.json",
             "parent_score_completion": ARCHIVED_SCORE / "completion.json"}
    sources = {name: h.binding(path) for name, path in files.items()}
    h.require(sources["build"]["sha256"] == args.build_sha256
              and sources["synthetic"]["sha256"] == args.synthetic_sha256
              and sources["protocol"]["sha256"] == args.protocol_sha256
              and sources["parent"]["sha256"] == PARENT_SHA
              and sources["prior_panel_checker"]["sha256"] == PREVIOUS_SHA
              and sources["parent_score"]["sha256"] == PARENT_SCORE_SHA,
              "Pinned source/build/gate/parent mismatch")
    build = h.load_json(files["build"])
    gate = h.load_json(files["synthetic"])
    complete = h.load_json(files["synthetic_completion"])
    h.require(complete["passed"] and complete["result_sha256"] == args.synthetic_sha256,
              "Synthetic completion does not bind the passing receipt")
    h.require(build["passed"] and build["executed"] is False
              and build["inverse_source_byte_exact"] and build["all_other_definitions_byte_exact"]
              and build["imports_and_learned_payloads_unchanged"], "Mechanical build did not qualify")
    h.require(gate["passed"] and gate["sources_unchanged"], "No fixed passing synthetic gate")
    h.require(build["candidate_sha256"] == sources["candidate"]["sha256"]
              and build["parent_sha256"] == gate["parent_sha256"] == PARENT_SHA
              and build["source_sha256"] == gate["source_sha256"] == sources["builder"]["sha256"]
              and build["protocol_sha256"] == gate["protocol_sha256"] == args.protocol_sha256
              and build["synthetic_result_sha256"] == args.synthetic_sha256
              and gate["checker_sha256"] == sources["synthetic_checker"]["sha256"]
              and build["terminal_function_sha256"] == gate["terminal_function_sha256"]
              and Path(build["candidate"]).resolve() == files["candidate"].resolve()
              and Path(build["parent"]).resolve() == files["parent"].resolve(),
              "Build and synthetic function/source ownership differ")
    # This verifier reads only sealed JSON and hashes artifacts; it never runs
    # either estimator or opens a truth array. Its entire legacy chain is pinned.
    baseline, _, _ = previous.verify_acquisition(h, panel, dev)
    h.require(h.digest(ARCHIVED / "result.json") == PARENT_RESULT_SHA
              and baseline["candidate_sha256"] == PARENT_SHA,
              "The reused baseline is not current0b84")
    old_score = h.load_json(files["parent_score"])
    score_complete = h.load_json(files["parent_score_completion"])
    h.require(score_complete["result_sha256"] == PARENT_SCORE_SHA
              and score_complete["qualification_passed"] and score_complete["retain_local_component"]
              and old_score["qualification_passed"] and old_score["hashes_unchanged"]
              and old_score["retain_local_component"] and old_score["candidate_sha256"] == PARENT_SHA,
              "Current parent score/frame is not the retained qualified panel")
    sources.update({"parent_source/" + key: value for key, value in baseline["source_bindings"].items()})
    for name in ("started.json", "result.json", "completion.json", "predictions_and_bills_sealed.json"):
        sources["parent_seal/" + name] = h.binding(ARCHIVED / name)
    h.check_bindings(sources)
    h.check_bindings(baseline["artifacts"])
    return sources, baseline, old_score, build, pins


def comparisons_pass(h, comparison, trial, parent_trial, call):
    h.require(comparison["call"] == trial["call"] == call
              and comparison["rank"] == trial["rank"] == parent_trial["rank"] == RANKS[call]
              and comparison["all16_byte_exact"]
              and comparison["strict_complete_cost_saving"]
              and len(comparison["per_layer_byte_exact"]) == 16
              and all(comparison["per_layer_byte_exact"])
              and trial["output_bytes_sha256"] == parent_trial["output_bytes_sha256"]
              and comparison["parent_flops"] == parent_trial["flops"]
              and comparison["saved_flops"] == parent_trial["flops"] - trial["flops"] > 0,
              "Exact all16 output or strictly lower complete-cost gate failed")


def acquire(h, panel, dev, previous, args, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable and "anaconda3" not in sys.prefix,
              "Use OFFPY data_gate/.venv, never AP/threadpoolctl")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(nice >= 10 and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
              and env == dict.fromkeys(BLAS_ENV, "1"),
              "Launch nice10, PYTHONDONTWRITEBYTECODE=1 and all BLAS controls1 before imports")
    sources, baseline, _, build, pins = prerequisites(h, panel, dev, previous, args)
    fixtures, rows = panel.fixtures(h)
    h.require(fixtures == baseline["fixture_bindings"]
              and [row["rank"] for row in rows] == list(RANKS), "Wrong canonical weights/seeds")
    versions = {name: importlib.metadata.version(name)
                for name in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions == baseline["versions"] and versions["flopscope"] == "0.12.0"
              and versions["whestbench"] == "0.16.0" and os.getpid() != baseline["pid"],
              "Native environment or fresh-process frame mismatch")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, role="candidate", utc=h.utc(), pid=os.getpid(),
                   executable=sys.executable, nice=nice, versions=versions, blas_env=env,
                   source_bindings=sources, fixture_bindings=fixtures, provenance_pins=pins,
                   candidate_sha256=build["candidate_sha256"], parent_sha256=PARENT_SHA,
                   baseline_result_sha256=PARENT_RESULT_SHA, sequence=list(RANKS),
                   one_estimator_instance=True, initialization_once_in_call0=True,
                   no_warmup=True, no_reset=True, instrumented=False, truth_files_accessed=False,
                   source_only_build_verified=True, synthetic_gate_verified=True,
                   archived_parent_reused=True, output_validation=OUTPUT_VALIDATION,
                   flop_budget=h.BUDGET, wall_limit_s=120, rss_limit_bytes=h.RSS_LIMIT,
                   production_residual_validated=False)
    start_sha = h.write_json(directory / "started.json", started)
    trials, artifacts, comparisons = [], {}, []
    result = dict(schema=SCHEMA, role="candidate", completed=False, qualification_passed=False,
                  pid=os.getpid(), started=started, started_sha256=start_sha,
                  versions=versions, blas_env=env, source_bindings=sources,
                  fixture_bindings=fixtures, provenance_pins=pins,
                  candidate_sha256=build["candidate_sha256"], parent_sha256=PARENT_SHA,
                  baseline_result_sha256=PARENT_RESULT_SHA, instrumented=False,
                  truth_files_accessed=False, output_validation=OUTPUT_VALIDATION,
                  production_residual_validated=False, archived_runtime_not_contemporaneous=True)
    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("terminal_dead_unwrapped_candidate", HERE / "estimator.py")
        h.require(spec is not None and spec.loader is not None, "Candidate import specification failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["candidate_import_wall_s"] = time.perf_counter() - t0
        h.require(h.digest(HERE / "estimator.py") == build["candidate_sha256"], "Candidate changed during import")
        estimator = None
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                      "Reported BLAS thread count differs from1")
            result["threadpools"] = pools
            for call, row in enumerate(rows):
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16, 1024, 1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid canonical Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"terminal-dead-panel-{row['rank']}")
                began = h.utc(); t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16, 1024), "Require concrete float32(16,1024), without repair")
                    materialized = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter() - t0
                original = np.asarray(served)
                out = np.asarray(materialized).copy()
                h.assert_output(np, original); h.assert_output(np, out)
                exact = original.tobytes(order="C") == out.tobytes(order="C")
                h.require(exact, "Official materialization changed served bytes")
                name = f"call_{call:02d}_rank{row['rank']}_output.npy"
                artifacts[name] = h.save_npy(np, directory / name, out)
                trial = dict(call=call, rank=row["rank"], began_utc=began, ended_utc=h.utc(),
                             flops=int(ctx.flops_used), context_wall_s=elapsed, meter_wall_s=float(ctx.wall_time_s),
                             flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                             flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                             native_residual_s=float(ctx.residual_wall_time_s), peak_rss_bytes=h.peak_rss_bytes(),
                             output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                             includes_first_call_initialization=call == 0, served_materialized_byte_exact=exact)
                trials.append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); artifacts[bill] = h.binding(directory / bill)
                previous.valid_trial(h, trial, call)
                ref = previous.load_output(h, np, ARCHIVED, baseline, call)
                comparison = dict(call=call, rank=row["rank"], all16_byte_exact=out.tobytes() == ref.tobytes(),
                                  per_layer_byte_exact=[out[l].tobytes() == ref[l].tobytes() for l in range(16)],
                                  parent_flops=baseline["trials"][call]["flops"],
                                  saved_flops=baseline["trials"][call]["flops"] - trial["flops"],
                                  strict_complete_cost_saving=trial["flops"] < baseline["trials"][call]["flops"])
                comparisons.append(comparison)
                comparisons_pass(h, comparison, trial, baseline["trials"][call], call)
                print(json.dumps(dict(role="candidate", **trial, comparison=comparison)), flush=True)
                del weights, mlp, served, materialized, original, out, ref
        for items in (sources, fixtures, artifacts, baseline["artifacts"]):
            h.check_bindings(items)
        seal = dict(schema=SCHEMA, sealed_utc=h.utc(), source_bindings=sources, fixture_bindings=fixtures,
                    candidate_sha256=build["candidate_sha256"], baseline_result_sha256=PARENT_RESULT_SHA,
                    provenance_pins=pins, trials=trials, artifacts=artifacts, comparisons=comparisons,
                    truth_files_accessed=False)
        sealed_sha = h.write_json(directory / "predictions_and_bills_sealed.json", seal)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                      exact_outputs_and_lower_complete_cost=True, predictions_and_bills_seal_sha256=sealed_sha)
    except BaseException:
        result["error"] = traceback.format_exc()
    result.update(trials=trials, artifacts=artifacts, comparisons=comparisons, completed_utc=h.utc())
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, role="candidate", completed_utc=h.utc(),
                 qualification_passed=result["qualification_passed"], result_sha256=result_sha, truth_files_accessed=False))
    print(json.dumps(dict(qualification_passed=result["qualification_passed"], result_sha256=result_sha,
                          error=result.get("error"))), flush=True)
    return 0 if result["qualification_passed"] else 1


def verify_acquisition(h, panel, dev, previous, args):
    sources, baseline, old_score, build, pins = prerequisites(h, panel, dev, previous, args)
    directory = ROOT / "candidate"
    complete = h.load_json(directory / "completion.json")
    h.require(complete["schema"] == SCHEMA and complete["role"] == "candidate"
              and complete["qualification_passed"] and not complete["truth_files_accessed"]
              and complete["result_sha256"] == h.digest(directory / "result.json"), "Missing qualified acquisition seal")
    record = h.load_json(directory / "result.json")
    h.require(record["schema"] == SCHEMA and record["role"] == "candidate" and record["completed"]
              and record["qualification_passed"] and record["hashes_unchanged"]
              and record["exact_outputs_and_lower_complete_cost"] and not record["truth_files_accessed"]
              and not record["instrumented"] and record["provenance_pins"] == pins
              and record["source_bindings"] == sources and record["candidate_sha256"] == build["candidate_sha256"]
              and record["parent_sha256"] == PARENT_SHA and record["baseline_result_sha256"] == PARENT_RESULT_SHA,
              "Invalid exact terminal candidate acquisition")
    start = h.load_json(directory / "started.json")
    h.require(start == record["started"] and h.digest(directory / "started.json") == record["started_sha256"]
              and start["pid"] == record["pid"] and start["nice"] >= 10 and start["sequence"] == list(RANKS)
              and start["source_bindings"] == sources and start["provenance_pins"] == pins
              and start["versions"] == record["versions"] == baseline["versions"]
              and start["blas_env"] == record["blas_env"] == dict.fromkeys(BLAS_ENV, "1")
              and start["one_estimator_instance"] and start["initialization_once_in_call0"]
              and start["no_warmup"] and start["no_reset"] and not start["instrumented"]
              and not start["truth_files_accessed"]
              and start["output_validation"] == record["output_validation"] == OUTPUT_VALIDATION,
              "Acquisition header/initialization/source boundary changed")
    fixtures, rows = panel.fixtures(h)
    h.require(record["fixture_bindings"] == start["fixture_bindings"] == fixtures == baseline["fixture_bindings"]
              and [row["rank"] for row in rows] == list(RANKS)
              and [t["rank"] for t in record["trials"]] == list(RANKS)
              and len(record["comparisons"]) == len(RANKS), "Incomplete or changed14-net frame")
    expected = set()
    for call, trial in enumerate(record["trials"]):
        previous.valid_trial(h, trial, call)
        name = f"call_{call:02d}_rank{RANKS[call]}_output.npy"
        bill = f"call_{call:02d}_bill.json"; expected.update((name, bill))
        h.require(trial["output_artifact"] == name and h.load_json(directory / bill) == trial, "Per-call bill changed")
        comparisons_pass(h, record["comparisons"][call], trial, baseline["trials"][call], call)
    h.require(set(record["artifacts"]) == expected, "All28 output/bill artifacts must be sealed")
    for name, item in record["artifacts"].items():
        h.require(Path(item["path"]).resolve() == (directory / name).resolve(), "Artifact escaped fixed panel")
    h.check_bindings(record["artifacts"]); h.check_bindings(fixtures)
    path = directory / "predictions_and_bills_sealed.json"
    h.require(h.digest(path) == record["predictions_and_bills_seal_sha256"], "Prediction seal changed")
    seal = h.load_json(path)
    h.require(seal["schema"] == SCHEMA and not seal["truth_files_accessed"]
              and seal["source_bindings"] == sources and seal["fixture_bindings"] == fixtures
              and seal["provenance_pins"] == pins and seal["trials"] == record["trials"]
              and seal["artifacts"] == record["artifacts"] and seal["comparisons"] == record["comparisons"]
              and seal["candidate_sha256"] == build["candidate_sha256"]
              and seal["baseline_result_sha256"] == PARENT_RESULT_SHA, "Prediction/bill seal is incomplete")
    return record, baseline, old_score


def score(h, panel, dev, previous, args, directory):
    candidate, baseline, old_score = verify_acquisition(h, panel, dev, previous, args)
    import numpy as np
    h.require(importlib.metadata.version("numpy") == candidate["versions"]["numpy"], "Scoring NumPy changed")
    records = {"parent": baseline, "candidate": candidate}
    paths = {"parent": ARCHIVED, "candidate": ROOT / "candidate"}
    predictions = {role: [previous.load_output(h, np, paths[role], record, call) for call in range(len(RANKS))]
                   for role, record in records.items()}
    for call in range(len(RANKS)):
        h.require(predictions["parent"][call].tobytes() == predictions["candidate"][call].tobytes(),
                  "Actual sealed bytes differ before scoring")
    role_seals = {f"{role}/{name}": h.binding(paths[role] / name) for role in records
                  for name in ("started.json", "result.json", "completion.json", "predictions_and_bills_sealed.json")}
    before = h.write_json(directory / "both_roles_verified_before_truth.json", dict(
        schema=SCHEMA, utc=h.utc(), role_seals=role_seals, source_bindings=candidate["source_bindings"],
        fixture_bindings=candidate["fixture_bindings"], ranks=list(RANKS), truth_files_accessed=False))
    truth_time = h.utc()
    h.write_json(directory / "truth_access_started.json", dict(utc=truth_time, prior_seal_sha256=before))
    truth_bindings, per_role = {}, {role: [] for role in records}
    for call, rank in enumerate(RANKS):
        path = panel.G19 / (f"mps_truth_rank{rank:04d}_indep4M.npz" if rank in PUBLIC else f"fresh_{rank}_truth4M.npz")
        truth_bindings[str(rank)] = h.binding(path)
        h.require(truth_bindings[str(rank)] == old_score["truth_bindings"][str(rank)], "Independent4M truth frame changed")
        with np.load(path, allow_pickle=False) as archive:
            mean = archive["mc_post"][15].astype(np.float64)
            se = archive["mc_se"][15].astype(np.float64)
        h.require(mean.shape == se.shape == (1024,) and bool(np.isfinite(mean).all())
                  and bool(np.isfinite(se).all()) and bool((se >= 0).all()), "Invalid independent4M truth")
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
                old = old_score["per_role"]["candidate"][call]
                h.require(old["rank"] == rank and all(item[key] == old[key] for key in (
                    "flops", "raw_observed", "mc_se_squared_mean", "raw_noise_corrected",
                    "multiplier", "adjusted_observed", "adjusted_noise_corrected")),
                    "Recomputed current parent differs from its authenticated score")
            else:
                h.require(raw == per_role["parent"][call]["raw_observed"], "Exact outputs produced unequal raw loss")
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
        h.require(math.isfinite(denominator) and denominator > 0, "Invalid parent adjusted denominator")
        ratios[group] = summaries["candidate"][group]["mean_adjusted_noise_corrected"] / denominator
        h.require(math.isfinite(ratios[group]), "Nonfinite adjusted ratio")
        if group != "combined14":
            gates[group] = ratios[group] < 1.0
    h.check_bindings(role_seals); h.check_bindings(truth_bindings)
    for record in records.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    retain = all(gates.values())
    result = dict(schema=SCHEMA, completed=True, qualification_passed=True, hashes_unchanged=True,
                  completed_utc=h.utc(), source_bindings=candidate["source_bindings"],
                  provenance_pins=candidate["provenance_pins"], role_seals=role_seals,
                  truth_bindings=truth_bindings, both_roles_verified_before_truth_sha256=before,
                  truth_access_started_utc=truth_time, per_role=per_role, summaries=summaries,
                  adjusted_candidate_over_parent=ratios, group_strict_gain_gate=gates,
                  exact_all16_bytes_all14=True, strictly_lower_complete_flops_all14=True,
                  retain_local_component=retain, baseline_recomputed_exact=True,
                  scientific_status="exact_component_pass" if retain else "group_score_gate_fail",
                  candidate_sha256=candidate["candidate_sha256"], parent_sha256=PARENT_SHA,
                  aggregation="Mean per-net (MSE-mean(mc_se15^2))*max(.1,actual completeF/2^41); no clipping; init once in call0",
                  frame="Same existing independent4M public8/virgin6; not unused holdout or official score",
                  archived_runtime_not_contemporaneous=True, production_residual_validated=False,
                  official_score=False, milestone_achieved=False, goal_achieved=False)
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, completed_utc=h.utc(),
                 qualification_passed=True, result_sha256=result_sha, retain_local_component=retain))
    print(json.dumps(dict(summaries=summaries, adjusted_candidate_over_parent=ratios,
                          retain_local_component=retain, result_sha256=result_sha)), flush=True)
    return 0 if retain else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--acquire", action="store_true")
    modes.add_argument("--score", action="store_true")
    for name in ("build", "synthetic", "protocol"):
        parser.add_argument("--" + name + "-sha256", type=sha_arg)
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    if any(getattr(args, name + "_sha256") is None for name in ("build", "synthetic", "protocol")):
        parser.error("Execution requires all three provenance SHA256 arguments")
    h, panel, dev, previous = modules()
    if args.acquire:
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Start with a new fixed acquisition")
    directory = ROOT / ("candidate" if args.acquire else "score")
    directory.mkdir(exist_ok=False)
    try:
        return (acquire(h, panel, dev, previous, args, directory) if args.acquire
                else score(h, panel, dev, previous, args, directory))
    except BaseException:
        h.write_json(directory / "preflight_or_score_failure.json", dict(
            schema=SCHEMA, utc=h.utc(), qualification_passed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
