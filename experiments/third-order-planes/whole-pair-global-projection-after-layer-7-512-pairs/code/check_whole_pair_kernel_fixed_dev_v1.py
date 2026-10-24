"""INERT fixed post7 pair-projection qualification and development score.

Explicit fresh-process roles: parent_nullref, null, parent, candidate;
then --score. No parameters, seeds, fixtures, model training or variants.
No acquisition opens or hashes a truth file. Actual score roles are
unwrapped, with the same instance/import/initialization schedule.
"""
from __future__ import annotations

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
ROOT = HERE / "whole_pair_kernel_fixed_dev_v1"
HELPER = HERE / "check_response_rows_adjusted_v1.py"
HELPER_SHA = "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304"
PARENT = "candidate_response_rows_current_runtime_v1.py"
CANDIDATE = "candidate_whole_pair_kernel_post7_k512_v1.py"
ROLES = ("parent_nullref", "null", "parent", "candidate")
RANKS = (35, 100, 35)
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SCHEMA = "whole_pair_kernel_fixed_dev_v1"
PINS = {
    PARENT: "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    CANDIDATE: "41663c0b58e313bc89a13a425c2e22e29a0e9b2bde740580a2dd846d65eb1b97",
    "whole_pair_kernel_core_v1.py": "86c51be6d15f8b1c8bce463a4fb649c4974a339b5bebb20b7592669075620654",
    "whole_pair_kernel_tiny_v2.json": "d8f65494cf1d0c8d49ba68ee8b346721a821210f1e617089c8eb87422b98df27",
    "build_whole_pair_kernel_candidate_v2.py": "d393362a5c905cac37f57726c01d6f459b1a0fc7f982eff93afdfc7facf73f96",
    "whole_pair_kernel_candidate_source_v1.json": "e594b6f4bb181a7d5d7528162ed85b4ec3a9931732922d61cdf44297641eea5c",
    "WHOLE_PAIR_KERNEL_FIXED_PILOT_PROTOCOL_V1.md": "b6b08db3cf5b203383b3f046bc76e40a91c9cac3742132caf4e96d304588c08d",
    "WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md": "82e2e3b90bfc9e4172b7b6ef18adb53265d7cfb6fd5e13a8e7b04884629e3161",
    HELPER.name: HELPER_SHA,
}


def load_helper():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Frozen pure-helper source mismatch")
    spec = importlib.util.spec_from_file_location("whole_pair_pure_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require(module.digest(HELPER) == HELPER_SHA, "Helper changed during import")
    return module


def sources(h):
    result = {name: h.binding(HERE / name) for name in PINS}
    for name, value in PINS.items():
        h.require(result[name]["sha256"] == value, "Frozen source mismatch: " + name)
    for name in (Path(__file__).name, "WHOLE_PAIR_KERNEL_ROOT_ADMISSION_V1.md"):
        result[name] = h.binding(HERE / name)
    proof = h.load_json(HERE / "whole_pair_kernel_candidate_source_v1.json")
    h.require(proof["candidate_sha256"] == PINS[CANDIDATE]
              and proof["inverse_transform_byte_exact"] and proof["head_payloads_byte_identical"]
              and proof["frozen_response_heads_all16"] and proof["final_Q_unreachable"],
              "Source admission prerequisite failed")
    tiny = h.load_json(HERE / "whole_pair_kernel_tiny_v2.json")
    h.require(tiny["passed"] and tiny["seals_unchanged"] and tiny["failure"] is None,
              "Tiny numerical prerequisite failed")
    h.check_bindings(result)
    return result


def instrumented(role):
    return role in ("parent_nullref", "null")


def candidate_name(role):
    return CANDIDATE if role in ("null", "candidate") else PARENT


def verify_role(h, role):
    directory = ROOT / role
    complete = h.load_json(directory / "completion.json")
    h.require(complete["schema"] == SCHEMA and complete["role"] == role
              and complete["qualification_passed"] and not complete["truth_files_accessed"],
              "Prerequisite role did not qualify: " + role)
    h.require(h.digest(directory / "result.json") == complete["result_sha256"], "Result seal changed")
    result = h.load_json(directory / "result.json")
    h.require(result["completed"] and result["qualification_passed"] and result["hashes_unchanged"]
              and result["schema"] == SCHEMA and result["role"] == role
              and not result["truth_files_accessed"], "Invalid acquisition result")
    h.require(result["source_bindings"] == sources(h), "Source frame changed")
    h.check_bindings(result["fixture_bindings"])
    h.check_bindings(result["artifacts"])
    h.require(h.digest(directory / "started.json") == result["started_sha256"], "Start seal changed")
    started = h.load_json(directory / "started.json")
    h.require(started == result["started"] and started["pid"] == result["pid"]
              and started["source_bindings"] == result["source_bindings"]
              and started["fixture_bindings"] == result["fixture_bindings"]
              and started["nice"] >= 10 and started["sequence"] == list(RANKS),
              "Wrong role provenance")
    h.require(result["versions"] == started["versions"]
              and result["versions"]["flopscope"] == "0.12.0"
              and result["versions"]["whestbench"] == "0.16.0"
              and result["blas_env"] == dict.fromkeys(BLAS_ENV, "1"), "Wrong native frame")
    h.require(result["candidate"] == candidate_name(role)
              and result["candidate_sha256"] == PINS[candidate_name(role)]
              and result["instrumented"] == instrumented(role)
              and result["projection_enabled"] == (role == "candidate")
              and result["repeat_byte_identical"], "Candidate identity or repeat mismatch")
    h.require([t["rank"] for t in result["trials"]] == list(RANKS), "Wrong fixed sequence")
    expected = set()
    for call, trial in enumerate(result["trials"]):
        name = f"call_{call:02d}_rank{RANKS[call]}_output.npy"
        bill = f"call_{call:02d}_bill.json"
        expected.update((name, bill))
        if instrumented(role):
            expected.add(f"call_{call:02d}_rank{RANKS[call]}_diagnostics.npz")
        h.require(trial["call"] == call and trial["output_artifact"] == name
                  and h.load_json(directory / bill) == trial
                  and 0 < trial["flops"] <= h.BUDGET and trial["context_wall_s"] < 120
                  and trial["peak_rss_bytes"] < h.RSS_LIMIT
                  and 0 <= trial["native_residual_s"] < float("inf")
                  and 0 < trial["meter_wall_s"] < 120
                  and 0 <= trial["flopscope_backend_s"] < float("inf")
                  and 0 <= trial["flopscope_overhead_s"] < float("inf")
                  and trial["instrumented"] == instrumented(role)
                  and trial["includes_first_call_initialization"] == (call == 0),
                  "Invalid completed native call")
    h.require(set(result["artifacts"]) == expected, "Incomplete artifact census")
    for name, item in result["artifacts"].items():
        h.require(Path(item["path"]).resolve() == (directory / name).resolve(), "Artifact escaped role")
    if role != "parent_nullref":
        h.require(result["parity_passed"], "Role compatibility did not qualify")
    return result


def prerequisite_roles(h, role):
    records = {r: verify_role(h, r) for r in ROLES[:ROLES.index(role)]}
    for record in records.values():
        h.require(record["pid"] != os.getpid(), "Each role requires a fresh process")
    return records


def load_outputs(h, np, role, record):
    arrays = []
    for trial in record["trials"]:
        value = np.load(ROOT / role / trial["output_artifact"], allow_pickle=False)
        h.assert_output(np, value)
        h.require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == trial["output_bytes_sha256"],
                  "Served output bytes changed")
        arrays.append(value)
    h.require(arrays[0].tobytes(order="C") == arrays[2].tobytes(order="C"), "Repeated output differs")
    return arrays


def acquisition(h, role, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable and "anaconda3" not in sys.prefix,
              "Only designated OFFPY is permitted; never AP/Torch")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    h.require(nice >= 10, "Launch at nice>=10")
    env = {name: os.environ.get(name) for name in BLAS_ENV}
    h.require(env == dict.fromkeys(BLAS_ENV, "1"), "Set all three BLAS controls1 before imports")
    priors = prerequisite_roles(h, role)
    source_set = sources(h)
    fixture_set, rows = h.fixture_bindings()
    versions = {name: importlib.metadata.version(name)
                for name in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0",
              "Wrong native package versions")
    for prior in priors.values():
        h.require(prior["source_bindings"] == source_set and prior["fixture_bindings"] == fixture_set
                  and prior["versions"] == versions, "Roles must share the complete frozen frame")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, role=role, pid=os.getpid(), utc=h.utc(), nice=nice,
                   executable=sys.executable, sequence=list(RANKS), versions=versions, blas_env=env,
                   source_bindings=source_set, fixture_bindings=fixture_set, truth_files_accessed=False,
                   instrumented=instrumented(role), flop_budget=h.BUDGET, native_wall_limit_s=120,
                   rss_limit_bytes=h.RSS_LIMIT, initialization_once_in_call0=True,
                   one_estimator_instance=True, no_warmup=True)
    start_sha = h.write_json(directory / "started.json", started)
    trials, artifacts, predictions = [], {}, []
    result = dict(schema=SCHEMA, role=role, pid=os.getpid(), completed=False,
                  qualification_passed=False, truth_files_accessed=False,
                  started=started, started_sha256=start_sha, source_bindings=source_set,
                  fixture_bindings=fixture_set, versions=versions, blas_env=env,
                  candidate=candidate_name(role), candidate_sha256=PINS[candidate_name(role)],
                  instrumented=instrumented(role), projection_enabled=role == "candidate",
                  production_residual_validated=False)
    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("fixed_whole_pair_candidate", HERE / candidate_name(role))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["candidate_import_wall_s"] = time.perf_counter() - t0
        if role in ("null", "candidate"):
            h.require(module._GPK_ENABLED is True, "Default fixed projection flag changed")
        if role == "null":
            module._GPK_ENABLED = False  # Diagnostic null only; never used for the actual candidate score.
        estimator = None
        row_by_rank = {row["rank"]: row for row in rows}
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                      "Reported BLAS thread count differs from1")
            result["threadpools"] = pools
            for call, rank in enumerate(RANKS):
                row = row_by_rank[rank]
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16,1024,1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"whole-pair-fixed-{rank}")
                captured_layers, captured_x, captured_delta = [], [], []
                original_step = module.gnn_step if instrumented(role) else None
                if instrumented(role):
                    def wrapped(l, x, sigma, w, state, payload):
                        before = int(context.flops_used)
                        captured_layers.append(int(l))
                        captured_x.append(np.asarray(x).copy())  # BEFORE the original head; never fed back.
                        h.require(int(context.flops_used) == before, "Feature diagnostic unexpectedly charged")
                        value = original_step(l, x, sigma, w, state, payload)
                        before = int(context.flops_used)
                        captured_delta.append(np.asarray(value[0]).copy())
                        h.require(int(context.flops_used) == before, "Delta diagnostic unexpectedly charged")
                        return value
                    module.gnn_step = wrapped
                began = h.utc(); t0 = time.perf_counter()
                try:
                    with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as context:
                        if estimator is None:
                            estimator = module.Estimator()
                        raw_returned = estimator.predict(mlp, h.BUDGET)
                        h.require(isinstance(raw_returned, np.ndarray)
                                  and raw_returned.shape == (16,1024)
                                  and raw_returned.dtype == np.dtype("float32"),
                                  "Expected concrete native float32 output, never a deferred/cast repair")
                        returned = materialise_predictions(raw_returned, depth=16, width=1024)
                        del raw_returned  # Same metered materialisation boundary as the official worker.
                    elapsed = time.perf_counter() - t0
                finally:
                    if instrumented(role):
                        module.gnn_step = original_step
                out = np.asarray(returned).copy()  # No cast/clamp/repair.
                h.assert_output(np, out)
                name = f"call_{call:02d}_rank{rank}_output.npy"
                artifacts[name] = h.save_npy(np, directory / name, out)
                predictions.append(out)
                if instrumented(role):
                    h.require(captured_layers == list(range(16)), "Full16 response head routing failed")
                    x, delta = np.stack(captured_x), np.stack(captured_delta)
                    h.require(x.shape == (16,1024,30) and delta.shape == (16,1024)
                              and bool(np.isfinite(x).all()) and bool(np.isfinite(delta).all()),
                              "Invalid pre-head diagnostics")
                    name = f"call_{call:02d}_rank{rank}_diagnostics.npz"
                    artifacts[name] = h.save_diagnostics(np, directory / name, x, delta)
                    del x, delta
                trial = dict(call=call, rank=rank, began_utc=began, ended_utc=h.utc(),
                             flops=int(context.flops_used), context_wall_s=elapsed,
                             meter_wall_s=float(context.wall_time_s),
                             flopscope_backend_s=float(context.flopscope_backend_time_s),
                             flopscope_overhead_s=float(context.flopscope_overhead_time_s),
                             native_residual_s=float(context.residual_wall_time_s),
                             peak_rss_bytes=h.peak_rss_bytes(),
                             output_artifact=f"call_{call:02d}_rank{rank}_output.npy",
                             output_bytes_sha256=hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                             includes_first_call_initialization=call == 0,
                             instrumented=instrumented(role), diagnostic_copies_zero_charge=True)
                trials.append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); artifacts[bill] = h.binding(directory / bill)
                print(json.dumps(dict(role=role, **trial)), flush=True)
                h.require(0 < trial["flops"] <= h.BUDGET and elapsed < 120
                          and trial["peak_rss_bytes"] < h.RSS_LIMIT
                          and 0 <= trial["native_residual_s"] < float("inf")
                          and 0 < trial["meter_wall_s"] < 120
                          and 0 <= trial["flopscope_backend_s"] < float("inf")
                          and 0 <= trial["flopscope_overhead_s"] < float("inf"),
                          "Native FLOP/wall/RSS qualification failed; preserve and stop")
                del weights, mlp, returned, out, captured_x, captured_delta
        repeat = predictions[0].tobytes(order="C") == predictions[2].tobytes(order="C")
        h.require(repeat, "Cold/repeated35 served bytes differ")
        parity = []
        if role != "parent_nullref":
            reference_role = "parent" if role == "candidate" else "parent_nullref"
            reference = priors[reference_role]
            refs = load_outputs(h, np, reference_role, reference)
            for call, (out, ref) in enumerate(zip(predictions, refs, strict=True)):
                layers = 8 if role == "candidate" else 16
                entry = dict(call=call, rank=RANKS[call], compared_output_layers=layers,
                             output_byte_identical=out[:layers].tobytes(order="C") == ref[:layers].tobytes(order="C"))
                h.require(entry["output_byte_identical"], "Insertion/prefix output parity failed")
                if role != "candidate":
                    entry["bill_identical"] = trials[call]["flops"] == reference["trials"][call]["flops"]
                    h.require(entry["bill_identical"], "Null/instrumentation changed complete bill")
                if role == "null":
                    filename = f"call_{call:02d}_rank{RANKS[call]}_diagnostics.npz"
                    with np.load(directory / filename, allow_pickle=False) as actual, np.load(
                            ROOT / reference_role / filename, allow_pickle=False) as expected:
                        h.require(set(actual.files) == set(expected.files) == {"x", "delta"}, "Diagnostic keys changed")
                        for key in ("x", "delta"):
                            a, b = actual[key], expected[key]
                            entry[key + "_byte_identical"] = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                            h.require(entry[key + "_byte_identical"], "Null pre-head feature/delta parity failed")
                parity.append(entry)
        h.check_bindings(source_set); h.check_bindings(fixture_set); h.check_bindings(artifacts)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                      repeat_byte_identical=True, parity=parity, parity_passed=True)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result.update(trials=trials, artifacts=artifacts, completed_utc=h.utc())
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, role=role,
                 qualification_passed=result["qualification_passed"], result_sha256=result_sha,
                 truth_files_accessed=False, completed_utc=h.utc()))
    print(json.dumps(dict(role=role, qualification_passed=result["qualification_passed"],
                          error=result.get("error"), result_sha256=result_sha)), flush=True)
    return 0 if result["qualification_passed"] else 1


def score(h, directory):
    records = {role: verify_role(h, role) for role in ROLES}
    h.require(len({r["pid"] for r in records.values()}) == 4, "Roles were not distinct fresh processes")
    for previous, later in zip(ROLES[:-1], ROLES[1:], strict=True):
        a, b = records[previous], records[later]
        h.require(a["completed_utc"] <= b["started"]["utc"]
                  and a["source_bindings"] == b["source_bindings"]
                  and a["fixture_bindings"] == b["fixture_bindings"]
                  and a["versions"] == b["versions"], "Role order or frozen frame changed")
    import numpy as np  # Scorer imports neither candidate nor FlopScope.
    h.require(importlib.metadata.version("numpy") == records["parent"]["versions"]["numpy"],
              "Scoring NumPy differs from acquisition")
    outputs = {role: load_outputs(h, np, role, records[role]) for role in ("parent", "candidate")}
    sealed = {f"{role}_{name}": h.binding(ROOT / role / name) for role in ROLES
              for name in ("started.json", "result.json", "completion.json")}
    before = h.write_json(directory / "all_roles_verified_before_truth.json", dict(
        schema=SCHEMA, utc=h.utc(), truth_files_accessed=False, source_bindings=sources(h),
        role_seals=sealed, actual_score_roles=["parent", "candidate"], ranks=[35,100],
        references_are_development_not_independent_panel=True))
    truth_started = h.utc()
    h.write_json(directory / "truth_access_started.json", dict(utc=truth_started, prior_seal_sha256=before))
    truths, truth_bindings = {}, {}
    for rank in (35,100):
        path = h.TRUTH_DIR / f"truthpost_rank{rank:04d}.npz"
        truth_bindings[str(rank)] = h.binding(path)
        with np.load(path, allow_pickle=False) as archive:
            truth = archive["mc_post"].astype(np.float64)
        h.require(truth.shape == (16,1024) and bool(np.isfinite(truth).all()), "Invalid development truth")
        truths[rank] = truth
    scored = {}
    for role in ("parent", "candidate"):
        per_call = []
        for trial, out in zip(records[role]["trials"], outputs[role], strict=True):
            mse = np.mean((out.astype(np.float64)-truths[trial["rank"]])**2, axis=1)
            raw = float(mse[-1]); multiplier = max(.1, trial["flops"] / h.BUDGET)
            per_call.append(dict(call=trial["call"], rank=trial["rank"], layer_mse=mse.tolist(),
                                 final_raw_mse=raw, flops=trial["flops"], multiplier=multiplier,
                                 adjusted=raw*multiplier, included_in_two_net_mean=trial["call"] < 2))
        scored[role] = dict(candidate=records[role]["candidate"], candidate_sha256=records[role]["candidate_sha256"],
                            per_call=per_call, mean_raw_mse=sum(t["final_raw_mse"] for t in per_call[:2])/2,
                            mean_flops=sum(t["flops"] for t in per_call[:2])/2,
                            mean_adjusted=sum(t["adjusted"] for t in per_call[:2])/2)
    h.require(scored["parent"]["mean_adjusted"] > 0 and scored["parent"]["mean_raw_mse"] > 0,
              "Invalid baseline denominator")
    h.require(all(scored["parent"]["per_call"][i]["adjusted"] > 0 for i in (0,1)),
              "Invalid per-net baseline denominator")
    per_net = [scored["candidate"]["per_call"][i]["adjusted"] / scored["parent"]["per_call"][i]["adjusted"]
               for i in (0,1)]
    ratio = scored["candidate"]["mean_adjusted"] / scored["parent"]["mean_adjusted"]
    advances = all(value < 1 for value in per_net) and ratio <= .99
    h.check_bindings(sealed); h.check_bindings(truth_bindings)
    for record in records.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    result = dict(schema=SCHEMA, qualification_passed=True, hashes_unchanged=True, completed_utc=h.utc(),
                  all_roles_verified_before_truth_sha256=before, truth_access_started_utc=truth_started,
                  truth_bindings=truth_bindings, role_seals=sealed, source_bindings=sources(h), roles=scored,
                  adjusted_candidate_over_parent=ratio, per_net_adjusted_ratios=per_net,
                  raw_candidate_over_parent=scored["candidate"]["mean_raw_mse"]/scored["parent"]["mean_raw_mse"],
                  flops_candidate_over_parent=scored["candidate"]["mean_flops"]/scored["parent"]["mean_flops"],
                  advances_to_independent_14_panel=advances,
                  scientific_status="advance_to_independent14" if advances else "fixed_arm_closed",
                  reference="Stored1e9 development truthpost mc_post; not independent14 or official score",
                  aggregation="Mean of per-net final MSE*max(.1,complete F/2^41); init once equally; repeat excluded",
                  official_score=False, milestone_achieved=False, goal_achieved=False,
                  production_residual_validated=False)
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, qualification_passed=True,
                 result_sha256=result_sha, advances_to_independent_14_panel=advances, completed_utc=h.utc()))
    print(json.dumps(dict(roles=scored, adjusted_ratio=ratio, per_net_adjusted_ratios=per_net,
                          advances_to_independent_14_panel=advances, result_sha256=result_sha)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--role", choices=ROLES)
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.role is None and not args.score:
        parser.print_help(); return 0
    h = load_helper()
    if args.role == "parent_nullref":
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Start the fixed new sequence with parent_nullref")
    directory = ROOT / ("score" if args.score else args.role)
    directory.mkdir(exist_ok=False)
    try:
        return score(h, directory) if args.score else acquisition(h, args.role, directory)
    except Exception as error:
        h.write_json(directory / "preflight_or_score_failure.json", dict(
            schema=SCHEMA, utc=h.utc(), role=args.role, qualification_passed=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
