"""Inert fixed parent/candidate35,100,35 acquisition and sealed later scoring."""

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
ROOT = HERE / "pair_polynomial_f32_fixed_dev_v1"
SCHEMA = "pair_polynomial_f32_fixed_dev_v1"
ROLES = ("parent", "candidate")
RANKS = (35, 100, 35)
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PARENT = "candidate_response_rows_current_runtime_v1.py"
CANDIDATE = "candidate_response_pair_polynomial_f32_v1.py"
HELPER = "check_response_rows_adjusted_v1.py"
ARCHIVED = HERE / "whole_pair_kernel_fixed_dev_v1/parent"
ARCHIVED_SHA = "a7d78e74b157f628d0d7881d552c22d0cf0a2085e05fb60e7b1f8ae653322c59"
PINS = {
    PARENT: "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    CANDIDATE: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    "PAIR_POLYNOMIAL_F32_FIXED_DEV_V1.md": "ee28da165ba80c6b606232917573b1c62290b64e8d6c8f26a2b2f2fcd636c503",
    "build_pair_polynomial_f32_candidate_v1.py": "88cbf06c2342f9d56afefd4038748540eabdcef1ac09cfe776ecaf818c52c0c5",
    "pair_polynomial_f32_build_v1.json": "e11ed8df55af26630774b506ef447ee3e6298a41f0f795336311f961c2ff4dad",
    "pair_polynomial_f32_source_v1.py": "fb758f0e3e36a0ee29cb7f69490af83ece4c7e12424cb82c5122ee4e00113198",
    "pair_polynomial_f32_preflight_v1/result.json": "e5441d15e13ce38c186b7d15e943191a689349cb52628e6668c5a018a9567908",
}


def load_helper():
    if hashlib.sha256((HERE/HELPER).read_bytes()).hexdigest() != PINS[HELPER]:
        raise RuntimeError("Frozen inert helper changed")
    spec = importlib.util.spec_from_file_location("fixed_fp32_dev_helpers", HERE/HELPER)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


def sources(h):
    result = {name: h.binding(HERE/name) for name in (*PINS, Path(__file__).name)}
    h.require(all(result[name]["sha256"] == value for name, value in PINS.items()), "Source pin mismatch")
    build = h.load_json(HERE/"pair_polynomial_f32_build_v1.json")
    h.require(build["candidate_sha256"] == PINS[CANDIDATE] and build["parent_sha256"] == PINS[PARENT]
              and build["inverse_byte_exact"] and build["all_other_source_byte_exact"]
              and build["same_imports_and_payloads"], "Invalid exact source inverse")
    result["archived_parent_result"] = h.binding(ARCHIVED/"result.json")
    h.require(result["archived_parent_result"]["sha256"] == ARCHIVED_SHA, "Archived parent seal changed")
    return result


def name_for(role):
    return PARENT if role == "parent" else CANDIDATE


def verify_role(h, role):
    directory = ROOT/role
    complete = h.load_json(directory/"completion.json")
    h.require(complete["qualification_passed"] and complete["role"] == role
              and complete["result_sha256"] == h.digest(directory/"result.json"), "Incomplete role seal")
    r = h.load_json(directory/"result.json")
    h.require(r["schema"] == SCHEMA and r["role"] == role and r["qualification_passed"]
              and r["completed"] and r["hashes_unchanged"] and not r["truth_files_accessed"]
              and not r["instrumented"] and r["repeat_byte_identical"]
              and r["compatibility_passed"], "Invalid acquisition")
    h.require(r["source_bindings"] == sources(h) and r["candidate"] == name_for(role)
              and r["candidate_sha256"] == PINS[name_for(role)], "Acquisition source frame changed")
    h.require(h.digest(directory/"started.json") == r["started_sha256"]
              and h.load_json(directory/"started.json") == r["started"], "Start header changed")
    s = r["started"]
    h.require(s["pid"] == r["pid"] and s["nice"] >= 10 and s["sequence"] == list(RANKS)
              and s["source_bindings"] == r["source_bindings"]
              and s["fixture_bindings"] == r["fixture_bindings"]
              and s["versions"] == r["versions"] and s["blas_env"] == r["blas_env"], "Invalid process header")
    h.require(r["versions"]["flopscope"] == "0.12.0" and r["versions"]["whestbench"] == "0.16.0"
              and r["blas_env"] == dict.fromkeys(BLAS_ENV, "1"), "Wrong native environment")
    h.check_bindings(r["fixture_bindings"]); h.check_bindings(r["artifacts"])
    h.require([t["rank"] for t in r["trials"]] == list(RANKS), "Wrong sequence")
    expected = set()
    for call, t in enumerate(r["trials"]):
        name = f"call_{call:02d}_rank{RANKS[call]}_output.npy"
        bill = f"call_{call:02d}_bill.json"
        expected.update((name, bill))
        h.require(t["call"] == call and t["output_artifact"] == name
                  and h.load_json(directory/bill) == t and 0 < t["flops"] <= h.BUDGET
                  and 0 < t["context_wall_s"] < 120 and 0 < t["meter_wall_s"] < 120
                  and 0 <= t["native_residual_s"] < math.inf
                  and 0 <= t["flopscope_backend_s"] < math.inf
                  and 0 <= t["flopscope_overhead_s"] < math.inf
                  and t["peak_rss_bytes"] < h.RSS_LIMIT
                  and t["includes_first_call_initialization"] == (call == 0), "Invalid completed bill")
    h.require(set(r["artifacts"]) == expected, "Incomplete outputs/bills")
    for name, binding in r["artifacts"].items():
        h.require(Path(binding["path"]).resolve() == (directory/name).resolve(), "Escaped role output")
    h.require(r["trials"][0]["flops"]-r["trials"][2]["flops"] == 1383913666,
              "Initialization price changed")
    return r


def outputs(h, np, directory, record):
    values = []
    for t in record["trials"]:
        a = np.load(directory/t["output_artifact"], allow_pickle=False)
        h.assert_output(np, a)
        h.require(hashlib.sha256(a.tobytes(order="C")).hexdigest() == t["output_bytes_sha256"], "Output bytes changed")
        values.append(a)
    h.require(values[0].tobytes() == values[2].tobytes(), "Repeated35 differs")
    return values


def acquire(h, role, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve() and "anaconda3" not in sys.executable,
              "Use OFFPY, never AP/threadpoolctl")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    h.require(nice >= 10 and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "nice10/no-bytecode required")
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(env == dict.fromkeys(BLAS_ENV, "1"), "Explicit launch thread controls required")
    prior = verify_role(h, "parent") if role == "candidate" else None
    source_set = sources(h)
    fixture_set, rows = h.fixture_bindings()
    versions = {key: importlib.metadata.version(key) for key in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0", "Wrong package versions")
    if prior:
        h.require(prior["pid"] != os.getpid() and prior["source_bindings"] == source_set
                  and prior["fixture_bindings"] == fixture_set and prior["versions"] == versions,
                  "Fresh paired process/frame required")
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
                   instrumented=False, no_warmup=True, one_estimator_instance=True,
                   initialization_once_in_call0=True, native_wall_limit_s=120, rss_limit_bytes=h.RSS_LIMIT)
    start_sha = h.write_json(directory/"started.json", started)
    trials, artifacts, predictions = [], {}, []
    result = dict(schema=SCHEMA, role=role, pid=os.getpid(), completed=False, qualification_passed=False,
                  started=started, started_sha256=start_sha, source_bindings=source_set,
                  fixture_bindings=fixture_set, versions=versions, blas_env=env,
                  candidate=name_for(role), candidate_sha256=PINS[name_for(role)],
                  truth_files_accessed=False, instrumented=False, production_residual_validated=False)
    try:
        t0 = time.perf_counter()
        spec = importlib.util.spec_from_file_location("fixed_fp32poly_candidate", HERE/name_for(role))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["candidate_import_wall_s"] = time.perf_counter()-t0
        estimator = None
        by_rank = {row["rank"]: row for row in rows}
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "BLAS threads differ")
            result["threadpools"] = pools
            for call, rank in enumerate(RANKS):
                row = by_rank[rank]
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16,1024,1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"fp32poly-fixed-{rank}")
                began = h.utc(); t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16,1024), "Concrete served float32 output required")
                    returned = materialise_predictions(served, depth=16, width=1024)
                    del served
                elapsed = time.perf_counter()-t0
                out = np.asarray(returned).copy()
                h.assert_output(np, out)
                name = f"call_{call:02d}_rank{rank}_output.npy"
                artifacts[name] = h.save_npy(np, directory/name, out)
                predictions.append(out)
                t = dict(call=call, rank=rank, began_utc=began, ended_utc=h.utc(),
                         flops=int(ctx.flops_used), context_wall_s=elapsed, meter_wall_s=float(ctx.wall_time_s),
                         flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                         flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                         native_residual_s=float(ctx.residual_wall_time_s), peak_rss_bytes=h.peak_rss_bytes(),
                         output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                         includes_first_call_initialization=call == 0)
                trials.append(t)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory/bill, t); artifacts[bill] = h.binding(directory/bill)
                print(json.dumps(dict(role=role, **t)), flush=True)
                h.require(0 < t["flops"] <= h.BUDGET and 0 < elapsed < 120
                          and 0 < t["meter_wall_s"] < 120 and t["peak_rss_bytes"] < h.RSS_LIMIT
                          and 0 <= t["native_residual_s"] < math.inf
                          and 0 <= t["flopscope_backend_s"] < math.inf
                          and 0 <= t["flopscope_overhead_s"] < math.inf, "Native resource gate failed")
                del weights, mlp, returned, out
        h.require(predictions[0].tobytes() == predictions[2].tobytes(), "Cold/repeated35 differs")
        h.require(trials[0]["flops"]-trials[2]["flops"] == 1383913666, "Initialization price changed")
        reference = prior if prior else h.load_json(ARCHIVED/"result.json")
        ref_directory = ROOT/"parent" if prior else ARCHIVED
        h.require(reference["fixture_bindings"] == fixture_set and reference["versions"] == versions
                  and reference["candidate_sha256"] == PINS[PARENT], "Reference frame differs")
        h.check_bindings(reference["artifacts"])
        refs = outputs(h, np, ref_directory, reference)
        comparisons = []
        for call, (a, b) in enumerate(zip(predictions, refs, strict=True)):
            errors = np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), axis=1)
            item = dict(call=call, rank=RANKS[call], per_layer_max_abs=errors.tolist(),
                        layer0_byte_exact=a[0].tobytes() == b[0].tobytes())
            h.require(item["layer0_byte_exact"], "Layer0 compatibility failed")
            if role == "parent":
                h.require(a.tobytes() == b.tobytes() and trials[call]["flops"] == reference["trials"][call]["flops"],
                          "Fresh parent differs from archived output/bill")
                item["archived_output_and_bill_exact"] = True
            else:
                h.require(bool((errors[1:] <= 5e-6).all()), "Fixed numerical trajectory gate failed")
            comparisons.append(item)
        h.check_bindings(source_set); h.check_bindings(fixture_set); h.check_bindings(artifacts)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                      repeat_byte_identical=True, compatibility_passed=True, comparisons=comparisons)
    except BaseException:
        result.update(error=traceback.format_exc())
    result.update(trials=trials, artifacts=artifacts, completed_utc=h.utc())
    seal = h.write_json(directory/"result.json", result)
    h.write_json(directory/"completion.json", dict(schema=SCHEMA, role=role, qualification_passed=result["qualification_passed"],
                 result_sha256=seal, completed_utc=h.utc(), truth_files_accessed=False))
    print(json.dumps(dict(role=role, qualification_passed=result["qualification_passed"], result_sha256=seal,
                         error=result.get("error"))), flush=True)
    return 0 if result["qualification_passed"] else 1


def score(h, directory):
    records = {role: verify_role(h, role) for role in ROLES}
    a, b = records["parent"], records["candidate"]
    h.require(a["pid"] != b["pid"] and a["completed_utc"] <= b["started"]["utc"]
              and a["source_bindings"] == b["source_bindings"] and a["fixture_bindings"] == b["fixture_bindings"]
              and a["versions"] == b["versions"], "Paired chronology/frame differs")
    import numpy as np
    h.require(importlib.metadata.version("numpy") == a["versions"]["numpy"], "Scoring NumPy changed")
    predictions = {role: outputs(h, np, ROOT/role, record) for role, record in records.items()}
    seals = {f"{role}_{name}": h.binding(ROOT/role/name) for role in ROLES
             for name in ("started.json", "result.json", "completion.json")}
    before = h.write_json(directory/"both_roles_verified_before_truth.json", dict(schema=SCHEMA, utc=h.utc(),
                          role_seals=seals, source_bindings=sources(h), truth_files_accessed=False))
    truth_time = h.utc()
    h.write_json(directory/"truth_access_started.json", dict(utc=truth_time, prior_seal_sha256=before))
    truths, truth_bindings = {}, {}
    for rank in (35,100):
        path = h.TRUTH_DIR/f"truthpost_rank{rank:04d}.npz"
        truth_bindings[str(rank)] = h.binding(path)
        with np.load(path, allow_pickle=False) as archive:
            truth = archive["mc_post"].astype(np.float64)
        h.require(truth.shape == (16,1024) and bool(np.isfinite(truth).all()), "Invalid dev truth")
        truths[rank] = truth
    scored = {}
    for role, record in records.items():
        calls = []
        for t, out in zip(record["trials"], predictions[role], strict=True):
            mse = np.mean((out.astype(np.float64)-truths[t["rank"]])**2, axis=1)
            raw = float(mse[-1]); multiplier = max(.1, t["flops"]/h.BUDGET)
            calls.append(dict(call=t["call"], rank=t["rank"], layer_mse=mse.tolist(), final_raw_mse=raw,
                              flops=t["flops"], multiplier=multiplier, adjusted=raw*multiplier,
                              included_in_two_net_mean=t["call"] < 2))
        scored[role] = dict(candidate=name_for(role), candidate_sha256=PINS[name_for(role)], per_call=calls,
                            mean_raw_mse=sum(t["final_raw_mse"] for t in calls[:2])/2,
                            mean_flops=sum(t["flops"] for t in calls[:2])/2,
                            mean_adjusted=sum(t["adjusted"] for t in calls[:2])/2)
    ratios = [scored["candidate"]["per_call"][i]["adjusted"]/scored["parent"]["per_call"][i]["adjusted"] for i in (0,1)]
    ratio = scored["candidate"]["mean_adjusted"]/scored["parent"]["mean_adjusted"]
    advance = all(x < 1 for x in ratios) and ratio <= .995
    h.check_bindings(seals); h.check_bindings(truth_bindings)
    for record in records.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    result = dict(schema=SCHEMA, qualification_passed=True, hashes_unchanged=True, completed_utc=h.utc(),
                  roles=scored, adjusted_candidate_over_parent=ratio, per_net_adjusted_ratios=ratios,
                  raw_candidate_over_parent=scored["candidate"]["mean_raw_mse"]/scored["parent"]["mean_raw_mse"],
                  flops_candidate_over_parent=scored["candidate"]["mean_flops"]/scored["parent"]["mean_flops"],
                  advances_to_independent_14_panel=advance, scientific_status="dev_gate_pass" if advance else "fixed_partition_closed",
                  source_bindings=sources(h), role_seals=seals, truth_bindings=truth_bindings,
                  both_roles_verified_before_truth_sha256=before, truth_access_started_utc=truth_time,
                  reference="Stored1e9 development35/100; NOT independent14 or official",
                  aggregation="Mean per-net raw*max(.1,completeF/2^41); initialization once equally; repeat excluded",
                  official_score=False, milestone_achieved=False, goal_achieved=False, production_residual_validated=False)
    seal = h.write_json(directory/"result.json", result)
    h.write_json(directory/"completion.json", dict(schema=SCHEMA, qualification_passed=True, result_sha256=seal,
                 advances_to_independent_14_panel=advance, completed_utc=h.utc()))
    print(json.dumps(dict(roles=scored, adjusted_ratio=ratio, per_net_adjusted_ratios=ratios,
                          advances_to_independent_14_panel=advance, result_sha256=seal)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--role", choices=ROLES)
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.role is None and not args.score:
        parser.print_help(); return 0
    h = load_helper()
    if args.role == "parent":
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Start fixed new sequence with parent")
    directory = ROOT/("score" if args.score else args.role)
    directory.mkdir(exist_ok=False)
    try:
        return score(h, directory) if args.score else acquire(h, args.role, directory)
    except BaseException:
        h.write_json(directory/"preflight_or_score_failure.json", dict(schema=SCHEMA, utc=h.utc(),
                     qualification_passed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
