"""Inert fixed null/candidate35,100,35 acquisition, then sealed raw/full-cost scoring."""
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
ROOT = HERE / "dev"
ARCHIVED = NEW / "terminal_dead_consumers_v1/panel/candidate"
ARCHIVED_SHA = "302d9bbaf1327e0f12aa8746406536e24ed85beca6ec23eb807c8a2a03cd7344"
PARENT_SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"
HELPER = NEW / "check_response_rows_adjusted_v1.py"
HELPER_SHA = "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304"
PRIOR_SCORE = NEW / "pair_polynomial_f32_fixed_dev_v1/score/result.json"
PRIOR_SCORE_SHA = "fc89efe6cd9117dcfb86a31469cfc356d61b5f70397338641d0357c99e534b81"
RANKS = (35, 100, 35)
ROLES = ("null", "candidate")
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SCHEMA = "resident_gram512_fixed_dev_v1"
INITIALIZATION_FLOPS = 1383913666
RSS_LIMIT = 8_000_000_000


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Import specification failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("Frozen utility source changed")
    return load(HELPER, "gram_dev_readonly_utilities")


def filename(role):
    return "estimator_null.py" if role == "null" else "estimator.py"


def sources(h, build_sha):
    files = [HERE / name for name in ("build.json", "source.py", "gram.py", "PROTOCOL.md",
             "check_synthetic.py", "synthetic/result.json", "synthetic/completion.json",
             "estimator.py", "estimator_null.py", "check_dev.py")]
    files += [HELPER, NEW / "terminal_dead_consumers_v1/estimator.py", ARCHIVED / "result.json",
              ARCHIVED / "completion.json", ARCHIVED / "predictions_and_bills_sealed.json",
              PRIOR_SCORE, PRIOR_SCORE.with_name("completion.json")]
    bindings = {str(path): h.binding(path) for path in files}
    build = h.load_json(HERE / "build.json")
    h.require(h.digest(HERE / "build.json") == build_sha and build["passed"] and not build["executed"],
              "Wrong fixed source build")
    h.require(build["parent_sha256"] == build["null_sha256"] == PARENT_SHA
              and build["inverse_source_byte_exact"] and build["null_source_byte_exact"]
              and build["all_other_definitions_byte_exact"] and build["imports_and_payloads_unchanged"],
              "Source/head/null compatibility failed")
    for key, name in (("candidate_sha256", "estimator.py"), ("null_sha256", "estimator_null.py"),
                      ("source_sha256", "source.py"), ("protocol_sha256", "PROTOCOL.md"),
                      ("helper_sha256", "gram.py"), ("synthetic_result_sha256", "synthetic/result.json")):
        h.require(build[key] == h.digest(HERE / name), "Build source/gate mismatch")
    gate = h.load_json(HERE / "synthetic/result.json")
    complete = h.load_json(HERE / "synthetic/completion.json")
    h.require(gate["passed"] and gate["sources_unchanged"] and complete["passed"]
              and complete["result_sha256"] == build["synthetic_result_sha256"], "Synthetic seal failed")
    for path, expected in gate["source_bindings"].items():
        h.require(h.digest(path) == expected, "Qualified synthetic source changed")
    h.require(h.digest(ARCHIVED / "result.json") == ARCHIVED_SHA, "Current parent receipt changed")
    archive = h.load_json(ARCHIVED / "result.json")
    ac = h.load_json(ARCHIVED / "completion.json")
    h.require(archive["qualification_passed"] and archive["candidate_sha256"] == PARENT_SHA
              and ac["qualification_passed"] and ac["result_sha256"] == ARCHIVED_SHA,
              "Parent acquisition is not qualified current source")
    h.check_bindings(archive["artifacts"])
    h.require(h.digest(PRIOR_SCORE) == PRIOR_SCORE_SHA, "Fixed development-reference receipt changed")
    prior_score = h.load_json(PRIOR_SCORE)
    pc = h.load_json(PRIOR_SCORE.with_name("completion.json"))
    h.require(prior_score["qualification_passed"] and prior_score["hashes_unchanged"]
              and pc["qualification_passed"] and pc["result_sha256"] == PRIOR_SCORE_SHA,
              "Prior development-reference seal failed")
    h.check_bindings(bindings)
    return bindings, build, archive


def verify(h, role, build_sha):
    directory = ROOT / role
    complete = h.load_json(directory / "completion.json")
    r = h.load_json(directory / "result.json")
    h.require(complete["passed"] and complete["result_sha256"] == h.digest(directory / "result.json")
              and r["passed"] and r["schema"] == SCHEMA and r["role"] == role
              and r["source_bindings"] == sources(h, build_sha)[0]
              and not r["truth_files_accessed"] and r["repeat_byte_identical"]
              and r["hashes_unchanged"], "Invalid completed acquisition")
    h.require(r["started"] == h.load_json(directory / "started.json")
              and r["started_sha256"] == h.digest(directory / "started.json")
              and r["started"]["nice"] >= 10 and r["started"]["sequence"] == list(RANKS)
              and r["started"]["blas_env"] == dict.fromkeys(BLAS_ENV, "1"), "Invalid process start")
    h.check_bindings(r["artifacts"]); h.check_bindings(r["fixture_bindings"])
    h.require([t["rank"] for t in r["trials"]] == list(RANKS)
              and r["trials"][0]["flops"]-r["trials"][2]["flops"] == INITIALIZATION_FLOPS,
              "Wrong sequence/initialization bill")
    expected_artifacts = {f"call_{i:02d}_rank{rank}_output.npy" for i, rank in enumerate(RANKS)}
    expected_artifacts.update(f"call_{i:02d}_bill.json" for i in range(3))
    h.require(set(r["artifacts"]) == expected_artifacts, "Incomplete outputs/bills")
    for name, binding in r["artifacts"].items():
        h.require(Path(binding["path"]).resolve() == (directory / name).resolve(), "Escaped acquisition artifact")
    for call, t in enumerate(r["trials"]):
        h.require(t["call"] == call and t["output"] == f"call_{call:02d}_rank{RANKS[call]}_output.npy"
                  and t["includes_initialization"] == (call == 0)
                  and h.load_json(directory / f"call_{call:02d}_bill.json") == t
                  and 0 < t["flops"] <= h.BUDGET and 0 < t["wall_s"] < 120
                  and 0 < t["meter_wall_s"] < 120 and t["rss_bytes"] < RSS_LIMIT
                  and all(math.isfinite(t[key]) and t[key] >= 0 for key in
                          ("native_residual_s", "backend_s", "overhead_s")), "Invalid full bill")
    return r


def outputs(h, np, directory, r):
    values = []
    for t in r["trials"]:
        value = np.load(directory / t["output"], allow_pickle=False)
        h.assert_output(np, value)
        h.require(hashlib.sha256(value.tobytes()).hexdigest() == t["output_sha256"], "Output identity")
        values.append(value)
    h.require(values[0].tobytes() == values[2].tobytes(), "Repeat byte identity")
    return values


def acquire(h, role, build_sha, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve() and "anaconda3" not in sys.executable,
              "Use OFFPY, never AP/threadpoolctl")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(nice >= 10 and env == dict.fromkeys(BLAS_ENV, "1")
              and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "nice10/BLAS1/no-bytecode")
    bindings, build, archive = sources(h, build_sha)
    fixtures, rows = h.fixture_bindings()
    h.require(set(fixtures) == {"manifest", "row_35", "row_100", "weights_35", "weights_100"}
              and all(value == archive["fixture_bindings"][key] for key, value in fixtures.items()),
              "Canonical fixtures differ from qualified current-parent archive")
    prior = verify(h, "null", build_sha) if role == "candidate" else None
    versions = {name: importlib.metadata.version(name) for name in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    h.require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0"
              and versions["numpy"] == "2.4.6" and versions == archive["versions"], "Native versions changed")
    if prior:
        h.require(prior["pid"] != os.getpid() and prior["fixture_bindings"] == fixtures
                  and prior["versions"] == versions, "Fresh matched process/fixtures required")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    started = dict(schema=SCHEMA, role=role, utc=h.utc(), pid=os.getpid(), nice=nice,
                   sequence=list(RANKS), blas_env=env, versions=versions, source_bindings=bindings,
                   fixture_bindings=fixtures, initialization_once=True, instrumented=False,
                   no_warmup=True, truth_files_accessed=False)
    start_sha = h.write_json(directory / "started.json", started)
    result = dict(schema=SCHEMA, role=role, passed=False, pid=os.getpid(), started=started,
                  started_sha256=start_sha, source_bindings=bindings, fixture_bindings=fixtures,
                  versions=versions, trials=[], artifacts={}, truth_files_accessed=False,
                  candidate_sha256=build["null_sha256" if role == "null" else "candidate_sha256"],
                  production_residual_validated=False)
    predictions = []
    try:
        module = load(HERE / filename(role), "gram_fixed_unwrapped_" + role)
        estimator = None
        by_rank = {row["rank"]: row for row in rows}
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "BLAS count")
            result["threadpools"] = pools
            for call, rank in enumerate(RANKS):
                row = by_rank[rank]
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16,1024,1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Phase2 canonical weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"resident-gram-{rank}")
                begin = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16,1024), "Concrete served FP32(16,1024)")
                    materialized = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter()-begin
                h.assert_output(np, served); h.assert_output(np, materialized)
                h.require(served.tobytes() == materialized.tobytes(), "Materializer changed served bytes")
                out = np.asarray(materialized).copy()
                name = f"call_{call:02d}_rank{rank}_output.npy"
                result["artifacts"][name] = h.save_npy(np, directory / name, out)
                predictions.append(out)
                trial = dict(call=call, rank=rank, flops=int(ctx.flops_used), wall_s=elapsed,
                             meter_wall_s=float(ctx.wall_time_s), backend_s=float(ctx.flopscope_backend_time_s),
                             overhead_s=float(ctx.flopscope_overhead_time_s),
                             native_residual_s=float(ctx.residual_wall_time_s), rss_bytes=h.peak_rss_bytes(),
                             output=name, output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                             includes_initialization=call == 0, served_materialized_byte_exact=True)
                result["trials"].append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); result["artifacts"][bill] = h.binding(directory / bill)
                print(json.dumps(dict(role=role, **trial)), flush=True)
                h.require(0 < trial["flops"] <= h.BUDGET and 0 < elapsed < 120
                          and 0 < trial["meter_wall_s"] < 120 and trial["rss_bytes"] < RSS_LIMIT
                          and all(math.isfinite(trial[key]) and trial[key] >= 0 for key in
                                  ("native_residual_s", "backend_s", "overhead_s")),
                          "Fixed native resource gate")
                del weights, mlp, served, materialized, out
        h.require(predictions[0].tobytes() == predictions[2].tobytes(), "Repeated35 changed")
        h.require(result["trials"][0]["flops"]-result["trials"][2]["flops"] == INITIALIZATION_FLOPS,
                  "Initialization price changed")
        if prior:
            refs = outputs(h, np, ROOT / "null", prior)
        else:
            refs = []
            for call in (0, 1, 0):
                t = archive["trials"][call]
                value = np.load(ARCHIVED / t["output_artifact"], allow_pickle=False)
                h.assert_output(np, value)
                h.require(hashlib.sha256(value.tobytes()).hexdigest() == t["output_bytes_sha256"], "Archive bytes")
                refs.append(value)
        comparisons = []
        for call, (value, ref) in enumerate(zip(predictions, refs, strict=True)):
            errors = np.max(np.abs(value.astype(np.float64)-ref.astype(np.float64)), axis=1)
            if role == "null":
                expected = archive["trials"][min(call, 1) if call < 2 else 0]["flops"]
                if call == 2:
                    expected -= INITIALIZATION_FLOPS
                h.require(value.tobytes() == ref.tobytes() and result["trials"][call]["flops"] == expected,
                          "Null differs from qualified parent bytes/full cost")
            else:
                h.require(bool((errors <= 1e-5).all()), "Fixed all16 trajectory fidelity gate")
            comparisons.append(dict(call=call, rank=RANKS[call], per_layer_max_abs=errors.tolist(),
                                    all16_byte_exact=value.tobytes() == ref.tobytes()))
        h.check_bindings(bindings); h.check_bindings(fixtures); h.check_bindings(result["artifacts"])
        h.require(h.peak_rss_bytes() < RSS_LIMIT, "Final native RSS cap")
        result.update(passed=True, repeat_byte_identical=True, hashes_unchanged=True, comparisons=comparisons)
    except BaseException:
        result["error"] = traceback.format_exc()
    result["completed_utc"] = h.utc()
    seal = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(passed=result["passed"], result_sha256=seal, role=role))
    print(json.dumps(dict(role=role, passed=result["passed"], result_sha256=seal, error=result.get("error"))), flush=True)
    return 0 if result["passed"] else 1


def score(h, build_sha, directory):
    records = {role: verify(h, role, build_sha) for role in ROLES}
    a, b = records["null"], records["candidate"]
    h.require(a["pid"] != b["pid"] and a["completed_utc"] <= b["started"]["utc"]
              and a["source_bindings"] == b["source_bindings"]
              and a["fixture_bindings"] == b["fixture_bindings"] and a["versions"] == b["versions"],
              "Matched chronology/configuration gate")
    import numpy as np
    h.require(importlib.metadata.version("numpy") == a["versions"]["numpy"], "Scoring environment")
    predictions = {role: outputs(h, np, ROOT / role, r) for role, r in records.items()}
    seals = {f"{role}/{name}": h.binding(ROOT / role / name) for role in ROLES
             for name in ("started.json", "result.json", "completion.json")}
    before = h.write_json(directory / "both_roles_verified_before_truth.json", dict(
        utc=h.utc(), role_seals=seals, source_bindings=sources(h, build_sha)[0], truth_files_accessed=False))
    h.write_json(directory / "truth_access_started.json", dict(utc=h.utc(), prior_seal_sha256=before))
    truths, truth_bindings = {}, {}
    prior_truths = h.load_json(PRIOR_SCORE)["truth_bindings"]
    for rank in (35, 100):
        path = h.TRUTH_DIR / f"truthpost_rank{rank:04d}.npz"
        truth_bindings[str(rank)] = h.binding(path)
        h.require(truth_bindings[str(rank)] == prior_truths[str(rank)],
                  "Development truth differs from the pinned prior1e9 frame")
        with np.load(path, allow_pickle=False) as data:
            value = data["mc_post"].astype(np.float64)
        h.require(value.shape == (16,1024) and bool(np.isfinite(value).all()), "Development truth")
        truths[rank] = value
    scored = {}
    for role, r in records.items():
        calls = []
        for t, out in zip(r["trials"], predictions[role], strict=True):
            mse = np.mean((out.astype(np.float64)-truths[t["rank"]])**2, axis=1)
            raw = float(mse[-1]); multiplier = max(.1, t["flops"]/h.BUDGET)
            calls.append(dict(call=t["call"], rank=t["rank"], layer_mse=mse.tolist(), raw=raw,
                              flops=t["flops"], multiplier=multiplier, adjusted=raw*multiplier))
        scored[role] = dict(candidate_sha256=r["candidate_sha256"], per_call=calls,
                            mean_raw=sum(t["raw"] for t in calls[:2])/2,
                            mean_flops=sum(t["flops"] for t in calls[:2])/2,
                            mean_adjusted=sum(t["adjusted"] for t in calls[:2])/2)
    ratios = [scored["candidate"]["per_call"][i]["adjusted"]/scored["null"]["per_call"][i]["adjusted"] for i in (0,1)]
    ratio = scored["candidate"]["mean_adjusted"]/scored["null"]["mean_adjusted"]
    advance = all(x < 1 for x in ratios) and ratio < 1
    h.check_bindings(seals); h.check_bindings(truth_bindings)
    for r in records.values():
        h.check_bindings(r["source_bindings"]); h.check_bindings(r["fixture_bindings"]); h.check_bindings(r["artifacts"])
    result = dict(schema=SCHEMA, passed=True, completed_utc=h.utc(), roles=scored,
                  per_net_adjusted_ratios=ratios, mean_adjusted_ratio=ratio,
                  advances_to_independent14=advance, role_seals=seals, truth_bindings=truth_bindings,
                  source_bindings=sources(h, build_sha)[0], both_roles_verified_before_truth_sha256=before,
                  reference="Stored1e9 development35/100; NOT independent14 or official",
                  aggregation="mean per-net raw*max(.1,completeF/2^41); initialization once; repeat excluded",
                  official_score=False, milestone_achieved=False, production_residual_validated=False)
    seal = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(passed=True, result_sha256=seal,
                 advances_to_independent14=advance))
    print(json.dumps(dict(roles=scored, per_net_adjusted_ratios=ratios, mean_adjusted_ratio=ratio,
                          advances_to_independent14=advance, result_sha256=seal)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--role", choices=ROLES)
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--build-sha")
    args = parser.parse_args()
    if args.role is None and not args.score:
        parser.print_help(); return 0
    h = helper()
    h.require(args.build_sha and len(args.build_sha) == 64
              and all(c in "0123456789abcdef" for c in args.build_sha), "Fixed build SHA required")
    if args.role == "null":
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Begin with fixed null acquisition")
    directory = ROOT / ("score" if args.score else args.role)
    directory.mkdir(exist_ok=False)
    try:
        return score(h, args.build_sha, directory) if args.score else acquire(h, args.role, args.build_sha, directory)
    except BaseException:
        h.write_json(directory / "failure.json", dict(passed=False, utc=h.utc(), error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
