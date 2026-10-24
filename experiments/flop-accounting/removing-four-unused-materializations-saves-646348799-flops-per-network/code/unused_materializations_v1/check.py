"""Inert exact-output native14+repeat acquisition and separate sealed scoring."""
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
ARCHIVE = NEW / "resident_gram512_v1/panel_v2/candidate"
OLD_SCORE = NEW / "resident_gram512_v1/panel_v2/score/result.json"
PARENT_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"
CANDIDATE_SHA = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
BUILD_SHA = "5e324e0226cfad657fec3861e3266070525bcd3614b21ec0e11f18afcf81553e"
ARCHIVE_SHA = "095e8ca059f71cde099781aef1aa5d9e38ce9f8ae70d4b58eadc7b93312a7bff"
SCORE_SHA = "e3dc33f0018ffb6d2e1b9855112bb4c7ae14e9805253694d147493a2bcf724cf"
HELPER_SHA = "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304"
PANEL_SHA = "6fcf38e55173a6053c51d536dc9791a0c925a84cd4164d39d159120c0d33acc6"
VERSIONS = dict(flopscope="0.12.0", whestbench="0.16.0", numpy="2.4.6", threadpoolctl="3.6.0")
BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
INITIALIZATION = 1383913666
RSS_LIMIT = 8_000_000_000
SCHEMA = "unused_materializations_exact14_v1"


def load(path, name, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Authenticated module changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    h = load(NEW / "check_response_rows_adjusted_v1.py", "unused_readonly_utilities", HELPER_SHA)
    panel = load(NEW / "check_response_rows_panel_v1.py", "unused_readonly_fixtures", PANEL_SHA)
    return h, panel


def prerequisites(h, panel):
    h.require(h.digest(HERE / "build.json") == BUILD_SHA, "Fixed build receipt changed")
    build = h.load_json(HERE / "build.json")
    h.require(build["passed"] and not build["executed"]
              and build["parent_sha256"] == PARENT_SHA
              and build["candidate_sha256"] == CANDIDATE_SHA
              and build["inverse_source_byte_exact"] and build["all_other_source_byte_exact"]
              and build["imports_and_payloads_unchanged"], "Invalid source compatibility")
    files = [HERE / n for n in ("build.py", "build.json", "PROTOCOL.md", "estimator.py", "check.py")]
    files += [Path(build["parent"]), ARCHIVE / "result.json", ARCHIVE / "completion.json",
              ARCHIVE / "predictions_and_bills_sealed.json", NEW / "check_response_rows_adjusted_v1.py",
              NEW / "check_response_rows_panel_v1.py"]
    bound = {str(p): h.binding(p) for p in files}
    h.require(h.digest(HERE / "build.py") == build["builder_sha256"]
              and h.digest(HERE / "PROTOCOL.md") == build["protocol_sha256"]
              and h.digest(HERE / "estimator.py") == CANDIDATE_SHA
              and h.digest(Path(build["parent"])) == PARENT_SHA, "Source binding failed")
    h.require(h.digest(ARCHIVE / "result.json") == ARCHIVE_SHA, "Qualified archive changed")
    old = h.load_json(ARCHIVE / "result.json")
    complete = h.load_json(ARCHIVE / "completion.json")
    h.require(old["qualification_passed"] and old["hashes_unchanged"]
              and old["candidate_sha256"] == PARENT_SHA and old["versions"] == VERSIONS
              and complete["qualification_passed"] and complete["result_sha256"] == ARCHIVE_SHA,
              "Parent qualification failed")
    h.require(h.digest(ARCHIVE / "predictions_and_bills_sealed.json")
              == old["predictions_and_bills_seal_sha256"], "Parent output/bill seal changed")
    h.check_bindings(old["artifacts"])
    fixtures, rows = panel.fixtures(h)
    h.require(fixtures == old["fixture_bindings"] and len(rows) == 14
              and [r["rank"] for r in rows] == [t["rank"] for t in old["trials"]], "Canonical14 mismatch")
    h.check_bindings(bound)
    return bound, fixtures, rows, old


def acquire(h, panel, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve() and "anaconda3" not in sys.executable,
              "OFFPY required, never AP/threadpoolctl")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10
              and {k: os.environ.get(k) for k in BLAS} == dict.fromkeys(BLAS, "1")
              and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "nice10/BLAS1/no bytecode")
    h.require({n: importlib.metadata.version(n) for n in VERSIONS} == VERSIONS, "Native version mismatch")
    sources, fixtures, rows, old = prerequisites(h, panel)
    started = dict(schema=SCHEMA, utc=h.utc(), pid=os.getpid(), executable=sys.executable,
                   nice=os.getpriority(os.PRIO_PROCESS, 0), versions=VERSIONS,
                   blas_env={k: os.environ.get(k) for k in BLAS}, source_bindings=sources,
                   fixture_bindings=fixtures, sequence=[r["rank"] for r in rows]+[35],
                   candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA,
                   initialization_once=True, instrumented=False, no_warmup=True,
                   truth_files_accessed=False, archived_runtime_not_contemporaneous=True)
    start_sha = h.write_json(directory / "started.json", started)
    result = dict(schema=SCHEMA, passed=False, started=started, started_sha256=start_sha,
                  candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA,
                  source_bindings=sources, fixture_bindings=fixtures, versions=VERSIONS,
                  trials=[], artifacts={}, truth_files_accessed=False,
                  production_residual_validated=False)
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        module = load(HERE / "estimator.py", "unused_fixed_estimator", CANDIDATE_SHA)
        estimator = None
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "BLAS count")
            result["threadpools"] = pools
            for call, row in enumerate(rows + [rows[0]]):
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16,1024,1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"unused-material-{row['rank']}")
                began = h.utc(); begin = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16,1024), "Concrete served FP32(16,1024)")
                    materialized = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter()-begin
                h.require(served.tobytes() == materialized.tobytes(), "Materializer changed served bytes")
                # Base NumPy serialization after the unchanged full budget boundary.
                out = np.frombuffer(materialized.tobytes(), dtype=np.float32).reshape(16,1024)
                h.assert_output(np, out)
                prior = old["trials"][call if call < 14 else 0]
                parent_flops = prior["flops"] - (INITIALIZATION if call == 14 else 0)
                ref = np.load(ARCHIVE / prior["output_artifact"], allow_pickle=False)
                h.assert_output(np, ref)
                h.require(hashlib.sha256(ref.tobytes()).hexdigest() == prior["output_bytes_sha256"], "Parent output bytes")
                name = f"call_{call:02d}_rank{row['rank']}_output.npy"
                result["artifacts"][name] = h.save_npy(np, directory / name, out)
                trial = dict(call=call, rank=row["rank"], began_utc=began, ended_utc=h.utc(),
                             flops=int(ctx.flops_used), parent_flops=parent_flops,
                             saved_flops=parent_flops-int(ctx.flops_used), context_wall_s=elapsed,
                             meter_wall_s=float(ctx.wall_time_s), backend_s=float(ctx.flopscope_backend_time_s),
                             overhead_s=float(ctx.flopscope_overhead_time_s),
                             native_residual_s=float(ctx.residual_wall_time_s), rss_bytes=h.peak_rss_bytes(),
                             output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                             all16_byte_exact=out.tobytes() == ref.tobytes(),
                             includes_initialization=call == 0)
                result["trials"].append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); result["artifacts"][bill] = h.binding(directory / bill)
                print(json.dumps(trial), flush=True)
                h.require(trial["all16_byte_exact"] and 0 < trial["flops"] < parent_flops <= h.BUDGET,
                          "Exact all16 bytes or strict complete-cost gate failed")
                h.require(0 < elapsed < 120 and 0 < trial["meter_wall_s"] < 120
                          and trial["rss_bytes"] < RSS_LIMIT
                          and all(math.isfinite(trial[k]) and trial[k] >= 0 for k in
                                  ("backend_s", "overhead_s", "native_residual_s")), "Native resource gate failed")
                del weights, mlp, served, materialized, out, ref
        h.require(result["trials"][0]["flops"]-result["trials"][-1]["flops"] == INITIALIZATION
                  and result["trials"][0]["output_bytes_sha256"] == result["trials"][-1]["output_bytes_sha256"],
                  "Repeat/initialization changed")
        for items in (sources, fixtures, old["artifacts"], result["artifacts"]):
            h.check_bindings(items)
        sealed = h.write_json(directory / "predictions_and_bills_sealed.json", dict(
            utc=h.utc(), candidate_sha256=CANDIDATE_SHA, source_bindings=sources,
            fixture_bindings=fixtures, artifacts=result["artifacts"], trials=result["trials"],
            truth_files_accessed=False))
        result.update(passed=True, hashes_unchanged=True, outputs_seal_sha256=sealed)
    except BaseException:
        result["error"] = traceback.format_exc()
    result["completed_utc"] = h.utc()
    seal = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(passed=result["passed"], result_sha256=seal))
    print(json.dumps(dict(passed=result["passed"], result_sha256=seal, error=result.get("error"))), flush=True)
    return 0 if result["passed"] else 1


def score(h, panel, directory):
    sources, fixtures, rows, old = prerequisites(h, panel)
    acquisition = HERE / "native"
    r = h.load_json(acquisition / "result.json")
    completion = h.load_json(acquisition / "completion.json")
    h.require(completion["passed"] and completion["result_sha256"] == h.digest(acquisition / "result.json")
              and r["passed"] and r["hashes_unchanged"] and r["source_bindings"] == sources
              and r["fixture_bindings"] == fixtures and len(r["trials"]) == 15,
              "Passing fixed acquisition required")
    h.require(h.digest(acquisition / "predictions_and_bills_sealed.json") == r["outputs_seal_sha256"], "Output seal")
    h.check_bindings(r["artifacts"])
    h.write_json(directory / "output_identity_verified_before_score_receipt.json", dict(
        utc=h.utc(), acquisition_sha256=h.digest(acquisition / "result.json"), truth_files_accessed=False))
    h.require(h.digest(OLD_SCORE) == SCORE_SHA, "Qualified same-reference score changed")
    baseline = h.load_json(OLD_SCORE)
    h.require(baseline["qualification_passed"] and baseline["hashes_unchanged"]
              and baseline["candidate_sha256"] == PARENT_SHA
              and h.load_json(OLD_SCORE.with_name("completion.json"))["result_sha256"] == SCORE_SHA,
              "Qualified same-byte raw reference required")
    scored = []
    for t, p in zip(r["trials"][:14], baseline["per_role"]["candidate"], strict=True):
        h.require(t["rank"] == p["rank"] and t["all16_byte_exact"]
                  and t["output_bytes_sha256"] == p["output_bytes_sha256"]
                  and t["parent_flops"] == p["flops"], "Same-output/same-reference frame failed")
        multiplier = max(.1, t["flops"]/h.BUDGET)
        scored.append(dict(rank=t["rank"], flops=t["flops"], saved_flops=t["saved_flops"],
                           raw_noise_corrected=p["raw_noise_corrected"], raw_observed=p["raw_observed"],
                           multiplier=multiplier, adjusted_noise_corrected=p["raw_noise_corrected"]*multiplier,
                           parent_adjusted=p["adjusted_noise_corrected"],
                           raw_provenance="Exact all16 output-byte identity to authenticated parent scored row"))
    summaries = {}
    for name, ranks in (("public8", panel.PUBLIC), ("virgin6", panel.VIRGIN), ("combined14", panel.RANKS)):
        items = [t for t in scored if t["rank"] in ranks]
        value = {"mean_"+k:sum(t[k] for t in items)/len(items) for k in
                 ("raw_noise_corrected", "raw_observed", "flops", "saved_flops", "adjusted_noise_corrected", "parent_adjusted")}
        value["adjusted_ratio"] = value["mean_adjusted_noise_corrected"]/value["mean_parent_adjusted"]
        h.require(value["adjusted_ratio"] < 1, "Group adjusted gate failed")
        summaries[name] = dict(n=len(items), **value)
    for items in (sources, fixtures, r["artifacts"]): h.check_bindings(items)
    result = dict(passed=True, schema=SCHEMA, completed_utc=h.utc(), candidate_sha256=CANDIDATE_SHA,
                  parent_sha256=PARENT_SHA, per_net=scored, summaries=summaries,
                  acquisition_sha256=h.digest(acquisition / "result.json"), parent_score_sha256=SCORE_SHA,
                  source_bindings=sources, fixture_bindings=fixtures,
                  raw_reused_by_exact_output_identity=True, truth_files_accessed=False,
                  frame="Existing independent4M public8/virgin6, not new unused holdout or official",
                  production_residual_validated=False, fully_qualified_component=False,
                  official_score=False, milestone_achieved=False, goal_achieved=False)
    seal = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(passed=True, result_sha256=seal))
    print(json.dumps(dict(summaries=summaries, result_sha256=seal)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--acquire", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if not args.acquire and not args.score: parser.print_help(); return 0
    h, panel = helpers()
    directory = HERE / ("native" if args.acquire else "score")
    directory.mkdir(exist_ok=False)
    try:
        return acquire(h, panel, directory) if args.acquire else score(h, panel, directory)
    except BaseException:
        h.write_json(directory / "preflight_failure.json", dict(passed=False, utc=h.utc(), error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
