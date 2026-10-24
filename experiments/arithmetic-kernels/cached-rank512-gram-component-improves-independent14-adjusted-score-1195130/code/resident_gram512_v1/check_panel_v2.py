"""INERT fixed resident-Gram independent14: --acquire, then separate --score.

One frozen candidate, one fresh OFFPY nice10/BLAS1 worker, one estimator
initialization in call0, and no warmup/reset/instrumentation. Reuse the sealed
qualified terminal-dead parent. Require all16 output drift <=1e-5, strictly
smaller COMPLETE FLOPs on every net, and first2 bytes/bills identical to the
frozen changed development calls. This is not a byte-exact parent rewrite.

All outputs and bills are sealed before any independent4M truth-file hashing
or decoding. Existing score JSON is provenance only during acquisition.
Score mean PER-NET unclipped NCraw*max(.1,completeF/2^41); retain only if BOTH
public8 and virgin6 improve. No minimum percentage, raw-only gate, fitting,
retry, variant, production qualification, milestone, or submission is implied.

V2 is a pre-score serialization repair: v1 stopped before any trial was sealed
because materialized.copy() retained FlopscopeArray. After the unchanged budget
boundary, base-NumPy byte views preserve dtype/values before exclusive saving.
The failed v1 source and artifacts remain untouched; no candidate or gate changed.
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
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = HERE / "panel_v2"
DEAD = NEW / "terminal_dead_consumers_v1"
ARCHIVED = DEAD / "panel/candidate"
ARCHIVED_SCORE = DEAD / "panel/score"
SCHEMA = "resident_gram512_fixed14_v2"
PUBLIC = (35, 100, 150, 200, 250, 300, 400, 500)
VIRGIN = (3016500, 3016501, 3016502, 3016503, 3016504, 3016505)
RANKS = PUBLIC + VIRGIN
CANDIDATE_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"
PARENT_SHA = "07b71a03e13f8daed17e0ee529c4113e88cd89c2cceb29ea0e2f148084e74d30"
BUILD_SHA = "3f04e57a820508928f409ddb45083cfcb7dee7499f6d62a5d85b6545ca267da3"
DEV_CHECKER_SHA = "69a1d802d812bdbe591e3de00aa12392e3db1cc208fc23565b406335f7123f14"
DEAD_CHECKER_SHA = "53a3441c383c83ce1ba9b0ef61ab64cabc1390aa61adfa1efbf95502c8aacabc"
DEV_SCORE_SHA = "fd015efa3d31488f7f15b940d9ceec0d2f7020b7dd1cbdc4cb262479e4154866"
PARENT_RESULT_SHA = "302d9bbaf1327e0f12aa8746406536e24ed85beca6ec23eb807c8a2a03cd7344"
PARENT_SCORE_SHA = "91dfa3cd9067433c738c51ed5aec5d0fc5b807f2efd6ee468331b5f54fe076f9"
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
VERSIONS = {"flopscope": "0.12.0", "whestbench": "0.16.0", "numpy": "2.4.6", "threadpoolctl": "3.6.0"}
RSS_LIMIT = 8_000_000_000
WALL_LIMIT = 120
DRIFT_LIMIT = 1e-5
OUTPUT_VALIDATION = (
    "Original concrete served and official materialized values are both checked "
    "after context exit. Shape/dtype assertion and official materialization occur "
    "inside. No deferred estimator work, cast, clamp, or repair; harness scans "
    "and archived-output comparisons are outside the estimator FLOP boundary."
)


def load_module(path, alias, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Frozen inert verifier changed: " + str(path))
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load inert verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Verifier changed during import")
    return module


def modules():
    """Only authenticated inert utility definitions; no candidate/array import."""
    dead = load_module(DEAD / "check_panel.py", "gram_panel_dead_verifier", DEAD_CHECKER_SHA)
    h, panel, _, _ = dead.modules()
    dev = load_module(HERE / "check_dev.py", "gram_panel_development_verifier", DEV_CHECKER_SHA)
    return h, panel, dev, dead


def prerequisites(h, panel, dev, dead):
    h.require(panel.PUBLIC == dead.PUBLIC == PUBLIC and panel.VIRGIN == dead.VIRGIN == VIRGIN
              and panel.RANKS == dead.RANKS == RANKS, "Canonical14 sequence changed")
    h.require(h.digest(ARCHIVED / "result.json") == PARENT_RESULT_SHA
              and h.digest(ARCHIVED_SCORE / "result.json") == PARENT_SCORE_SHA
              and h.digest(HERE / "dev/score/result.json") == DEV_SCORE_SHA,
              "Frozen passing parent/development receipt changed")
    parent_header = h.load_json(ARCHIVED / "result.json")
    # The fully authenticated legacy verifier hashes outputs, not truth arrays.
    _, old_panel, old_dev, previous = dead.modules()
    baseline, _, _ = dead.verify_acquisition(
        h, old_panel, old_dev, previous, SimpleNamespace(**parent_header["provenance_pins"]))
    bound, build, _ = dev.sources(h, BUILD_SHA)
    development = {role: dev.verify(h, role, BUILD_SHA) for role in ("null", "candidate")}
    d0, d1 = development["null"], development["candidate"]
    h.require(build["candidate_sha256"] == d1["candidate_sha256"] == CANDIDATE_SHA
              and build["parent_sha256"] == d0["candidate_sha256"] == baseline["candidate_sha256"] == PARENT_SHA
              and Path(build["candidate"]).resolve() == (HERE / "estimator.py").resolve()
              and Path(build["parent"]).resolve() == (DEAD / "estimator.py").resolve(),
              "Different parent/head/build/candidate frame")
    h.require(d0["pid"] != d1["pid"] and d0["completed_utc"] <= d1["started"]["utc"]
              and baseline["versions"] == d0["versions"] == d1["versions"] == VERSIONS
              and d0["fixture_bindings"] == d1["fixture_bindings"]
              and all(value == baseline["fixture_bindings"][key] for key, value in d0["fixture_bindings"].items()),
              "Development process/fixture/runtime compatibility failed")
    for call in (0, 1):
        p, n = baseline["trials"][call], d0["trials"][call]
        h.require(p["rank"] == n["rank"] == RANKS[call] and p["flops"] == n["flops"]
                  and p["output_bytes_sha256"] == n["output_sha256"], "Fresh null no longer corroborates parent")
    dev_score = h.load_json(HERE / "dev/score/result.json")
    dc = h.load_json(HERE / "dev/score/completion.json")
    dev_seals = {f"{role}/{name}": h.binding(HERE / "dev" / role / name)
                 for role in development for name in ("started.json", "result.json", "completion.json")}
    h.require(dc["passed"] and dc["advances_to_independent14"] and dc["result_sha256"] == DEV_SCORE_SHA
              and dev_score["passed"] and dev_score["advances_to_independent14"]
              and dev_score["role_seals"] == dev_seals
              and dev_score["source_bindings"] == bound
              and dev_score["roles"]["candidate"]["candidate_sha256"] == CANDIDATE_SHA
              and dev_score["roles"]["null"]["candidate_sha256"] == PARENT_SHA
              and len(dev_score["per_net_adjusted_ratios"]) == 2
              and all(math.isfinite(v) and 0 < v < 1 for v in dev_score["per_net_adjusted_ratios"])
              and 0 < dev_score["mean_adjusted_ratio"] < 1, "Development objective/seal did not qualify")
    old_score = h.load_json(ARCHIVED_SCORE / "result.json")
    pc = h.load_json(ARCHIVED_SCORE / "completion.json")
    h.require(pc["qualification_passed"] and pc["retain_local_component"] and pc["result_sha256"] == PARENT_SCORE_SHA
              and old_score["qualification_passed"] and old_score["hashes_unchanged"]
              and old_score["retain_local_component"] and old_score["candidate_sha256"] == PARENT_SHA
              and old_score["role_seals"]["candidate/result.json"]["sha256"] == PARENT_RESULT_SHA,
              "Reused independent14 parent is not the retained qualified score")
    sources = {"gram/" + key: value for key, value in bound.items()}
    sources.update({"parent_source/" + key: value for key, value in baseline["source_bindings"].items()})
    sources.update({"dev_seal/" + key: value for key, value in dev_seals.items()})
    for role, record in development.items():
        sources.update({"dev_artifact/" + role + "/" + key: value for key, value in record["artifacts"].items()})
    for label, directory, names in (
        ("parent", ARCHIVED, ("started.json", "result.json", "completion.json", "predictions_and_bills_sealed.json")),
        ("parent_score", ARCHIVED_SCORE, ("result.json", "completion.json")),
        ("dev_score", HERE / "dev/score", ("result.json", "completion.json", "both_roles_verified_before_truth.json", "truth_access_started.json")),
    ):
        sources.update({label + "/" + name: h.binding(directory / name) for name in names})
    sources["panel_checker"] = h.binding(__file__)
    sources["parent_panel_checker"] = h.binding(DEAD / "check_panel.py")
    h.require(sources["parent_panel_checker"]["sha256"] == DEAD_CHECKER_SHA
              and h.digest(HERE / "check_dev.py") == DEV_CHECKER_SHA, "Verifier pin changed")
    h.check_bindings(sources); h.check_bindings(baseline["artifacts"])
    return sources, baseline, development, old_score


def load_output(h, np, directory, record, call, development=False):
    trial = record["trials"][call]
    name = trial["output" if development else "output_artifact"]
    path = directory / name
    h.require(Path(record["artifacts"][name]["path"]).resolve() == path.resolve()
              and h.digest(path) == record["artifacts"][name]["sha256"], "Sealed output file changed/escaped")
    value = np.load(path, allow_pickle=False)
    h.assert_output(np, value)
    h.require(hashlib.sha256(value.tobytes()).hexdigest()
              == trial["output_sha256" if development else "output_bytes_sha256"], "Served-byte identity changed")
    return value


def valid_trial(h, trial, call):
    times = ("context_wall_s", "meter_wall_s", "flopscope_backend_s", "flopscope_overhead_s", "native_residual_s")
    h.require(trial["call"] == call and trial["rank"] == RANKS[call]
              and type(trial["flops"]) is int and 0 < trial["flops"] <= h.BUDGET
              and all(math.isfinite(trial[key]) and trial[key] >= 0 for key in times)
              and 0 < trial["context_wall_s"] < WALL_LIMIT and 0 < trial["meter_wall_s"] < WALL_LIMIT
              and 0 < trial["peak_rss_bytes"] < RSS_LIMIT
              and trial["includes_first_call_initialization"] == (call == 0)
              and trial["served_materialized_byte_exact"], "Complete native output/bill/resource gate failed")


def comparisons_pass(h, comparison, trial, parent_trial, dev_candidate, call):
    errors = comparison["per_layer_max_abs"]
    h.require(comparison["call"] == trial["call"] == call
              and comparison["rank"] == trial["rank"] == parent_trial["rank"] == RANKS[call]
              and len(errors) == 16 and all(math.isfinite(v) and 0 <= v <= DRIFT_LIMIT for v in errors)
              and comparison["numerical_passed"] and comparison["strict_complete_cost_saving"]
              and comparison["parent_flops"] == parent_trial["flops"]
              and comparison["saved_flops"] == parent_trial["flops"] - trial["flops"] > 0,
              "All16 fidelity or strictly lower per-net complete bill failed")
    if call < 2:
        d = dev_candidate["trials"][call]
        h.require(comparison["development_output_and_bill_exact"] and trial["flops"] == d["flops"]
                  and trial["output_bytes_sha256"] == d["output_sha256"], "Changed development bytes/full bill differ")


def acquire(h, panel, dev, dead, directory):
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable and "anaconda3" not in sys.prefix, "Use OFFPY, never AP/threadpoolctl")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    h.require(nice >= 10 and env == dict.fromkeys(BLAS_ENV, "1")
              and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Require nice10/BLAS1/no bytecode before imports")
    sources, baseline, development, _ = prerequisites(h, panel, dev, dead)
    fixtures, rows = panel.fixtures(h)
    versions = {key: importlib.metadata.version(key) for key in VERSIONS}
    h.require(fixtures == baseline["fixture_bindings"] and [row["rank"] for row in rows] == list(RANKS)
              and versions == VERSIONS and os.getpid() not in (baseline["pid"], *(r["pid"] for r in development.values())),
              "Fresh same-environment canonical14 frame required")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    from whestbench.scoring import materialise_predictions
    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, role="candidate", utc=h.utc(), pid=os.getpid(), executable=sys.executable,
                   nice=nice, versions=versions, blas_env=env, sequence=list(RANKS), source_bindings=sources,
                   fixture_bindings=fixtures, candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA,
                   baseline_result_sha256=PARENT_RESULT_SHA, development_score_sha256=DEV_SCORE_SHA,
                   one_estimator_instance=True, initialization_once_in_call0=True, no_warmup=True, no_reset=True,
                   instrumented=False, truth_files_accessed=False, archived_parent_reused=True,
                   output_validation=OUTPUT_VALIDATION, flop_budget=h.BUDGET, wall_limit_s=WALL_LIMIT,
                   rss_limit_bytes=RSS_LIMIT, production_residual_validated=False)
    start_sha = h.write_json(directory / "started.json", started)
    trials, artifacts, comparisons = [], {}, []
    result = dict(schema=SCHEMA, role="candidate", completed=False, qualification_passed=False,
                  pid=os.getpid(), started=started, started_sha256=start_sha, versions=versions, blas_env=env,
                  source_bindings=sources, fixture_bindings=fixtures, candidate_sha256=CANDIDATE_SHA,
                  parent_sha256=PARENT_SHA, baseline_result_sha256=PARENT_RESULT_SHA,
                  development_score_sha256=DEV_SCORE_SHA, instrumented=False, truth_files_accessed=False,
                  output_validation=OUTPUT_VALIDATION, archived_runtime_not_contemporaneous=True,
                  production_residual_validated=False)
    try:
        t0 = time.perf_counter()
        module = load_module(HERE / "estimator.py", "resident_gram_fixed14_unwrapped_candidate", CANDIDATE_SHA)
        result["candidate_import_wall_s"] = time.perf_counter() - t0
        estimator = None
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"), "Reported BLAS count !=1")
            result["threadpools"] = pools
            for call, row in enumerate(rows):
                weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
                h.require(weights.shape == (16, 1024, 1024) and weights.dtype.kind == "f"
                          and bool(np.isfinite(weights).all()), "Invalid canonical Phase2 weights")
                mlp = MLP(width=1024, depth=16, weights=list(weights),
                          seed=int(derive_estimator_seed(row["mlp_seed"])), name=f"resident-gram-panel-{row['rank']}")
                began = h.utc(); t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=WALL_LIMIT, quiet=True) as ctx:
                    if estimator is None:
                        estimator = module.Estimator()
                    served = estimator.predict(mlp, h.BUDGET)
                    h.require(isinstance(served, np.ndarray) and served.dtype == np.dtype("float32")
                              and served.shape == (16, 1024), "Concrete served float32(16,1024), without repair")
                    materialized = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter() - t0
                original = np.asarray(served)
                returned = np.asarray(materialized)
                h.assert_output(np, original); h.assert_output(np, returned)
                h.require(original.tobytes() == returned.tobytes(), "Materializer changed original served bytes")
                out = returned.copy()
                name = f"call_{call:02d}_rank{row['rank']}_output.npy"
                artifacts[name] = h.save_npy(np, directory / name, out)
                trial = dict(call=call, rank=row["rank"], began_utc=began, ended_utc=h.utc(), flops=int(ctx.flops_used),
                             context_wall_s=elapsed, meter_wall_s=float(ctx.wall_time_s),
                             flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                             flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                             native_residual_s=float(ctx.residual_wall_time_s), peak_rss_bytes=h.peak_rss_bytes(),
                             output_artifact=name, output_bytes_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
                             includes_first_call_initialization=call == 0, served_materialized_byte_exact=True)
                trials.append(trial)
                bill = f"call_{call:02d}_bill.json"
                h.write_json(directory / bill, trial); artifacts[bill] = h.binding(directory / bill)
                valid_trial(h, trial, call)
                ref = load_output(h, np, ARCHIVED, baseline, call)
                errors = np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64)), axis=1)
                comparison = dict(call=call, rank=row["rank"], per_layer_max_abs=errors.tolist(),
                                  all16_byte_exact=out.tobytes() == ref.tobytes(), numerical_passed=bool((errors <= DRIFT_LIMIT).all()),
                                  parent_flops=baseline["trials"][call]["flops"],
                                  saved_flops=baseline["trials"][call]["flops"] - trial["flops"],
                                  strict_complete_cost_saving=trial["flops"] < baseline["trials"][call]["flops"])
                if call < 2:
                    ref_dev = load_output(h, np, HERE / "dev/candidate", development["candidate"], call, True)
                    comparison["development_output_and_bill_exact"] = (
                        out.tobytes() == ref_dev.tobytes() and trial["flops"] == development["candidate"]["trials"][call]["flops"])
                    del ref_dev
                comparisons.append(comparison)
                comparisons_pass(h, comparison, trial, baseline["trials"][call], development["candidate"], call)
                print(json.dumps(dict(role="candidate", **trial, comparison=comparison)), flush=True)
                del weights, mlp, served, materialized, original, returned, out, ref, errors
        for items in (sources, fixtures, artifacts, baseline["artifacts"]):
            h.check_bindings(items)
        result["final_peak_rss_bytes"] = h.peak_rss_bytes()
        h.require(result["final_peak_rss_bytes"] < RSS_LIMIT, "Final acquisition RSS cap exceeded")
        seal = dict(schema=SCHEMA, sealed_utc=h.utc(), source_bindings=sources, fixture_bindings=fixtures,
                    candidate_sha256=CANDIDATE_SHA, baseline_result_sha256=PARENT_RESULT_SHA,
                    development_score_sha256=DEV_SCORE_SHA, trials=trials, artifacts=artifacts,
                    comparisons=comparisons, truth_files_accessed=False)
        sealed_sha = h.write_json(directory / "predictions_and_bills_sealed.json", seal)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True, compatibility_passed=True,
                      strictly_lower_complete_flops_all14=True, predictions_and_bills_seal_sha256=sealed_sha)
    except BaseException:
        result["error"] = traceback.format_exc()
    result.update(trials=trials, artifacts=artifacts, comparisons=comparisons, completed_utc=h.utc())
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, role="candidate", completed_utc=h.utc(),
                 qualification_passed=result["qualification_passed"], result_sha256=result_sha, truth_files_accessed=False))
    print(json.dumps(dict(qualification_passed=result["qualification_passed"], result_sha256=result_sha,
                          error=result.get("error"))), flush=True)
    return 0 if result["qualification_passed"] else 1


def verify_acquisition(h, panel, dev, dead):
    """JSON/hash-only public verifier; returns (record, canonical_rows, seals)."""
    sources, baseline, development, _ = prerequisites(h, panel, dev, dead)
    directory = ROOT / "candidate"
    complete = h.load_json(directory / "completion.json")
    h.require(complete["schema"] == SCHEMA and complete["role"] == "candidate" and complete["qualification_passed"]
              and not complete["truth_files_accessed"] and complete["result_sha256"] == h.digest(directory / "result.json"),
              "Missing qualified native14 acquisition seal")
    record = h.load_json(directory / "result.json")
    h.require(record["schema"] == SCHEMA and record["role"] == "candidate" and record["completed"]
              and record["qualification_passed"] and record["hashes_unchanged"] and record["compatibility_passed"]
              and record["strictly_lower_complete_flops_all14"] and not record["truth_files_accessed"]
              and not record["instrumented"] and record["source_bindings"] == sources
              and record["candidate_sha256"] == CANDIDATE_SHA and record["parent_sha256"] == PARENT_SHA
              and record["baseline_result_sha256"] == PARENT_RESULT_SHA
              and record["development_score_sha256"] == DEV_SCORE_SHA, "Invalid resident-Gram native14 record")
    start = h.load_json(directory / "started.json")
    h.require(start == record["started"] and h.digest(directory / "started.json") == record["started_sha256"]
              and start["pid"] == record["pid"] and start["nice"] >= 10 and start["sequence"] == list(RANKS)
              and start["source_bindings"] == sources and start["versions"] == record["versions"] == VERSIONS
              and start["blas_env"] == record["blas_env"] == dict.fromkeys(BLAS_ENV, "1")
              and start["one_estimator_instance"] and start["initialization_once_in_call0"]
              and start["no_warmup"] and start["no_reset"] and not start["instrumented"]
              and not start["truth_files_accessed"] and start["output_validation"] == record["output_validation"] == OUTPUT_VALIDATION
              and record["pid"] not in (baseline["pid"], *(r["pid"] for r in development.values()))
              and 0 < record["final_peak_rss_bytes"] < RSS_LIMIT
              and math.isfinite(record["candidate_import_wall_s"]) and record["candidate_import_wall_s"] >= 0,
              "Native process/initialization/resource boundary changed")
    fixtures, rows = panel.fixtures(h)
    h.require(record["fixture_bindings"] == start["fixture_bindings"] == fixtures == baseline["fixture_bindings"]
              and [row["rank"] for row in rows] == list(RANKS)
              and [t["rank"] for t in record["trials"]] == list(RANKS)
              and len(record["comparisons"]) == 14, "Incomplete or changed14-net fixtures")
    expected = set()
    for call, trial in enumerate(record["trials"]):
        valid_trial(h, trial, call)
        name = f"call_{call:02d}_rank{RANKS[call]}_output.npy"
        bill = f"call_{call:02d}_bill.json"; expected.update((name, bill))
        h.require(trial["output_artifact"] == name and h.load_json(directory / bill) == trial, "Per-call complete bill changed")
        comparisons_pass(h, record["comparisons"][call], trial, baseline["trials"][call], development["candidate"], call)
    h.require(set(record["artifacts"]) == expected, "All28 output/bill artifacts must be sealed")
    for name, item in record["artifacts"].items():
        h.require(Path(item["path"]).resolve() == (directory / name).resolve(), "Panel artifact escaped fixed directory")
    h.check_bindings(record["artifacts"]); h.check_bindings(fixtures)
    path = directory / "predictions_and_bills_sealed.json"
    h.require(h.digest(path) == record["predictions_and_bills_seal_sha256"], "Prediction/bill seal changed")
    seal = h.load_json(path)
    h.require(seal["schema"] == SCHEMA and not seal["truth_files_accessed"]
              and seal["source_bindings"] == sources and seal["fixture_bindings"] == fixtures
              and seal["trials"] == record["trials"] and seal["artifacts"] == record["artifacts"]
              and seal["comparisons"] == record["comparisons"] and seal["candidate_sha256"] == CANDIDATE_SHA
              and seal["baseline_result_sha256"] == PARENT_RESULT_SHA and seal["development_score_sha256"] == DEV_SCORE_SHA
              and start["utc"] <= seal["sealed_utc"] <= record["completed_utc"] <= complete["completed_utc"],
              "Incomplete or misordered pre-truth seal")
    seals = {name: h.binding(directory / name) for name in
             ("started.json", "result.json", "completion.json", "predictions_and_bills_sealed.json")}
    return record, rows, seals


def score(h, panel, dev, dead, directory):
    candidate, _, candidate_seals = verify_acquisition(h, panel, dev, dead)
    _, baseline, development, old_score = prerequisites(h, panel, dev, dead)
    import numpy as np
    h.require(importlib.metadata.version("numpy") == VERSIONS["numpy"], "Scoring NumPy frame changed")
    records = {"parent": baseline, "candidate": candidate}
    paths = {"parent": ARCHIVED, "candidate": ROOT / "candidate"}
    predictions = {role: [load_output(h, np, paths[role], record, i) for i in range(14)] for role, record in records.items()}
    for call in range(14):
        errors = np.max(np.abs(predictions["candidate"][call].astype(np.float64)
                               - predictions["parent"][call].astype(np.float64)), axis=1)
        h.require(errors.tolist() == candidate["comparisons"][call]["per_layer_max_abs"]
                  and bool((errors <= DRIFT_LIMIT).all()), "Actual sealed trajectory differs from acquisition")
        if call < 2:
            ref = load_output(h, np, HERE / "dev/candidate", development["candidate"], call, True)
            h.require(ref.tobytes() == predictions["candidate"][call].tobytes(), "Actual development bytes changed")
    role_seals = {"candidate/" + key: value for key, value in candidate_seals.items()}
    role_seals.update({"parent/" + name: h.binding(ARCHIVED / name) for name in candidate_seals})
    before = h.write_json(directory / "both_roles_verified_before_truth.json", dict(
        schema=SCHEMA, utc=h.utc(), role_seals=role_seals, source_bindings=candidate["source_bindings"],
        fixture_bindings=candidate["fixture_bindings"], ranks=list(RANKS), truth_files_accessed=False))
    truth_time = h.utc()
    access_sha = h.write_json(directory / "truth_access_started.json", dict(utc=truth_time, prior_seal_sha256=before))
    truth_bindings, per_role = {}, {role: [] for role in records}
    for call, rank in enumerate(RANKS):
        path = panel.G19 / (f"mps_truth_rank{rank:04d}_indep4M.npz" if rank in PUBLIC else f"fresh_{rank}_truth4M.npz")
        truth_bindings[str(rank)] = h.binding(path)
        h.require(truth_bindings[str(rank)] == old_score["truth_bindings"][str(rank)], "Independent4M truth binding changed")
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
                        includes_first_call_initialization=call == 0, raw_observed=raw, mc_se_squared_mean=noise,
                        raw_noise_corrected=nc, adjusted_observed=raw * multiplier, adjusted_noise_corrected=nc * multiplier,
                        output_bytes_sha256=trial["output_bytes_sha256"])
            h.require(all(math.isfinite(item[key]) for key in ("raw_observed", "mc_se_squared_mean",
                      "raw_noise_corrected", "adjusted_observed", "adjusted_noise_corrected")), "Nonfinite score")
            if role == "parent":
                old = old_score["per_role"]["candidate"][call]
                h.require(old["rank"] == rank and all(item[key] == old[key] for key in (
                    "flops", "raw_observed", "mc_se_squared_mean", "raw_noise_corrected", "multiplier",
                    "adjusted_observed", "adjusted_noise_corrected", "output_bytes_sha256")),
                    "Recomputed retained parent differs from its authenticated independent14 score")
            per_role[role].append(item)
    summaries = {}
    for role, values in per_role.items():
        summaries[role] = {}
        for group, ranks in (("public8", PUBLIC), ("virgin6", VIRGIN), ("combined14", RANKS)):
            subset = [item for item in values if item["rank"] in ranks]
            fields = ("raw_observed", "raw_noise_corrected", "mc_se_squared_mean", "flops", "adjusted_observed", "adjusted_noise_corrected")
            summaries[role][group] = dict(n=len(subset), **{
                "mean_" + key: sum(item[key] for item in subset) / len(subset) for key in fields})
    ratios, gates = {}, {}
    for group in ("public8", "virgin6", "combined14"):
        denominator = summaries["parent"][group]["mean_adjusted_noise_corrected"]
        h.require(math.isfinite(denominator) and denominator > 0, "Invalid parent group adjusted denominator")
        ratios[group] = summaries["candidate"][group]["mean_adjusted_noise_corrected"] / denominator
        h.require(math.isfinite(ratios[group]), "Nonfinite group adjusted ratio")
        if group != "combined14":
            gates[group] = ratios[group] < 1.0
    h.check_bindings(role_seals); h.check_bindings(truth_bindings)
    for record in records.values():
        for key in ("source_bindings", "fixture_bindings", "artifacts"):
            h.check_bindings(record[key])
    retain = all(gates.values())
    result = dict(schema=SCHEMA, completed=True, qualification_passed=True, hashes_unchanged=True,
                  completed_utc=h.utc(), source_bindings=candidate["source_bindings"], role_seals=role_seals,
                  truth_bindings=truth_bindings, both_roles_verified_before_truth_sha256=before,
                  truth_access_started_utc=truth_time, truth_access_started_sha256=access_sha,
                  per_role=per_role, summaries=summaries, adjusted_candidate_over_parent=ratios,
                  group_strict_gain_gate=gates, all16_drift_within_1e5_all14=True,
                  maximum_absolute_drift=max(max(c["per_layer_max_abs"]) for c in candidate["comparisons"]),
                  all16_byte_exact_all14=all(c["all16_byte_exact"] for c in candidate["comparisons"]),
                  strictly_lower_complete_flops_all14=True, first_two_development_bytes_and_bills_exact=True,
                  retain_local_component=retain, baseline_recomputed_exact=True,
                  scientific_status="component_pass" if retain else "group_score_gate_fail",
                  candidate_sha256=CANDIDATE_SHA, parent_sha256=PARENT_SHA, development_score_sha256=DEV_SCORE_SHA,
                  aggregation="Mean per-net (MSE-mean(mc_se15^2))*max(.1,actual completeF/2^41); no clipping; init once in call0",
                  frame="Same existing independent4M public8/virgin6; not unused holdout or official score",
                  archived_runtime_not_contemporaneous=True, production_residual_validated=False,
                  official_score=False, milestone_achieved=False, goal_achieved=False)
    result_sha = h.write_json(directory / "result.json", result)
    h.write_json(directory / "completion.json", dict(schema=SCHEMA, completed_utc=h.utc(), qualification_passed=True,
                 result_sha256=result_sha, retain_local_component=retain))
    print(json.dumps(dict(summaries=summaries, adjusted_candidate_over_parent=ratios,
                          retain_local_component=retain, result_sha256=result_sha)), flush=True)
    return 0 if retain else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--acquire", action="store_true")
    modes.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if not args.acquire and not args.score:
        parser.print_help(); return 0
    h, panel, dev, dead = modules()
    if args.acquire:
        ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(), "Start with a new fixed acquisition")
    directory = ROOT / ("candidate" if args.acquire else "score")
    directory.mkdir(exist_ok=False)
    try:
        return acquire(h, panel, dev, dead, directory) if args.acquire else score(h, panel, dev, dead, directory)
    except BaseException:
        h.write_json(directory / "preflight_or_score_failure.json", dict(
            schema=SCHEMA, utc=h.utc(), qualification_passed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
