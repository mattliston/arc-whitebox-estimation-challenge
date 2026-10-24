"""Inert, fixed response-row exception acquisition and later sealed scoring.

Run bank and response in DIFFERENT fresh nice-10 designated-native-environment processes, one at a
time. Each explicit --role acquires only ranks35/100/35 and never opens or
hashes a truth file. --score verifies BOTH completed acquisitions before
any truth access; it imports neither candidate nor FlopScope. No argument
and --help are inert. There is no training, threshold, rank, or head option.

Diagnostic native copies are NOT production residual-time qualification.
The separate unchanged NumPy-free client/server release gate still applies.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
GATE_ENV = HERE.parent / "data_gate/.venv"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TRUTH_DIR = HERE.parent / "public_g22_operator_features_v1/truthpost"
BUDGET = 2**41
RSS_LIMIT = 8 * 1024**3
RANKS = (35, 100, 35)
SCHEMA = "response_rows_adjusted_exception_v1"
PROTOCOL = "RESPONSE_ROWS_ADJUSTED_EXCEPTION_V1.md"
BUILDER = "build_response_rows_current_runtime_v1.py"
BANK = "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
RESPONSE = "candidate_response_rows_current_runtime_v1.py"
PINNED = {
    BANK: "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757",
    RESPONSE: "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    BUILDER: "235cd451a3dc4fc940aefc0c4d92e718412e2c0ecefd9f587412a995957f261d",
    PROTOCOL: "ba7b74f2c00bab570e4b722393a39a56eebc905d19d284f2d69dec2fe4c95cbb",
    "payload_response_rows_v1.npz": "6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c",
    "radial_response_rows_v1.py": "bb496d4a205d4669d35050e1f56560fbd303a8125a215cb560a64ee4e2a785f5",
    "response_rows_closedloop_r35_v1.npz": "8c428c6a7cc8ede17a6b10464719d35513413aec17f086f02809a9a5253f69ca",
    "response_rows_closedloop_r100_v1.npz": "c1e2d0dca91f57e52f92f334404442aad153097ec453b536741280631c3bd194",
    "response_rows_closedloop_r35_v1.json": "88933a29d2f5eda070ce5d72af640233b0969165f2fbfa9eecacdc23494fd794",
    "response_rows_closedloop_r100_v1.json": "99775fe62a638e5313270540086987a66d7e894e9aac2d87aebe3ccd458796df",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return digest(path)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    value = digest(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed while hashing: {path}")
    return dict(path=str(path), sha256=value, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def check_bindings(items):
    for item in items.values():
        require(digest(item["path"]) == item["sha256"],
                f"Frozen file changed: {item['path']}")


def source_bindings(response_sha):
    require(response_sha == PINNED[RESPONSE], "Only the fixed reviewed response source is authorized")
    pins = dict(PINNED)
    result = {name: binding(HERE / name) for name in
              (*pins, Path(__file__).name)}
    for name, expected in pins.items():
        require(result[name]["sha256"] == expected, f"Source pin mismatch: {name}")
    return result


def fixture_bindings():
    manifest_path = STORE / "manifest.json"
    frozen = {"manifest": binding(manifest_path)}
    manifest = load_json(manifest_path)["roles"]["observer_fit"]["row_manifests"]
    rows = []
    for rank in (35, 100):
        entry = manifest[rank] if isinstance(manifest, list) else manifest[str(rank)]
        row_path = (STORE / entry["path"]).resolve()
        require(row_path.is_relative_to(STORE.resolve()), "Fixture path escaped fixed store")
        frozen[f"row_{rank}"] = binding(row_path)
        row = load_json(row_path)
        weights = row_path.parent / "weights.npy"
        frozen[f"weights_{rank}"] = binding(weights)
        rows.append(dict(rank=rank, row_path=str(row_path), weights_path=str(weights),
                         mlp_seed=int(row["mlp_seed"])))
    check_bindings(frozen)
    return frozen, rows


def peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def save_npy(np, path, array):
    with Path(path).open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return binding(path)


def save_diagnostics(np, path, features, deltas):
    with Path(path).open("xb") as stream:
        np.savez(stream, x=features, delta=deltas)
    return binding(path)


def assert_output(np, array):
    require(array.shape == (16, 1024), f"Wrong served shape: {array.shape}")
    require(array.dtype == np.dtype("float32"), f"Wrong served dtype: {array.dtype}")
    require(bool(np.isfinite(array).all()), "Nonfinite served output")
    require(bool((array >= 0).all()), "Served output was not nonnegative")


def acquire(args, directory):
    # Reject AP/Torch environments BEFORE importing threadpoolctl.
    require(Path(sys.prefix).resolve() == GATE_ENV.resolve(),
            "Acquisition requires the designated data_gate/.venv, never AP/Torch")
    require("anaconda3" not in sys.executable and "anaconda3" not in sys.prefix,
            "Never import threadpoolctl in the AP interpreter")
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    require(nice >= 10, "Launch this fixed worker with nice >=10")
    sources = source_bindings(args.response_sha256)
    fixtures, rows = fixture_bindings()
    versions = {name: importlib.metadata.version(name)
                for name in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0",
            "This native gate is pinned to FlopScope0.12.0 / Whest0.16.0")
    import numpy as np
    import flopscope as flops
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed

    flops.configure(symmetry_warnings=False)
    started = dict(schema=SCHEMA, mode="acquire", role=args.role, utc=utc(),
                   pid=os.getpid(), executable=sys.executable, nice=nice,
                   response_sha256=args.response_sha256,
                   protocol_sha256=PINNED[PROTOCOL], source_bindings=sources,
                   fixture_bindings=fixtures, fixture_rows=rows, versions=versions,
                   versions_sha256=canonical_digest(versions), sequence=list(RANKS),
                   blas_threads=1, flop_budget=BUDGET, native_wall_limit_s=120,
                   truth_files_accessed=False, first_call_initialization_included=True)
    start_sha = write_json(directory / "started.json", started)
    predictions, diagnostics, trials, artifacts = [], [], [], {}
    result = dict(schema=SCHEMA, mode="acquire", role=args.role,
                  qualification_passed=False, completed=False,
                  started_sha256=start_sha, truth_files_accessed=False,
                  source_bindings=sources, fixture_bindings=fixtures,
                  response_sha256=args.response_sha256,
                  protocol_sha256=PINNED[PROTOCOL], versions=versions,
                  versions_sha256=canonical_digest(versions), pid=os.getpid())
    try:
        # Same fixtures, order, precision and preparation for both processes.
        prepared = {}
        for row in rows:
            weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
            require(weights.shape == (16, 1024, 1024), "Wrong Phase2 weight shape")
            require(weights.dtype.kind == "f" and bool(np.isfinite(weights).all()),
                    "Invalid Phase2 weights")
            prepared[row["rank"]] = MLP(
                width=1024, depth=16, weights=list(weights),
                seed=int(derive_estimator_seed(row["mlp_seed"])),
                name=f"response-adjusted-fixed-{row['rank']}")
        check_bindings(fixtures)
        candidate_name = BANK if args.role == "bank" else RESPONSE
        imported_at = time.perf_counter()
        spec = importlib.util.spec_from_file_location("fixed_response_adjusted_candidate", HERE / candidate_name)
        require(spec is not None and spec.loader is not None, "Candidate import specification failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["candidate_import_wall_s"] = time.perf_counter() - imported_at
        require(digest(HERE / candidate_name) == sources[candidate_name]["sha256"],
                "Candidate changed during import")
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            require(all(pool["num_threads"] == 1 for pool in pools
                        if pool.get("user_api") == "blas"), "BLAS did not use one thread")
            result["threadpools"] = pools
            for call, rank in enumerate(RANKS):
                captured_layers, captured_x, captured_delta = [], [], []
                instrumented = args.role == "response" and call < 2
                original_step = module.gnn_step if instrumented else None
                if instrumented:
                    def wrapped(l, x, sigma, w, state, payload):
                        # Pre-head x is copied BEFORE the actual head; delta AFTER.
                        captured_layers.append(int(l))
                        captured_x.append(np.asarray(x).copy())
                        value = original_step(l, x, sigma, w, state, payload)
                        captured_delta.append(np.asarray(value[0]).copy())
                        return value
                    module.gnn_step = wrapped
                began_utc = utc()
                t0 = time.perf_counter()
                try:
                    with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120,
                                            quiet=True) as context:
                        returned = module.Estimator().predict(prepared[rank], BUDGET)
                    elapsed = time.perf_counter() - t0
                finally:
                    if instrumented:
                        module.gnn_step = original_step
                # No clamp/cast repair here: preserve and check the served bytes.
                out = np.asarray(returned).copy()
                assert_output(np, out)
                output_name = f"call_{call:02d}_rank{rank}_output.npy"
                artifacts[output_name] = save_npy(np, directory / output_name, out)
                predictions.append(out)
                if instrumented:
                    require(captured_layers == list(range(16)), "Full16 head routing failed")
                    x, delta = np.stack(captured_x), np.stack(captured_delta)
                    require(x.shape == (16, 1024, 30) and delta.shape == (16, 1024),
                            "Wrong response feature/delta shape")
                    require(bool(np.isfinite(x).all()) and bool(np.isfinite(delta).all()),
                            "Nonfinite response diagnostics")
                    name = f"call_{call:02d}_rank{rank}_diagnostics.npz"
                    artifacts[name] = save_diagnostics(np, directory / name, x, delta)
                    diagnostics.append((rank, x, delta))
                flops_used = int(context.flops_used)
                residual = float(context.residual_wall_time_s)
                require(0 < flops_used <= BUDGET and math.isfinite(residual),
                        "Invalid completed native bill")
                trial = dict(call=call, rank=rank, began_utc=began_utc, ended_utc=utc(),
                             flops=flops_used, utilization=flops_used / BUDGET,
                             multiplier=max(0.1, flops_used / BUDGET),
                             context_wall_s=elapsed, diagnostic_residual_s=residual,
                             context_completed=True, peak_rss_bytes=peak_rss_bytes(),
                             instrumented=instrumented, output_artifact=output_name,
                             output_dtype=str(out.dtype), output_shape=list(out.shape),
                             finite=True, nonnegative=True,
                             output_bytes_sha256=hashlib.sha256(out.tobytes(order="C")).hexdigest(),
                             includes_first_call_initialization=call == 0)
                trials.append(trial)
                bill_name = f"call_{call:02d}_bill.json"
                write_json(directory / bill_name, trial)
                artifacts[bill_name] = binding(directory / bill_name)
                print(json.dumps(dict(role=args.role, **trial)), flush=True)
        check_bindings(sources)
        check_bindings(fixtures)
        repeat = predictions[0].tobytes(order="C") == predictions[2].tobytes(order="C")
        seal = dict(schema=SCHEMA, role=args.role, sealed_utc=utc(),
                    source_bindings=sources, fixture_bindings=fixtures,
                    response_sha256=args.response_sha256, protocol_sha256=PINNED[PROTOCOL],
                    trials=trials, artifacts=artifacts, truth_files_accessed=False,
                    repeat_byte_identical=repeat, versions_sha256=canonical_digest(versions))
        seal_sha = write_json(directory / "predictions_and_bills_sealed.json", seal)
        # Archived predictions/features are parity references, not the1e9 truths.
        parity = []
        if args.role == "response":
            for call, (rank, x, delta) in enumerate(diagnostics):
                reference_path = HERE / f"response_rows_closedloop_r{rank}_v1.npz"
                require(digest(reference_path) == PINNED[reference_path.name],
                        "Archived response reference changed")
                with np.load(reference_path, allow_pickle=False) as reference:
                    require(set(reference.files) == {"mean", "x", "delta"},
                            "Unexpected archived reference fields")
                    mean_ref, x_ref, d_ref = reference["mean"], reference["x"], reference["delta"]
                    require(mean_ref.shape == (16, 1024) and x_ref.shape == x.shape and
                            d_ref.shape == delta.shape, "Archived reference shape mismatch")
                    require(all(bool(np.isfinite(v).all()) for v in (mean_ref, x_ref, d_ref)),
                            "Nonfinite archived reference")
                    mean_error = predictions[call].astype(np.float64) - mean_ref.astype(np.float64)
                    x_error = x.astype(np.float64) - x_ref.astype(np.float64)
                    d_error = delta.astype(np.float64) - d_ref.astype(np.float64)
                    item = dict(rank=rank, output_max_abs=float(np.max(np.abs(mean_error))),
                                output_rms=float(np.sqrt(np.mean(mean_error**2))),
                                delta_max_abs=float(np.max(np.abs(d_error))),
                                feature_max_abs=float(np.max(np.abs(x_error))),
                                layer_output_max_abs=np.max(np.abs(mean_error), axis=1).tolist(),
                                layer_delta_max_abs=np.max(np.abs(d_error), axis=1).tolist(),
                                layer_feature_max_abs=np.max(np.abs(x_error), axis=(1, 2)).tolist(),
                                reference_sha256=PINNED[reference_path.name])
                    item["passed"] = item["output_max_abs"] < 1e-5 and item["delta_max_abs"] < 1e-6
                    parity.append(item)
        check_bindings(sources)
        check_bindings(fixtures)
        resource_passed = all(t["peak_rss_bytes"] < RSS_LIMIT and
                              t["context_wall_s"] < 120 for t in trials)
        passed = repeat and resource_passed
        if args.role == "response":
            passed = passed and len(parity) == 2 and all(p["passed"] for p in parity)
        result.update(completed=True, qualification_passed=bool(passed), trials=trials,
                      response_parity=parity, repeat_byte_identical=repeat,
                      predictions_and_bills_seal_sha256=seal_sha,
                      hashes_unchanged=True, native_resource_gate=resource_passed,
                      production_residual_validated=False,
                      resource_qualification="Native diagnostics only; persistent numpy-free release gate still required.",
                      completed_utc=utc())
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), trials=trials, completed_utc=utc())
    result_sha = write_json(directory / "result.json", result)
    write_json(directory / "completion.json", dict(
        schema=SCHEMA, role=args.role, completed_utc=utc(),
        qualification_passed=result["qualification_passed"], result_sha256=result_sha,
        response_sha256=args.response_sha256, protocol_sha256=PINNED[PROTOCOL],
        truth_files_accessed=False))
    print(json.dumps(dict(role=args.role, qualification_passed=result["qualification_passed"],
                          result_sha256=result_sha)), flush=True)
    return 0 if result["qualification_passed"] else 1


def verify_role(root, role, response_sha):
    directory = root / role
    complete = load_json(directory / "completion.json")
    require(complete["schema"] == SCHEMA and complete["role"] == role,
            "Wrong role completion schema")
    require(complete["qualification_passed"] and not complete["truth_files_accessed"],
            "An acquisition did not qualify")
    require(digest(directory / "result.json") == complete["result_sha256"],
            "Role result seal mismatch")
    result = load_json(directory / "result.json")
    require(result["completed"] and result["qualification_passed"] and result["hashes_unchanged"],
            "Incomplete acquisition")
    require(result["schema"] == SCHEMA and result["role"] == role and
            not result["truth_files_accessed"], "Wrong acquisition identity/ownership")
    require(result["response_sha256"] == complete["response_sha256"] == response_sha and
            result["protocol_sha256"] == complete["protocol_sha256"] == PINNED[PROTOCOL],
            "Response/protocol pin mismatch")
    current_sources = source_bindings(response_sha)
    require({k: v["sha256"] for k, v in result["source_bindings"].items()} ==
            {k: v["sha256"] for k, v in current_sources.items()}, "Source set changed")
    check_bindings(result["source_bindings"])
    check_bindings(result["fixture_bindings"])
    started = load_json(directory / "started.json")
    require(digest(directory / "started.json") == result["started_sha256"], "Started seal changed")
    require(started["source_bindings"] == result["source_bindings"] and
            started["fixture_bindings"] == result["fixture_bindings"] and
            started["pid"] == result["pid"], "Acquisition provenance mismatch")
    require(result["versions"]["flopscope"] == "0.12.0" and
            result["versions"]["whestbench"] == "0.16.0" and
            result["versions_sha256"] == canonical_digest(result["versions"]),
            "Runtime version mismatch")
    seal_path = directory / "predictions_and_bills_sealed.json"
    require(digest(seal_path) == result["predictions_and_bills_seal_sha256"],
            "Prediction/bill seal mismatch")
    seal = load_json(seal_path)
    require(seal["trials"] == result["trials"] and
            seal["source_bindings"] == result["source_bindings"] and
            seal["fixture_bindings"] == result["fixture_bindings"] and
            seal["repeat_byte_identical"] and result["repeat_byte_identical"] and
            not seal["truth_files_accessed"], "Prediction seal ownership mismatch")
    require([t["rank"] for t in result["trials"]] == list(RANKS) and
            [t["call"] for t in result["trials"]] == [0, 1, 2], "Wrong fixed sequence")
    for trial in result["trials"]:
        require(load_json(directory / f"call_{trial['call']:02d}_bill.json") == trial,
                "Per-call bill differs from seal")
        require(trial["context_completed"] and 0 < trial["flops"] <= BUDGET and
                trial["peak_rss_bytes"] < RSS_LIMIT and trial["context_wall_s"] < 120 and
                trial["finite"] and trial["output_dtype"] == "float32" and
                trial["output_shape"] == [16, 1024], "Invalid native trial")
        require(trial["instrumented"] == (role == "response" and trial["call"] < 2),
                "Wrong diagnostic schedule")
    if role == "response":
        require([p["rank"] for p in result["response_parity"]] == [35, 100] and
                all(p["passed"] and p["output_max_abs"] < 1e-5 and
                    p["delta_max_abs"] < 1e-6 for p in result["response_parity"]),
                "Response parity did not pass")
    else:
        require(result["response_parity"] == [], "Bank must have no diagnostic wrapper")
    for name, item in seal["artifacts"].items():
        require(Path(item["path"]).resolve() == (directory / name).resolve(),
                "Output artifact escaped role directory")
    check_bindings(seal["artifacts"])
    return result, seal


def score(args, directory):
    # No fixture/candidate/runtime import or estimator call is permitted here.
    roles = {}
    for role in ("bank", "response"):
        roles[role] = verify_role(args.output_root, role, args.response_sha256)
    bank, response = roles["bank"][0], roles["response"][0]
    require(bank["pid"] != response["pid"], "Acquire roles in separate fresh processes")
    require(bank["source_bindings"] == response["source_bindings"] and
            bank["fixture_bindings"] == response["fixture_bindings"] and
            bank["versions"] == response["versions"], "Roles do not share the exact frozen frame")
    import numpy as np
    require(importlib.metadata.version("numpy") == bank["versions"]["numpy"],
            "Scoring NumPy differs from acquisition")
    outputs = {}
    for role, (acquired, seal) in roles.items():
        arrays = []
        for trial in acquired["trials"]:
            path = args.output_root / role / trial["output_artifact"]
            require(digest(path) == seal["artifacts"][path.name]["sha256"], "Output digest mismatch")
            array = np.load(path, allow_pickle=False)
            assert_output(np, array)
            require(hashlib.sha256(array.tobytes(order="C")).hexdigest() ==
                    trial["output_bytes_sha256"], "Served bytes mismatch")
            arrays.append(array)
        require(arrays[0].tobytes(order="C") == arrays[2].tobytes(order="C"),
                "Warm repeat differs from first served output")
        outputs[role] = arrays
    # Both complete roles and all outputs/bills are authenticated BEFORE any
    # truth bytes are hashed/opened. No target-conditioned route can be chosen.
    inputs_seal = dict(schema=SCHEMA, sealed_utc=utc(),
                       checker_sha256=digest(__file__), response_sha256=args.response_sha256,
                       protocol_sha256=PINNED[PROTOCOL], truth_files_accessed=False,
                       role_completion_bindings={role: binding(args.output_root / role / "completion.json")
                                                 for role in roles},
                       role_result_bindings={role: binding(args.output_root / role / "result.json")
                                             for role in roles},
                       role_prediction_seal_bindings={role: binding(args.output_root / role / "predictions_and_bills_sealed.json")
                                                      for role in roles})
    input_sha = write_json(directory / "both_roles_verified_before_truth.json", inputs_seal)
    truth_started = utc()
    write_json(directory / "truth_access_started.json", dict(utc=truth_started,
               both_roles_seal_sha256=input_sha, ranks=[35, 100]))
    truth_bindings, truths = {}, {}
    for rank in (35, 100):
        path = TRUTH_DIR / f"truthpost_rank{rank:04d}.npz"
        truth_bindings[str(rank)] = binding(path)
        with np.load(path, allow_pickle=False) as data:
            truth = data["mc_post"].astype(np.float64)
        require(truth.shape == (16, 1024) and bool(np.isfinite(truth).all()),
                "Invalid fixed1e9 truth")
        truths[rank] = truth
    scored = {}
    for role, (acquired, _) in roles.items():
        per_call = []
        for trial, out in zip(acquired["trials"], outputs[role], strict=True):
            layer_mse = np.mean((out.astype(np.float64) - truths[trial["rank"]])**2, axis=1)
            raw = float(layer_mse[-1])
            multiplier = max(0.1, trial["flops"] / BUDGET)
            per_call.append(dict(call=trial["call"], rank=trial["rank"],
                                 layer_mse=layer_mse.tolist(), final_raw_mse=raw,
                                 flops=trial["flops"], utilization=trial["flops"] / BUDGET,
                                 multiplier=multiplier, adjusted=raw * multiplier,
                                 output_sha256=trial["output_bytes_sha256"],
                                 included_in_two_net_mean=trial["call"] < 2))
        scored[role] = dict(candidate=BANK if role == "bank" else RESPONSE,
                            candidate_sha256=acquired["source_bindings"][BANK if role == "bank" else RESPONSE]["sha256"],
                            per_call=per_call,
                            mean_raw_mse=sum(t["final_raw_mse"] for t in per_call[:2]) / 2,
                            mean_flops=sum(t["flops"] for t in per_call[:2]) / 2,
                            mean_adjusted=sum(t["adjusted"] for t in per_call[:2]) / 2,
                            warm_rank35=per_call[2])
    require(scored["bank"]["mean_adjusted"] > 0 and scored["bank"]["mean_raw_mse"] > 0,
            "Bank comparison denominator is zero")
    ratio = scored["response"]["mean_adjusted"] / scored["bank"]["mean_adjusted"]
    for role, (acquired, seal) in roles.items():
        check_bindings(acquired["source_bindings"])
        check_bindings(acquired["fixture_bindings"])
        check_bindings(seal["artifacts"])
    check_bindings(truth_bindings)
    for key in ("role_completion_bindings", "role_result_bindings", "role_prediction_seal_bindings"):
        check_bindings(inputs_seal[key])
    result = dict(schema=SCHEMA, mode="score", completed_utc=utc(),
                  qualification_passed=True, hashes_unchanged=True,
                  both_roles_verified_before_truth_sha256=input_sha,
                  truth_access_started_utc=truth_started, truth_bindings=truth_bindings,
                  response_sha256=args.response_sha256, protocol_sha256=PINNED[PROTOCOL],
                  roles=scored, adjusted_response_over_bank=ratio,
                  raw_response_over_bank=scored["response"]["mean_raw_mse"] / scored["bank"]["mean_raw_mse"],
                  advances_to_independent_14_panel=ratio < 1.0,
                  scientific_status="advance_to_panel" if ratio < 1.0 else "fixed_exception_closed",
                  reference="Existing1e9 truthpost mc_post; fixed development ranks35/100",
                  aggregation="Mean of two per-net raw*max(.1,F/2^41); call0 initialization included, call2 excluded",
                  official_score=False, milestone_achieved=False, goal_achieved=False,
                  production_residual_validated=False,
                  qualification="Two development nets only; independent14 panel and exact-source persistent release checks remain mandatory.")
    result_sha = write_json(directory / "result.json", result)
    write_json(directory / "completion.json", dict(schema=SCHEMA, completed_utc=utc(),
               result_sha256=result_sha, qualification_passed=True,
               advances_to_independent_14_panel=ratio < 1.0))
    print(json.dumps(dict(adjusted_response_over_bank=ratio,
                          advances_to_independent_14_panel=ratio < 1.0,
                          result_sha256=result_sha)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--role", choices=("bank", "response"))
    mode.add_argument("--score", action="store_true")
    parser.add_argument("--response-sha256", help="Required frozen derived-candidate SHA, identical in all three processes")
    parser.add_argument("--output-root", type=Path,
                        default=HERE / "response_rows_adjusted_reevaluation_v1")
    args = parser.parse_args()
    if args.role is None and not args.score:
        parser.print_help()
        return 0
    require(args.response_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", args.response_sha256),
            "Supply the reviewed frozen --response-sha256")
    args.output_root = (HERE / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    require(args.output_root.parent == HERE and args.output_root.name not in ("bank", "response", "score"),
            "Output root must be a new named direct child of this versioned package")
    args.output_root.mkdir(exist_ok=True)
    directory = args.output_root / ("score" if args.score else args.role)
    directory.mkdir(exist_ok=False)
    try:
        return score(args, directory) if args.score else acquire(args, directory)
    except Exception as error:
        # Preserve an exclusive failure receipt; never repair/reuse a failed run.
        write_json(directory / "preflight_or_score_failure.json", dict(
            schema=SCHEMA, utc=utc(), mode="score" if args.score else args.role,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            qualification_passed=False))
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
