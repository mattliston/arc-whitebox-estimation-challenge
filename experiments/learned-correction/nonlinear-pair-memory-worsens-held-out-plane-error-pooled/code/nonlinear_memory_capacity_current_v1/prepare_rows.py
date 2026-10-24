"""Inert-by-default current-frame preparation; no estimator, head, MC or fit.

Only --execute opens captured arrays. Predictor fields are completed and
sealed before current-layer K21 labels are decoded. All outputs are new files.
The arithmetic diagnostics are not a native bill or a score/accuracy gate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


SELF = Path(__file__).resolve()
HERE = SELF.parent
NEW = HERE.parent
GATE_ENV = NEW.parent / "data_gate/.venv"
CAPTURE = HERE / "capture_current.py"
CAPTURE_PROTOCOL = HERE / "CAPTURE_PROTOCOL.md"
CANDIDATE = NEW / "candidate_response_pair_polynomial_f32_v1.py"
HELPER = NEW / "check_response_rows_adjusted_v1.py"
REFERENCE = NEW / "pair_polynomial_f32_panel_v1/candidate/result.json"
FIRST_RESULT = HERE / "rows/row_0035/result.json"
FIRST_RESULT_SHA = "ac2a177262d7175c8c105fe798b1db7a35a5bcebc9b6cc68aaaa651800b39463"
PINS = {
    CAPTURE: "3950d1b43e62b660093bd1140d5e9662b3e6f64a979b66807c1d9aed32fb918d",
    CAPTURE_PROTOCOL: "d610cf7dc1000ca13c9fbb586e82e3f7fb671ea3aaca3dca427c613e105e4947",
    CANDIDATE: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    REFERENCE: "72b41dbc14cd9b98f823b8f3b0aa9116839d5181264a8b597c8f3f47b835e595",
}
RANKS = (35, 100, 150, 200, 250, 300, 400, 500)
N, LAYERS = 1024, 15
UNIT_FLOOR = 1e-10
DECODER_AMPLITUDE = .05
RSS_LIMIT, WALL_LIMIT = 4 * 1024**3, 300
INIT_FLOPS = 1383913666
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PREDICTOR_KEYS = ("V", "rho", "Kstar", "alpha", "gate", "pdf")
OUTPUT_KEYS = PREDICTOR_KEYS + ("target_loss_only",)
SCHEMA = "nonlinear_memory_current_prepared_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return sha(path)


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed during hashing: {path.name}")
    return dict(path=str(path), sha256=digest, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def authenticate(item, expected_path):
    path = Path(expected_path).resolve()
    require(Path(item["path"]).resolve() == path, f"Wrong bound path: {path.name}")
    actual = binding(path)
    require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
            f"Frozen file changed: {path.name}")
    return actual


def check_bindings(items):
    for item in items.values():
        authenticate(item, item["path"])


def peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def authenticate_capture(rank):
    """Read JSON and hash bytes only; never decode any array here."""
    sources = {str(path): binding(path) for path in (*PINS, SELF)}
    require(all(sources[str(path)]["sha256"] == digest for path, digest in PINS.items()),
            "Frozen source/protocol/reference mismatch")
    first_completion = read_json(FIRST_RESULT.with_name("completion.json"))
    require(first_completion["passed"] and
            first_completion["result_sha256"] == sha(FIRST_RESULT) == FIRST_RESULT_SHA,
            "Rank35 capture completion is not the frozen passing receipt")
    directory = HERE / "rows" / f"row_{rank:04d}"
    paths = {name: directory / name for name in ("started.json", "result.json", "completion.json")}
    inputs = {f"capture/{name}": binding(path) for name, path in paths.items()}
    completion = read_json(paths["completion.json"])
    require(completion["passed"] and completion["result_sha256"] == inputs["capture/result.json"]["sha256"],
            "Capture completion/result seal mismatch")
    row = read_json(paths["result.json"])
    started = read_json(paths["started.json"])
    require(row["passed"] and row["completed"] and row["rank"] == rank and
            row["target_access"] is False and row["deterministic_teacher_labels"] and
            row["final_score_evidence"] is False and row["source_inverse_exact"] and
            row["archive_output_and_bill_exact"] and row["same_source_control_exact"],
            "Capture is not fully source/numerically qualified")
    require(row["capture_sha256"] == PINS[CAPTURE] and
            row["protocol_sha256"] == PINS[CAPTURE_PROTOCOL], "Wrong capture law")
    for key in ("rank", "capture_sha256", "protocol_sha256", "source_bindings", "fixture_bindings", "versions"):
        require(started[key] == row[key], f"Started/final capture ownership differs: {key}")
    require(set(row["source_bindings"]) == {str(path) for path in PINS},
            "Unexpected capture source set")
    for path in PINS:
        item = row["source_bindings"][str(path)]
        require(item["sha256"] == PINS[path], "Capture source pin mismatch")
        authenticate(item, path)
    require(row["peak_rss_bytes"] < 8 * 1024**3 and row["elapsed_s"] < 900,
            "Capture resource qualification failed")
    inventory = row["capture_inventory"]
    require(all(inventory[key] == list(range(15)) for key in ("pre", "post", "aux")) and
            inventory["heads"] == list(range(1, 15)) and inventory["layer"] == 15,
            "Incomplete capture chronology")
    require([item["layer"] for item in row["post_checks"]] == list(range(14)) and
            all(item["max_scaled_direct_error"] <= 1e-12 for item in row["post_checks"]),
            "Unqualified physical post-plane reconstruction")
    reference = read_json(REFERENCE)
    require(reference["candidate_sha256"] == PINS[CANDIDATE] and
            reference["completed"] and reference["qualification_passed"], "Wrong native reference")
    trial = next(item for item in reference["trials"] if item["rank"] == rank)
    warm = trial["flops"] - (INIT_FLOPS if trial["includes_first_call_initialization"] else 0)
    require([item["role"] for item in row["bills"]] == ["instrumented_cold", "uninstrumented_warm"] and
            [item["flops"] for item in row["bills"]] == [warm + INIT_FLOPS, warm] and
            all(item["output_bytes_sha256"] == trial["output_bytes_sha256"] for item in row["bills"]),
            "Capture/native output or initialization frame mismatch")
    fixtures = row["fixture_bindings"]
    require(set(fixtures) == {"manifest", f"row_{rank}", f"weights_{rank}", "reference_output"},
            "Unexpected fixture set")
    for key in ("manifest", f"row_{rank}", f"weights_{rank}"):
        require(fixtures[key] == reference["fixture_bindings"][key], "Wrong canonical fixture")
        inputs[f"fixture/{key}"] = authenticate(fixtures[key], fixtures[key]["path"])
    require(fixtures["reference_output"] == reference["artifacts"][trial["output_artifact"]],
            "Reference output manifest mismatch")
    names = ({f"pre_{l:02d}.npz" for l in range(15)} |
             {f"aux_{l:02d}.npz" for l in range(15)} |
             {f"post_{l:02d}.npz" for l in range(15)} |
             {f"head_{l:02d}.npz" for l in range(1, 15)} | {"served.npy"})
    require(set(row["artifacts"]) == names, "Capture artifact manifest is incomplete")
    for name, item in row["artifacts"].items():
        require(Path(item["path"]).resolve() == (directory / name).resolve(),
                "Capture artifact escaped its row directory")
    consumed = ({f"pre_{l:02d}.npz" for l in range(15)} |
                {f"aux_{l:02d}.npz" for l in range(15)} |
                {f"post_{l:02d}.npz" for l in range(14)})
    for name in sorted(consumed):
        inputs[f"capture/{name}"] = authenticate(row["artifacts"][name], directory / name)
    # Head, served-output and unused post14 files are not opened, even for hashing.
    return row, sources, inputs, directory


def execute(rank):
    timer = time.perf_counter()
    require(Path(sys.prefix).resolve() == GATE_ENV.resolve(), "Designated OFFPY only")
    require("anaconda3" not in sys.executable and
            not any(name == "torch" or name.startswith("torch.") for name in sys.modules),
            "No Torch interpreter/import is allowed")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and
            all(os.environ.get(name) == "1" for name in BLAS_ENV), "Explicit one-thread launch required")
    output_dir = HERE / "prepared"
    output_dir.mkdir(exist_ok=True)
    base = output_dir / f"row_{rank:04d}"
    paths = {key: Path(str(base) + suffix) for key, suffix in {
        "started": ".started.json", "causal": ".causal.json", "array": ".npz",
        "result": ".json", "completion": ".completion.json"}.items()}
    require(not any(path.exists() for path in paths.values()), "Refuse overwrite or partial-row retry")
    result = dict(schema=SCHEMA, rank=rank, passed=False, completed=False,
                  mc_truth_files_accessed=False, head_files_accessed=False,
                  current_teacher_labels_decoded=False, final_score_evidence=False)
    sources, inputs = {}, {}

    def resources():
        require(peak_rss_bytes() < RSS_LIMIT, "Prepared-row peak RSS reached4GiB")
        require(time.perf_counter() - timer < WALL_LIMIT, "Prepared-row300s limit exhausted")

    def watchdog(signum, frame):
        raise TimeoutError("Fixed300s prepared-row watchdog exhausted")

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(WALL_LIMIT)
    try:
        capture, sources, inputs, capture_dir = authenticate_capture(rank)
        source_sha = sources[str(SELF)]["sha256"]
        if rank != 35:
            first_path = output_dir / "row_0035.json"
            first_complete_path = output_dir / "row_0035.completion.json"
            first_complete = read_json(first_complete_path)
            require(first_complete["passed"] and first_complete["result_sha256"] == sha(first_path),
                    "Rank35 preparation is not sealed")
            first = read_json(first_path)
            require(first["passed"] and first["completed"] and first["prepare_sha256"] == source_sha and
                    first["capture_sha256"] == PINS[CAPTURE], "First-row preparation source changed")
            inputs["first_preparation/result"] = binding(first_path)
            inputs["first_preparation/completion"] = binding(first_complete_path)
        versions = {name: importlib.metadata.version(name) for name in capture["versions"]}
        require(versions == capture["versions"], "Native preparation environment changed")
        result.update(prepare_sha256=source_sha, capture_sha256=PINS[CAPTURE],
                      source_bindings=sources, input_bindings=inputs, versions=versions,
                      capture_result_sha256=inputs["capture/result.json"]["sha256"],
                      capture_artifact_manifest_authenticated=True,
                      consumed_artifacts_byte_rehashed=True,
                      unconsumed_artifacts_not_opened=["head_*.npz", "served.npy", "post_14.npz"],
                      predictor_keys=list(PREDICTOR_KEYS), target_key="target_loss_only")
        write_json(paths["started"], dict(**result, utc=utc(), pid=os.getpid(),
                   nice=os.getpriority(os.PRIO_PROCESS, 0), rss_limit_bytes=RSS_LIMIT,
                   watchdog_s=WALL_LIMIT, unit_floor=UNIT_FLOOR,
                   decoder_amplitude_diagnostic=DECODER_AMPLITUDE))
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits

        def array(value, shape, dtype, name):
            require(value.shape == shape and value.dtype == np.dtype(dtype), f"Wrong shape/dtype: {name}")
            require(bool(np.isfinite(value).all()), f"Nonfinite input: {name}")
            return value

        def array_record(value):
            require(value.flags.c_contiguous, "Noncontiguous prepared array")
            return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                        sha256=hashlib.sha256(memoryview(value)).hexdigest())

        weights = np.load(inputs[f"fixture/weights_{rank}"]["path"], mmap_mode="r", allow_pickle=False)
        array(weights, (16, N, N), np.float32, "canonical W32")
        data = {key: np.zeros((LAYERS, N, N), dtype=np.float32) for key in ("V", "rho", "Kstar")}
        data.update({key: np.zeros((LAYERS, N), dtype=np.float32) for key in ("alpha", "gate", "pdf")})
        sigma64 = np.empty((LAYERS, N), dtype=np.float64)
        units, star_comparisons = [], []
        indices = np.diag_indices(N)
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            require(all(pool["num_threads"] == 1 for pool in pools if pool.get("user_api") == "blas"),
                    "Reported BLAS pool is not single-threaded")
            result["threadpools"] = pools
            result["explicit_blas_env"] = {name: os.environ[name] for name in BLAS_ENV}
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                for l in range(LAYERS):
                    with np.load(capture_dir / f"pre_{l:02d}.npz", allow_pickle=False) as pre:
                        current_C = array(pre["C"], (N, N), np.float64, f"pre{l} C")
                        mu = array(pre["mu"], (N,), np.float64, f"pre{l} mu")
                    with np.load(capture_dir / f"aux_{l:02d}.npz", allow_pickle=False) as aux:
                        sigma = array(aux["sigma"], (N,), np.float64, f"pre{l} native sigma")
                        require(bool((sigma > 0).all()), "Nonpositive native sigma")
                        sigma64[l] = sigma
                        for key in ("alpha", "gate", "pdf"):
                            data[key][l] = array(aux[key], (N,), np.float64, f"pre{l} {key}").astype(np.float32)
                    rho64 = current_C / (sigma[:, None] * sigma[None, :])
                    data["rho"][l] = rho64.astype(np.float32)
                    diag = np.diag(current_C)
                    unit = dict(layer=l, native_sigma_min=float(sigma.min()), native_sigma_max=float(sigma.max()),
                                pre_variance_min=float(diag.min()), negative_pre_variance_count=int((diag < 0).sum()),
                                native_sigma_squared_vs_floored_pre_variance_max_abs=float(np.max(np.abs(sigma*sigma-np.maximum(diag, UNIT_FLOOR)))),
                                alpha32_vs_mu_over_native_sigma_max_abs=float(np.max(np.abs(data["alpha"][l].astype(np.float64)-mu/sigma))),
                                rho_abs_max=float(np.max(np.abs(rho64))), rho_abs_gt_one_count=int((np.abs(rho64) > 1).sum()))
                    if l:
                        with np.load(capture_dir / f"post_{l-1:02d}.npz", allow_pickle=False) as post:
                            variance = array(post["variance"], (N,), np.float64, f"post{l-1} variance")
                            physical = array(post["physical_K21"], (N, N), np.float64, f"post{l-1} physical K21")
                        old_scale = np.sqrt(np.maximum(variance, UNIT_FLOOR))
                        denominator = (old_scale * old_scale)[:, None] * old_scale[None, :]
                        d64 = physical / denominator
                        v64 = (old_scale[:, None] * weights[l].astype(np.float64)) / sigma[None, :]
                        v32, d32 = v64.astype(np.float32), d64.astype(np.float32)
                        e32 = d32.copy()
                        e32[indices] -= np.float32(2/3) * np.diag(d32)
                        q32 = e32 @ v32
                        k32 = (v32*v32).T @ q32 + np.float32(2) * ((v32*q32).T @ v32)
                        data["V"][l], data["Kstar"][l] = v32, k32
                        unit.update(previous_post_variance_min=float(variance.min()),
                                    previous_post_negative_variance_count=int((variance < 0).sum()),
                                    previous_post_below_units_floor_count=int((variance < UNIT_FLOOR).sum()),
                                    post_D_abs_max=float(np.max(np.abs(d64))), V_abs_max=float(np.max(np.abs(v64))),
                                    physical_units_roundtrip_relative_l2=float(np.linalg.norm(d64*denominator-physical)/max(np.linalg.norm(physical), 1e-12)))
                        if rank == 35:
                            e64 = d64.copy()
                            e64[indices] -= (2/3) * np.diag(d64)
                            q64 = e64 @ v64
                            k64 = (v64*v64).T @ q64 + 2 * ((v64*q64).T @ v64)
                            delta = k32.astype(np.float64) - k64
                            reference_rms = float(np.sqrt(np.mean(k64*k64)))
                            error_rms = float(np.sqrt(np.mean(delta*delta)))
                            star_comparisons.append(dict(layer=l, fp64_reference_rms=reference_rms,
                                 fp32_minus_fp64_rms=error_rms,
                                 relative_rms_with_1e_12_floor=error_rms/max(reference_rms, 1e-12),
                                 max_abs=float(np.max(np.abs(delta)))))
                            require(bool(np.isfinite(k64).all()), "Nonfinite independent FP64 star reference")
                            del e64, q64, k64, delta
                        del physical, denominator, d64, d32, e32, q32, k32, v64, v32
                    units.append(unit)
                    for key in PREDICTOR_KEYS:
                        require(bool(np.isfinite(data[key][l]).all()), f"Nonfinite prepared predictor {key}:{l}")
                    resources()
                predictor_seal = {key: array_record(data[key]) for key in PREDICTOR_KEYS}
                causal_sha = write_json(paths["causal"], dict(schema=SCHEMA, rank=rank, utc=utc(),
                    current_teacher_labels_decoded=False, prepare_sha256=source_sha,
                    capture_result_sha256=result["capture_result_sha256"], predictors=predictor_seal))
                # Only now decode the current action label. Its scale was fixed above.
                data["target_loss_only"] = np.zeros((LAYERS, N, N), dtype=np.float32)
                energies = []
                result["current_teacher_labels_decoded"] = True
                for l in range(1, LAYERS):
                    with np.load(capture_dir / f"pre_{l:02d}.npz", allow_pickle=False) as pre:
                        teacher = array(pre["K21_loss_only"], (N, N), np.float64, f"pre{l} loss-only K21")
                    target64 = teacher / ((sigma64[l]*sigma64[l])[:, None] * sigma64[l][None, :])
                    data["target_loss_only"][l] = target64.astype(np.float32)
                    residual = data["target_loss_only"][l].astype(np.float64) - data["Kstar"][l].astype(np.float64)
                    remainder = np.maximum(np.abs(residual)-DECODER_AMPLITUDE, 0)
                    energies.append(dict(layer=l, zero_decoder_mse=float(np.mean(residual*residual)),
                        ideal_pointwise_decoder_range_floor_mse=float(np.mean(remainder*remainder)),
                        target_rounding_max_abs=float(np.max(np.abs(data["target_loss_only"][l].astype(np.float64)-target64)))))
                    require(bool(np.isfinite(data["target_loss_only"][l]).all()), "Nonfinite prepared label")
                    resources()
        require(set(data) == set(OUTPUT_KEYS), "Prepared field contract changed")
        require(all(array_record(data[key]) == predictor_seal[key] for key in PREDICTOR_KEYS),
                "A label changed a prepared predictor")
        require(not np.any(data["V"][0]) and not np.any(data["Kstar"][0]) and
                not np.any(data["target_loss_only"][0]), "Layer0 zero convention changed")
        records = {key: array_record(data[key]) for key in OUTPUT_KEYS}
        with paths["array"].open("xb") as stream:
            np.savez(stream, **data)
        output = binding(paths["array"])
        check_bindings(sources)
        check_bindings(inputs)
        require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules), "Unexpected Torch import")
        resources()
        result.update(passed=True, completed=True, source_hashes_unchanged=True,
                      input_hashes_unchanged=True, output=output, array_contract=records,
                      causal_predictors_seal_sha256=causal_sha, predictor_fields_unchanged_after_labels=True,
                      units=units, first_rank35_fp64_vs_fp32_star=star_comparisons,
                      star_comparison_is_report_only=True, diagnostic_energies=energies,
                      diagnostics_are_not_normalizers_or_tuning=True,
                      target_scale_owner="native pre sigma, fixed before target decode",
                      star32_dense_product_subtotal=14*3*N*N*(2*N-1),
                      reference64_dense_product_subtotal=(14*3*N*N*(2*N-1)*2 if rank == 35 else 0),
                      product_subtotals_exclude_pointwise_casts_checks_io=True)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(elapsed_s=time.perf_counter()-timer, peak_rss_bytes=peak_rss_bytes(), ended_utc=utc())
    result_sha = write_json(paths["result"], result)
    write_json(paths["completion"], dict(schema=SCHEMA, rank=rank, passed=result["passed"],
               result_sha256=result_sha, utc=utc()))
    print(json.dumps({key: result.get(key) for key in ("rank", "passed", "elapsed_s", "peak_rss_bytes", "error")}), flush=True)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rank", type=int, choices=RANKS, default=35)
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    return execute(args.rank)


if __name__ == "__main__":
    raise SystemExit(main())
