"""One inert-by-default, target-blind current-parent final-state capture."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "terminal_summary_capture_v1"
PARENT = "candidate_response_pair_polynomial_f32_v1.py"
HELPER = "check_response_rows_adjusted_v1.py"
PROTOCOL = "TERMINAL_SUMMARY_CAPTURE_PROTOCOL_V1.md"
REFERENCE = "pair_polynomial_f32_fixed_dev_v1/candidate/result.json"
PINS = {
    PARENT: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    REFERENCE: "11a66c5a85cabde9c23bca95dba81e0585d0ede999db09ac26c0d538a988a08a",
}
BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute():
    if hashlib.sha256((HERE / HELPER).read_bytes()).hexdigest() != PINS[HELPER]:
        raise RuntimeError("Frozen inert helper changed")
    h = import_path("terminal_capture_helpers", HERE / HELPER)
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable, "OFFPY only")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
              and all(os.environ.get(k) == "1" for k in BLAS), "Launch controls required")
    source_set = {name: h.binding(HERE / name) for name in
                  (*PINS, PROTOCOL, Path(__file__).name)}
    h.require(all(source_set[k]["sha256"] == v for k, v in PINS.items()), "Source mismatch")
    fixtures, rows = h.fixture_bindings()
    reference = h.load_json(HERE / REFERENCE)
    h.require(reference["completed"] and reference["qualification_passed"]
              and reference["candidate_sha256"] == PINS[PARENT]
              and reference["fixture_bindings"] == fixtures
              and reference["trials"][0]["rank"] == 35, "Wrong reference frame")
    expected = reference["trials"][0]
    output_name = expected["output_artifact"]
    reference_output = reference["artifacts"][output_name]
    h.check_bindings({"reference_output": reference_output})
    versions = {k: importlib.metadata.version(k) for k in
                ("numpy", "flopscope", "whestbench", "threadpoolctl")}
    h.require(versions == reference["versions"] and versions["flopscope"] == "0.12.0"
              and versions["whestbench"] == "0.16.0", "Wrong versions")
    OUT.mkdir(exist_ok=False)
    started = dict(utc=h.utc(), pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                   executable=sys.executable, source_bindings=source_set,
                   fixture_bindings=fixtures, versions=versions,
                   blas_env={k: os.environ[k] for k in BLAS}, rank=35,
                   truth_files_accessed=False, one_call=True, warmup=False)
    start_sha = h.write_json(OUT / "started.json", started)
    result = dict(schema="terminal_summary_capture_v1", started_sha256=start_sha,
                  completed=False, qualification_passed=False, truth_files_accessed=False,
                  source_bindings=source_set, fixture_bindings=fixtures, versions=versions,
                  final_score_evidence=False, new_estimator_executed=False)
    arrays, kernels, artifacts = {}, [], {}
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = import_path("terminal_capture_parent", HERE / PARENT)
        raw_source = (HERE / PARENT).read_text()
        nodes = [node for node in ast.parse(raw_source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_predict_binary"]
        source = ast.get_source_segment(raw_source, nodes[-1])
        anchor = "        pair_map = _nonlinear_first_gaussian if l == 0 else nonlinear\n"
        insertion = ("        if l == depth - 1:\n"
                     "            _terminal_capture_input(l, w32, mu, Cpre, Bpre, bpre, pp, lam, "
                     "star_pre, cp_pre, K21, k4, k22, live_indices, alive, response_indices, "
                     "response_mask, omitted)\n")
        h.require(source.count(anchor) == 1, "Ambiguous source anchor")
        instrumented = source.replace(anchor, insertion + anchor)
        h.require(instrumented.replace(insertion, "") == source, "Source inverse failed")
        state = dict(layer=-1, final_captured=False)
        ctx_ref = [None]

        def capture_input(layer, *values):
            h.require(layer == 15 and not state["final_captured"], "Unexpected capture")
            names = ("w", "mu", "pre_C", "pre_B", "bpre", "pp", "lam", "star_pre",
                     "cp_pre", "K21", "k4", "k22", "alive_indices", "alive",
                     "response_indices", "response_mask", "omitted")
            arrays.update(zip(names, values, strict=True))
            state["final_captured"] = True

        original_predict = module._predict_binary
        original_push = module._push_bank_shared_newborn
        original_mm = module._mm
        original_map = module.nonlinear
        original_head = module.gnn_step

        def observe_push(*args, **kwargs):
            state["layer"] += 1
            return original_push(*args, **kwargs)

        def observe_mm(left, right, *args, **kwargs):
            before = int(ctx_ref[0].flops_used)
            value = original_mm(left, right, *args, **kwargs)
            if state["layer"] == 15:
                kernels.append(dict(left_shape=list(left.shape), right_shape=list(right.shape),
                                    flops=int(ctx_ref[0].flops_used)-before,
                                    after_input_capture=state["final_captured"]))
            return value

        def observe_map(*args, **kwargs):
            value = original_map(*args, **kwargs)
            if state["final_captured"]:
                arrays["host"] = value[0]
                arrays["post_C"] = value[1]
                arrays.update({f"aux_{key}": val for key, val in value[4].items()})
            return value

        def observe_head(layer, x, sigma, w, previous, payload):
            if layer == 15:
                arrays["features"] = x
                arrays["state_previous"] = previous
            value = original_head(layer, x, sigma, w, previous, payload)
            if layer == 15:
                arrays["delta"] = value[0]
            return value

        module._terminal_capture_input = capture_input
        exec(compile(instrumented, "<terminal-observation-only>", "exec"), module.__dict__)
        module._push_bank_shared_newborn = observe_push
        module._mm = observe_mm
        module.nonlinear = observe_map
        module.gnn_step = observe_head
        row = next(item for item in rows if item["rank"] == 35)
        weights = np.load(row["weights_path"], mmap_mode="r", allow_pickle=False)
        h.require(weights.shape == (16, 1024, 1024) and np.isfinite(weights).all(), "Invalid weights")
        mlp = MLP(width=1024, depth=16, weights=list(weights),
                  seed=int(derive_estimator_seed(row["mlp_seed"])), name="terminal-summary-capture-35")
        try:
            with threadpool_limits(limits=1):
                pools = threadpool_info()
                h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                          "BLAS thread mismatch")
                result["threadpools"] = pools
                begin = h.utc()
                t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    ctx_ref[0] = ctx
                    served = module.Estimator().predict(mlp, h.BUDGET)
                    returned = materialise_predictions(served, depth=16, width=1024)
                elapsed = time.perf_counter()-t0
        finally:
            module._predict_binary = original_predict
            module._push_bank_shared_newborn = original_push
            module._mm = original_mm
            module.nonlinear = original_map
            module.gnn_step = original_head
            del module._terminal_capture_input
        out = np.asarray(returned)
        h.assert_output(np, out)
        output_hash = hashlib.sha256(out.tobytes()).hexdigest()
        h.require(output_hash == expected["output_bytes_sha256"]
                  and int(ctx.flops_used) == expected["flops"], "Exact output/bill parity failed")
        h.require(state["layer"] == 15 and state["final_captured"] and kernels,
                  "Incomplete final capture")
        bill = dict(flops=int(ctx.flops_used), wall_s=elapsed, meter_wall_s=float(ctx.wall_time_s),
                    backend_s=float(ctx.flopscope_backend_time_s),
                    overhead_s=float(ctx.flopscope_overhead_time_s),
                    diagnostic_residual_s=float(ctx.residual_wall_time_s),
                    peak_rss_bytes=h.peak_rss_bytes(), began_utc=begin, ended_utc=h.utc(),
                    first_context_initialization_included=True, output_bytes_sha256=output_hash,
                    archive_output_and_complete_bill_exact=True, production_runtime_evidence=False)
        h.require(0 < bill["flops"] <= h.BUDGET and 0 < elapsed < 120
                  and all(math.isfinite(bill[k]) and bill[k] >= 0 for k in
                          ("meter_wall_s", "backend_s", "overhead_s", "diagnostic_residual_s"))
                  and h.peak_rss_bytes() < h.RSS_LIMIT, "Native resource failure")
        h.write_json(OUT / "bill.json", bill)
        artifacts["bill.json"] = h.binding(OUT / "bill.json")
        artifacts["output.npy"] = h.save_npy(np, OUT / "output.npy", out)
        shapes = {}
        for name, value in arrays.items():
            a = np.asarray(value)
            h.require(np.isfinite(a).all(), f"Nonfinite capture {name}")
            shapes[name] = dict(shape=list(a.shape), dtype=str(a.dtype), bytes=int(a.nbytes))
            artifacts[f"{name}.npy"] = h.save_npy(np, OUT / f"{name}.npy", a)
            h.require(h.peak_rss_bytes() < h.RSS_LIMIT, "Capture save RSS failure")
        geometry = dict(n=1024, R=int(arrays["pp"].shape[1]),
                        response_rows=int(arrays["response_indices"].shape[0]),
                        alive_columns=int(arrays["alive_indices"].shape[0]))
        h.check_bindings(source_set)
        h.check_bindings(fixtures)
        h.check_bindings(artifacts)
        result.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                      source_inverse_exact=True, actual_parent_call=True,
                      archive_output_and_complete_bill_exact=True, geometry=geometry,
                      bill=bill, final_kernels=kernels, array_shapes=shapes,
                      peak_rss_bytes=h.peak_rss_bytes())
    except BaseException:
        result["error"] = traceback.format_exc()
    result.update(artifacts=artifacts, ended_utc=h.utc())
    seal = h.write_json(OUT / "result.json", result)
    h.write_json(OUT / "completion.json", dict(qualification_passed=result["qualification_passed"],
                 result_sha256=seal, truth_files_accessed=False, utc=h.utc()))
    print(h.canonical_digest(result), flush=True)
    print({key: result.get(key) for key in
           ("qualification_passed", "geometry", "bill", "final_kernels", "error")}, flush=True)
    return 0 if result["qualification_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
