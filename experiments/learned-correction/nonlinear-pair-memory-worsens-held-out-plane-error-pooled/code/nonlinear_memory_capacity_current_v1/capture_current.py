"""Inert-by-default observation-only current0b84 teacher capture; Phase2."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT_DIR = HERE.parent
PARENT = PARENT_DIR / "candidate_response_pair_polynomial_f32_v1.py"
HELPER = PARENT_DIR / "check_response_rows_adjusted_v1.py"
REFERENCE = PARENT_DIR / "pair_polynomial_f32_panel_v1/candidate/result.json"
PROTOCOL = HERE / "CAPTURE_PROTOCOL.md"
PINS = {
    PARENT: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    REFERENCE: "72b41dbc14cd9b98f823b8f3b0aa9116839d5181264a8b597c8f3f47b835e595",
}
RANKS = (35, 100, 150, 200, 250, 300, 400, 500)
INIT_FLOPS = 1383913666
BLOCK = 2048
RSS_LIMIT = 8 * 1024**3
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(rank):
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != PINS[HELPER]:
        raise RuntimeError("Frozen helper changed")
    h = load_module("memory_capture_helper", HELPER)
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve(), "OFFPY only")
    h.require("anaconda3" not in sys.executable and "torch" not in sys.modules,
              "Never use the Torch interpreter for capture")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and
              all(os.environ.get(k) == "1" for k in BLAS_ENV), "Launch controls missing")
    h.require(shutil.disk_usage(HERE).free >= 5 * 1024**3, "Disk headroom <5GiB")
    sources = {str(p): h.binding(p) for p in (*PINS, PROTOCOL, Path(__file__))}
    h.require(all(sources[str(p)]["sha256"] == v for p, v in PINS.items()), "Source pin mismatch")
    capture_sha = sources[str(Path(__file__))]["sha256"]
    protocol_sha = sources[str(PROTOCOL)]["sha256"]
    if rank != 35:
        first_path = HERE / "rows/row_0035/result.json"
        first_completion = h.load_json(HERE / "rows/row_0035/completion.json")
        h.require(first_completion["passed"] and first_completion["result_sha256"] == h.digest(first_path),
                  "First-row receipt digest changed")
        first = h.load_json(first_path)
        h.require(first["passed"] and first["capture_sha256"] == capture_sha
                  and first["protocol_sha256"] == protocol_sha, "First row not qualified")
    reference = h.load_json(REFERENCE)
    h.require(reference["qualification_passed"] and reference["completed"]
              and reference["candidate_sha256"] == PINS[PARENT], "Wrong reference")
    expected = next(x for x in reference["trials"] if x["rank"] == rank)
    warm_flops = expected["flops"] - (INIT_FLOPS if expected["includes_first_call_initialization"] else 0)
    fixture = {k: reference["fixture_bindings"][k] for k in
               ("manifest", f"row_{rank}", f"weights_{rank}")}
    fixture["reference_output"] = reference["artifacts"][expected["output_artifact"]]
    h.check_bindings(fixture)
    versions = {k: importlib.metadata.version(k) for k in reference["versions"]}
    h.require(versions == reference["versions"], "Environment changed")
    rowmeta = h.load_json(fixture[f"row_{rank}"]["path"])
    outdir = HERE / "rows" / f"row_{rank:04d}"
    outdir.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, completed=False, rank=rank, target_access=False,
                  deterministic_teacher_labels=True, final_score_evidence=False,
                  capture_sha256=capture_sha, protocol_sha256=protocol_sha,
                  source_bindings=sources, fixture_bindings=fixture, versions=versions)
    h.write_json(outdir / "started.json", dict(**result, utc=h.utc(), pid=os.getpid(),
                 nice=os.getpriority(os.PRIO_PROCESS, 0), watchdog_s=900,
                 context_cap_s=600, rss_cap_bytes=RSS_LIMIT, reconstruction_block=BLOCK))
    artifacts, post_checks = {}, []
    timer = time.perf_counter()

    def watchdog(signum, frame):
        raise TimeoutError("Fixed900s capture watchdog exhausted")

    signal.signal(signal.SIGALRM, watchdog)
    signal.alarm(900)
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = load_module("memory_capture_current_candidate", PARENT)
        weights = np.load(fixture[f"weights_{rank}"]["path"], mmap_mode="r", allow_pickle=False)
        h.require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
                  and np.isfinite(weights).all(), "Wrong Phase2 weights")
        mlp = MLP(width=1024, depth=16, weights=list(weights),
                  seed=int(derive_estimator_seed(rowmeta["mlp_seed"])), name=f"memory-capture-{rank}")
        observed = dict(layer=-1, pre=[], post=[], heads=[], aux=[],
                        diagnostic_cp_block_flops_subtotal=0, covariance_diagonals=[])
        previous = {}

        def resource_check():
            h.require(h.peak_rss_bytes() < RSS_LIMIT, "Capture exceeded8GiB peak RSS")
            h.require(time.perf_counter() - timer < 900, "Capture watchdog exhausted")

        def save(name, **values):
            arrays = {k: np.asarray(v) for k, v in values.items()}
            for k, v in arrays.items():
                h.require(np.isfinite(v).all(), f"Nonfinite capture {name}:{k}")
            path = outdir / name
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            artifacts[name] = h.binding(path)
            resource_check()

        def pre(l, m, C, mu, Cpre, K21, alive, live_indices,
                response_indices, response_mask, omitted, radial):
            h.require(l == observed["layer"] + 1, "Layer sequence changed")
            observed["layer"] = l
            if l > 14:
                return
            if l:
                h.require(previous["layer"] == l - 1 and
                          np.array_equal(np.asarray(m), previous["m"]) and
                          np.array_equal(np.diag(np.asarray(C)), previous["variance"]),
                          "Previous post-state ownership changed")
            values = dict(mu=mu, C=Cpre, alive=alive, live_indices=live_indices,
                          response_indices=response_indices, response_mask=response_mask,
                          omitted=omitted, incoming_radial=np.array(radial))
            if l:
                values["K21_loss_only"] = K21
            save(f"pre_{l:02d}.npz", **values)
            observed["pre"].append(l)

        def post(l, m, C, p, lam, star, b, alive, radial):
            h.require(l == len(observed["post"]), "Post sequence changed")
            c = np.asarray(C)
            variance = np.diag(c).copy()
            observed["covariance_diagonals"].append(dict(layer=l, minimum=float(variance.min()),
                          negative_count=int((variance < 0).sum()),
                          below_units_floor_count=int((variance < 1e-10).sum())))
            previous.update(layer=l, m=np.asarray(m).copy(), variance=variance)
            values = dict(m=m, C=C, variance=variance, alive=alive, radial=np.array(radial))
            if l < 14:
                pp, ll = np.asarray(p), np.asarray(lam)
                h.require(pp.dtype == np.float32 and ll.dtype == np.float32
                          and np.asarray(star).dtype == np.float32, "Carrier dtype changed")
                cp = np.zeros((1024, 1024), dtype=np.float64)
                for start in range(0, pp.shape[1], BLOCK):
                    stop = min(start + BLOCK, pp.shape[1])
                    block = pp[:, start:stop].astype(np.float64)
                    coeff = ll[start:stop].astype(np.float64)
                    cp += (block * block) @ (block * coeff).T
                    h.require(np.isfinite(cp).all(), "Nonfinite CP reconstruction block")
                    # Ordinary FP64 metering convention:2 per scalar arithmetic op.
                    r = stop-start
                    observed["diagnostic_cp_block_flops_subtotal"] += 2*(2*1024*r + 2*1024*1024*r)
                    del block, coeff
                    resource_check()
                ss = np.asarray(star).astype(np.float64)
                bb = np.asarray(b).astype(np.float64)
                bc = c.copy()
                np.fill_diagonal(bc, variance * np.asarray(alive))
                st = ss.T / 3
                st[np.diag_indices(1024)] += 2 * np.diag(ss) / 3
                physical = cp + st + np.diag(bc)[:, None]*bb[None, :] + 2*bc*bb[:, None]
                checks = []
                ll64 = ll.astype(np.float64)
                for i, j in ((0, 0), (0, 1), (1, 0), (17, 23), (23, 17),
                             (127, 128), (511, 1000), (1023, 1023)):
                    pi, pj = pp[i].astype(np.float64), pp[j].astype(np.float64)
                    terms = ll64*pi*pi*pj
                    star_entry = ss[j, i]/3 + (2*ss[i, i]/3 if i == j else 0)
                    trace_entry = bc[i, i]*bb[j] + 2*bc[i, j]*bb[i]
                    direct = terms.sum(dtype=np.float64) + star_entry + trace_entry
                    scale = max(1., float(np.abs(terms).sum()) + abs(star_entry) + abs(trace_entry))
                    checks.append(float(abs(physical[i, j] - direct) / scale))
                h.require(max(checks) <= 1e-12, "Physical plane orientation/numerical certificate failed")
                h.require(np.isfinite(physical).all(), "Nonfinite physical plane")
                values["physical_K21"] = physical
                post_checks.append(dict(layer=l, columns=int(pp.shape[1]),
                                        max_scaled_direct_error=max(checks)))
            save(f"post_{l:02d}.npz", **values)
            observed["post"].append(l)
            print(json.dumps(dict(rank=rank, captured_post=l,
                       elapsed_s=time.perf_counter()-timer, peak_rss_bytes=h.peak_rss_bytes())), flush=True)

        original_predict, original_head = module._predict_binary, module.gnn_step
        original_map, original_first = module.nonlinear, module._nonlinear_first_gaussian
        source_text = PARENT.read_text()
        funcs = [x for x in ast.parse(source_text).body if isinstance(x, ast.FunctionDef)
                 and x.name == "_predict_binary"]
        source = ast.get_source_segment(source_text, funcs[-1])
        insertions = [
            ("        pair_map = _nonlinear_first_gaussian if l == 0 else nonlinear\n",
             "        _observe_pre(l, m, C, mu, Cpre, K21, alive, live_indices, response_indices, response_mask, omitted, radial)\n"),
            ("        previous, previous_alive = (live_indices, alive)\n",
             "        _observe_post(l, m, C, p, lam, star, b, alive, radial_new)\n"),
        ]
        changed = source
        for anchor, added in insertions:
            h.require(changed.count(anchor) == 1, "Ambiguous observation anchor")
            changed = changed.replace(anchor, added + anchor)
        inverse = changed
        for _, added in insertions:
            inverse = inverse.replace(added, "")
        h.require(inverse == source, "Observation source inverse failed")

        def observe_map(function):
            def wrapped(*args, **kwargs):
                value = function(*args, **kwargs)
                l = observed["layer"]
                if l <= 14:
                    vals = {k: v for k, v in value[4].items() if k != "pre_cov"}
                    save(f"aux_{l:02d}.npz", host=value[0], **vals)
                    observed["aux"].append(l)
                return value
            return wrapped

        def observe_head(l, x, sigma, w, state, payload):
            if 1 <= l <= 14:
                before = np.asarray(state).copy()
                xx = np.asarray(x).copy()
            value = original_head(l, x, sigma, w, state, payload)
            if 1 <= l <= 14:
                save(f"head_{l:02d}.npz", state_previous_consumer_only=before,
                     features_consumer_only=xx, delta_consumer_only=value[0])
                observed["heads"].append(l)
            return value

        module._observe_pre, module._observe_post = pre, post
        exec(compile(changed, "<current-memory-observation-only>", "exec"), module.__dict__)
        module.nonlinear, module._nonlinear_first_gaussian = observe_map(original_map), observe_map(original_first)
        module.gnn_step = observe_head
        bills = []
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                      "BLAS controls failed")
            result["threadpools"] = pools
            try:
                t0 = time.perf_counter()
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=600, quiet=True) as ctx:
                    returned = materialise_predictions(module.Estimator().predict(mlp, h.BUDGET), depth=16, width=1024)
                out = np.asarray(returned)
                h.assert_output(np, out)
                output_sha = hashlib.sha256(out.tobytes()).hexdigest()
                h.require(output_sha == expected["output_bytes_sha256"] and
                          int(ctx.flops_used) == warm_flops + INIT_FLOPS,
                          "Archived served bytes/complete native bill mismatch")
                bills.append(dict(role="instrumented_cold", flops=int(ctx.flops_used),
                             wall_s=time.perf_counter()-t0, residual_s=float(ctx.residual_wall_time_s),
                             diagnostic_only=True, output_bytes_sha256=output_sha))
                artifacts["served.npy"] = h.save_npy(np, outdir / "served.npy", out)
            finally:
                module._predict_binary, module.gnn_step = original_predict, original_head
                module.nonlinear, module._nonlinear_first_gaussian = original_map, original_first
                del module._observe_pre, module._observe_post
            resource_check()
            t0 = time.perf_counter()
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx2:
                control = materialise_predictions(module.Estimator().predict(mlp, h.BUDGET), depth=16, width=1024)
            control_array = np.asarray(control)
            h.assert_output(np, control_array)
            h.require(out.tobytes() == control_array.tobytes() and int(ctx2.flops_used) == warm_flops,
                      "Same-source uninstrumented control mismatch")
            bills.append(dict(role="uninstrumented_warm", flops=int(ctx2.flops_used),
                         wall_s=time.perf_counter()-t0, residual_s=float(ctx2.residual_wall_time_s),
                         diagnostic_only=True, output_bytes_sha256=hashlib.sha256(control_array.tobytes()).hexdigest()))
        h.require(observed["pre"] == list(range(15)) and observed["post"] == list(range(15))
                  and observed["aux"] == list(range(15)) and observed["heads"] == list(range(1, 15)),
                  "Incomplete capture")
        h.check_bindings(sources)
        h.check_bindings(fixture)
        h.check_bindings(artifacts)
        resource_check()
        result.update(passed=True, completed=True, source_inverse_exact=True,
                      archive_output_and_bill_exact=True, same_source_control_exact=True,
                      bills=bills, post_checks=post_checks, capture_inventory=observed)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(artifacts=artifacts, ended_utc=h.utc(), elapsed_s=time.perf_counter()-timer,
                  peak_rss_bytes=h.peak_rss_bytes())
    result_sha = h.write_json(outdir / "result.json", result)
    h.write_json(outdir / "completion.json", dict(passed=result["passed"], result_sha256=result_sha, utc=h.utc()))
    print(json.dumps({k: result.get(k) for k in ("rank", "passed", "elapsed_s", "peak_rss_bytes", "bills", "error")}), flush=True)
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
