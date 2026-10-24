"""Inert single-seed terminal-summary screen; no truth or network execution."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "terminal_summary64_proxy_v1"
CAPTURE = HERE / "terminal_summary_capture_v2"
PARENT = HERE / "candidate_response_pair_polynomial_f32_v1.py"
HELPER = HERE / "check_response_rows_adjusted_v1.py"
PROTOCOL = HERE / "TERMINAL_SUMMARY64_PROXY_PROTOCOL_V1.md"
PINS = {
    PARENT: "0b84ab6fa96d32804c9adec350d41766e47d09a71a936b95652893bb1e364a96",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    CAPTURE / "result.json": "5b30d63806115d1686e4e6dac9a556fda0815c580ca77140daeddb6ded15ddbd",
}
Q = 64
SEED_SALT = 0x5445524D3634


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != PINS[HELPER]:
        raise RuntimeError("Frozen inert helper changed")
    h = import_path("terminal64_proxy_helpers", HELPER)
    h.require(Path(sys.prefix).resolve() == h.GATE_ENV.resolve()
              and "anaconda3" not in sys.executable, "OFFPY only")
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    h.require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
              and all(os.environ.get(k) == "1" for k in
                      ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")),
              "Explicit launch controls required")
    bindings = {str(p): h.binding(p) for p in (*PINS, PROTOCOL, Path(__file__).resolve())}
    h.require(all(bindings[str(p)]["sha256"] == sha for p, sha in PINS.items()),
              "Frozen source/capture pin mismatch")
    capture = h.load_json(CAPTURE / "result.json")
    h.require(capture["qualification_passed"]
              and capture["archive_output_and_complete_bill_exact"]
              and not capture["truth_files_accessed"], "Unqualified capture")
    h.check_bindings(capture["artifacts"])
    versions = {k: importlib.metadata.version(k) for k in
                ("numpy", "flopscope", "whestbench", "threadpoolctl")}
    h.require(versions == capture["versions"], "Capture environment mismatch")
    row = h.load_json(capture["fixture_bindings"]["row_35"]["path"])
    OUT.mkdir(exist_ok=False)
    started = dict(utc=h.utc(), pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                   source_bindings=bindings, versions=versions, q=Q, seed_salt=SEED_SALT,
                   rank=35, truth_files_accessed=False, estimator_calls=0, seed_count=1)
    start_sha = h.write_json(OUT / "started.json", started)
    result = dict(schema="terminal_summary64_proxy_v1", started_sha256=start_sha,
                  completed=False, screen_passed=False, final_score_evidence=False,
                  replacement_compute_evidence=False, truth_files_accessed=False,
                  estimator_calls=0, q=Q, seed_salt=SEED_SALT, source_bindings=bindings)
    artifacts = {}
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.seeds import derive_estimator_seed
        module = import_path("terminal64_proxy_parent", PARENT)
        names = ("K21", "cp_pre", "pre_B", "bpre", "star_pre", "mu", "pre_C", "k4", "k22",
                 "host", "post_C", "aux_sigma", "aux_post_v", "w", "features", "state_previous",
                 "delta", "output")
        inputs = {name: np.load(CAPTURE / f"{name}.npy", allow_pickle=False) for name in names}
        h.require(all(np.isfinite(a).all() for a in inputs.values()), "Nonfinite captured input")
        seed = (int(derive_estimator_seed(row["mlp_seed"])) ^ SEED_SALT) & ((1 << 63) - 1)
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                      "BLAS thread mismatch")
            t0 = time.perf_counter()
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                a = {name: fnp.asarray(value) for name, value in inputs.items()}
                P = {k: fnp.asarray(v, dtype=fnp.float32) for k, v in module._GNN_LITERAL.items()}
                n = int(a["K21"].shape[0])
                h.require(n == 1024, "Only captured Phase2 geometry")
                J = fnp.random.default_rng(seed).choice(n, size=Q, replace=False)
                cp0 = fnp.zeros((n, n), dtype=fnp.float64)
                fnp.fill_diagonal(cp0, fnp.diag(a["cp_pre"]))
                K0 = cp0 + a["star_pre"] + fnp.diag(a["pre_B"])[:, None] * a["bpre"][None, :] + 2 * a["pre_B"] * a["bpre"][:, None]
                h.require(bool(fnp.all(fnp.diag(K0) == fnp.diag(a["K21"]))), "Baseline changed exact total diagonal")
                host0, C0, _, _, aux0 = module.nonlinear(a["mu"], a["pre_C"], K0, a["k4"], a["k22"], "covariance")
                h.require(bool(fnp.all(host0 == a["host"]))
                          and bool(fnp.all(aux0["sigma"] == a["aux_sigma"]))
                          and bool(fnp.all(aux0["post_v"] == a["aux_post_v"])),
                          "Baseline changed host or marginal normalization")
                den = (a["aux_sigma"] * a["aux_sigma"])[:, None] * a["aux_sigma"][None, :]
                S0, S = K0 / den, a["K21"] / den
                row2 = fnp.sum(S0 * S0, axis=1) + (n / Q) * fnp.sum(S[:, J] * S[:, J] - S0[:, J] * S0[:, J], axis=1)
                col2 = fnp.sum(S0 * S0, axis=0) + (n / Q) * fnp.sum(S[J, :] * S[J, :] - S0[J, :] * S0[J, :], axis=0)
                postscale = fnp.sqrt(fnp.maximum(a["aux_post_v"], 1e-12))
                postden = fnp.outer(postscale, postscale)
                Z0 = fnp.clip(C0 / postden, -1, 1)
                Z = fnp.clip(a["post_C"] / postden, -1, 1)
                post2 = fnp.mean(Z0 * Z0, axis=1) + fnp.sum(Z[:, J] * Z[:, J] - Z0[:, J] * Z0[:, J], axis=1) / Q
                post1 = fnp.mean(Z0, axis=1) + fnp.sum(Z[:, J] - Z0[:, J], axis=1) / Q
                finite = all(bool(fnp.all(fnp.isfinite(v))) for v in (row2, col2, post2, post1))
                nonnegative = all(bool(fnp.all(v >= 0)) for v in (row2, col2, post2))
                h.require(finite and nonnegative, "Nonfinite/negative summary; no repair or resampling")
                x = a["features"].copy()
                x[:, 22] = fnp.sqrt(row2).astype(fnp.float32)
                x[:, 23] = fnp.sqrt(col2).astype(fnp.float32)
                x[:, 27] = fnp.sqrt(post2).astype(fnp.float32)
                x[:, 28] = post1.astype(fnp.float32)
                original_delta, _ = module.gnn_step(15, a["features"], a["aux_sigma"], a["w"], a["state_previous"], P)
                delta, _ = module.gnn_step(15, x, a["aux_sigma"], a["w"], a["state_previous"], P)
                original_output = fnp.maximum(a["host"] + original_delta.astype(fnp.float64), 0).astype(fnp.float32)
                output = fnp.maximum(a["host"] + delta.astype(fnp.float64), 0).astype(fnp.float32)
            wall = time.perf_counter() - t0
        h.require(np.array_equal(np.asarray(original_delta), inputs["delta"])
                  and np.array_equal(np.asarray(original_output), inputs["output"][15]),
                  "Original current-P head/served-output control failed")
        output_np = np.asarray(output)
        drift = output_np.astype(np.float64) - inputs["output"][15].astype(np.float64)
        rms, maximum = float(np.sqrt(np.mean(drift * drift))), float(np.max(np.abs(drift)))
        h.require(np.isfinite(output_np).all() and (output_np >= 0).all()
                  and h.peak_rss_bytes() < h.RSS_LIMIT and wall < 120, "Proxy numerical/resource failure")
        feature_rms = np.sqrt(np.mean((np.asarray(x).astype(np.float64) - inputs["features"]) ** 2, axis=0))
        for name, value in {"indices": np.asarray(J), "features": np.asarray(x), "delta": np.asarray(delta),
                            "output15": output_np, "row2": np.asarray(row2), "col2": np.asarray(col2),
                            "post2": np.asarray(post2), "post1": np.asarray(post1)}.items():
            artifacts[f"{name}.npy"] = h.save_npy(np, OUT / f"{name}.npy", value)
            h.require(h.peak_rss_bytes() < h.RSS_LIMIT, "Save resource failure")
        h.check_bindings(bindings)
        h.check_bindings(capture["artifacts"])
        h.check_bindings(artifacts)
        result.update(completed=True, screen_passed=bool(rms <= 5e-7 and maximum <= 5e-6),
                      original_head_and_output_bytes_exact=True, baseline_marginals_exact=True,
                      nonnegative_squared_estimates=True, hashes_unchanged=True, seed=seed,
                      served_drift_rms=rms, served_drift_max_abs=maximum,
                      feature_drift_rms={str(i): float(feature_rms[i]) for i in (22, 23, 27, 28)},
                      optimistic_exact_samples=True, optimistic_exact_linear_features=True,
                      diagnostic_flops=int(ctx.flops_used), diagnostic_wall_s=wall,
                      diagnostic_residual_s=float(ctx.residual_wall_time_s),
                      peak_rss_bytes=h.peak_rss_bytes(), threadpools=pools)
    except BaseException:
        result["error"] = traceback.format_exc()
    result.update(artifacts=artifacts, ended_utc=h.utc())
    seal = h.write_json(OUT / "result.json", result)
    h.write_json(OUT / "completion.json", dict(result_sha256=seal, completed=result["completed"],
                 screen_passed=result["screen_passed"], utc=h.utc()))
    print({k: v for k, v in result.items() if k not in ("artifacts", "source_bindings", "threadpools")}, flush=True)
    return 0 if result["completed"] else 1


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
