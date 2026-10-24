"""Inert fixed active-congruence fixtures. No network or truth arrays."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "resident_gram512_v1/estimator.py"
PARENT_SHA = "0c21db7a74ae0a271d52fddb5b925318be61100e0676393c87f2f15ce9a997dc"
OUT = HERE / "synthetic"
WIDTHS = (1024, 992, 960, 896, 832, 768)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def timeout(_signum, _frame):
    raise TimeoutError("Fixed120s synthetic cap")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if sys.argv[1:] != ["--execute"]:
        print("Inert. --execute runs the fixed synthetic prerequisites only.")
        return
    require(not OUT.exists(), "Refusing existing outputs")
    require(Path(sys.prefix).resolve() == (HERE.parent.parent / "data_gate/.venv").resolve(),
            "Use OFFPY, never AP/threadpoolctl")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Requires nice10")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Disable bytecode writes")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        require(os.environ.get(name) == "1", "Requires BLAS1")
    require(digest(PARENT) == PARENT_SHA, "Frozen parent changed")
    before = {str(p): digest(p) for p in
              (PARENT, HERE / "congruence.py", HERE / "PROTOCOL.md", Path(__file__))}
    OUT.mkdir()
    result = dict(passed=False, source_bindings=before, records=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  candidate_bill=False, network_arrays=False, truth_arrays=False,
                  fixed_fixture_order=list(WIDTHS), nice=os.getpriority(os.PRIO_PROCESS, 0))
    write(OUT / "started.json", result)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        versions = {name: importlib.metadata.version(name)
                    for name in ("flopscope", "whestbench", "numpy")}
        require(versions == {"flopscope": "0.12.0", "whestbench": "0.16.0", "numpy": "2.4.6"},
                "Requires fixed OFFPY versions")
        result["versions"] = versions
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info, threadpool_limits
        flops.configure(symmetry_warnings=False)
        parent = load(PARENT, "active_congruence_parent_source")
        changed = load(HERE / "congruence.py", "active_congruence_fixed_helper")
        changed.fnp, changed._mm = fnp, parent._mm
        changed._resident_gram512 = parent._resident_gram512
        with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=120, quiet=True) as setup:
            arena = fnp.empty((parent._PACKED_ELEMENTS,), dtype=fnp.float32)
            parent._SCRATCH = [arena,
                parent._cp3_bind_arena_views(arena, parent._PACKED_PUSH),
                parent._cp3_bind_arena_views(arena, parent._PACKED_READ),
                None, None, None, None]
            for depth, specs in ((3, parent._PACKED_SQUARE), (4, parent._PACKED_SQUARE4),
                                 (5, parent._PACKED_SQUARE5)):
                parent._SCRATCH[depth] = parent._cp3_bind_arena_views(arena, specs)
        result["shared_setup_flops"] = int(setup.flops_used)

        def original(wa, ca):
            return parent._mm(wa.T, parent._mm(ca, wa))

        def measured(fn, wa, ca):
            begin = time.monotonic()
            with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=120, quiet=True) as ctx:
                value = fn(wa, ca)
            elapsed = time.monotonic() - begin
            array = np.asarray(value).copy()
            require(array.shape == (1024, 1024) and array.dtype == np.dtype("float32")
                    and bool(np.isfinite(array).all()), "Finite concrete FP32 output")
            require(rss() < 2 * 1024**3, "Fixed2GiB RSS cap")
            return array, int(ctx.flops_used), elapsed

        with threadpool_limits(limits=1):
            pools = threadpool_info()
            require(all(p["num_threads"] == 1 for p in pools if p.get("user_api") == "blas"),
                    "Reported BLAS count differs from1")
            result["threadpools"] = pools
            for a in WIDTHS:
                rng = np.random.default_rng(938612 + a)
                wa = rng.normal(0, np.sqrt(2 / 1024), (a, 1024)).astype(np.float32)
                factor = rng.normal(0, 1 / np.sqrt(8), (a, 8))
                diagonal = np.geomspace(.0001, .3, a)
                if a == 896:
                    diagonal[::9] *= -1
                base = .03 * (factor @ factor.T) + np.diag(diagonal)
                skew = rng.normal(0, 1, (a, a))
                ca = (base + 1e-9 * (skew - skew.T)).astype(np.float32)
                w64, c64 = wa.astype(np.float64), ca.astype(np.float64)
                exact_original = w64.T @ (c64 @ w64)
                exact_symmetric = w64.T @ (((c64 + c64.T) * .5) @ w64)
                cs32 = (ca + ca.T) * np.float32(.5)
                exact_rounded_symmetric = w64.T @ (cs32.astype(np.float64) @ w64)
                norm = max(float(np.linalg.norm(exact_symmetric)), 1e-30)
                old, fa, ta = measured(original, wa, ca)
                new, fb, tb = measured(changed._active_congruence512, wa, ca)
                errors = {}
                for name, value, reference in (("parent", old, exact_original),
                                               ("changed", new, exact_symmetric)):
                    error = value.astype(np.float64) - reference
                    errors[name] = dict(relative_frobenius=float(np.linalg.norm(error) / norm),
                                        max_abs=float(np.max(np.abs(error))))
                discrepancy = float(np.linalg.norm(new.astype(np.float64) - old.astype(np.float64)) / norm)
                record = dict(active=a, parent_flops=fa, changed_flops=fb, saved_flops=fa-fb,
                              parent_wall_s=ta, changed_wall_s=tb, errors=errors,
                              changed_parent_relative_frobenius=discrepancy,
                              covariance_asymmetry_relative=float(np.linalg.norm(c64-c64.T)/np.linalg.norm(c64)),
                              symmetric_original_reference_relative=float(np.linalg.norm(exact_symmetric-exact_original)/norm),
                              rounded_symmetric_reference_relative=float(np.linalg.norm(exact_rounded_symmetric-exact_symmetric)/norm),
                              changed_output_asymmetry_relative=float(np.linalg.norm(new.astype(np.float64)-new.astype(np.float64).T)/norm),
                              parent_output_sha256=sha256(old.tobytes()).hexdigest(),
                              changed_output_sha256=sha256(new.tobytes()).hexdigest())
                result["records"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                require(all(e["relative_frobenius"] <= 2e-5 and e["max_abs"] <= 5e-5
                            for e in errors.values()) and discrepancy <= 2e-5,
                        "Fixed numerical fidelity gate failed")
                require(fb < fa, "Fixed strict function-cost gate failed")
        require({p: digest(p) for p in before} == before, "Frozen source changed during test")
        require(rss() < 2 * 1024**3 and time.monotonic()-started < 120,
                "Fixed final synthetic RSS/wall cap")
        result.update(passed=True, sources_unchanged=True)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(elapsed_s=time.monotonic()-started, peak_rss_bytes=rss(),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / "result.json", result)
    write(OUT / "completion.json", dict(passed=result["passed"], result_sha256=digest(OUT / "result.json")))
    print(json.dumps(dict(passed=result["passed"], elapsed_s=result["elapsed_s"],
                          peak_rss_bytes=result["peak_rss_bytes"], error=result.get("error"))), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
