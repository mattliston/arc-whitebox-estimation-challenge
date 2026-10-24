"""Inert exact live-output qualification; no network, targets or model fit."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "synthetic"


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
    raise TimeoutError("120s hard synthetic wall cap")


def main():
    if sys.argv[1:] != ["--execute"]:
        print("Inert. --execute runs only fixed synthetic live-output/cost checks.")
        return
    require(not OUT.exists(), "Refusing existing synthetic output")
    require(Path(sys.prefix).resolve() == (HERE.parent.parent / "data_gate/.venv").resolve(),
            "Use OFFPY only")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Requires nice10")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Disable bytecode writes")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        require(os.environ.get(key) == "1", "Requires BLAS1")
    OUT.mkdir()
    begin = time.monotonic()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    result = dict(passed=False, records=[], network_arrays=False, truth_arrays=False,
                  candidate_bill=False, started_utc=datetime.now(timezone.utc).isoformat())
    try:
        from source import sources, PARENT, PARENT_SHA
        before = {str(p): digest(p) for p in
                  (PARENT, HERE / "source.py", HERE / "PROTOCOL.md", Path(__file__))}
        functions, terminal, _ = sources()
        result.update(source_sha256=digest(HERE / "source.py"), parent_sha256=PARENT_SHA,
                      protocol_sha256=digest(HERE / "PROTOCOL.md"), checker_sha256=digest(__file__),
                      terminal_function_sha256=sha256(terminal.encode()).hexdigest(),
                      source_bindings=before)
        write(OUT / "started.json", result)
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits
        require(importlib.metadata.version("flopscope") == "0.12.0", "Requires FNP0.12")
        flops.configure(symmetry_warnings=False)
        namespace = dict(flops=flops, fnp=fnp)
        for name in ("gaussian_raw", "derivatives", "_nonlinear_pair_batch"):
            exec(compile(functions[name], str(PARENT) + ":" + name, "exec"), namespace)
        exec(compile(terminal, "qualified_terminal_function", "exec"), namespace)

        def fixture(n, kind):
            rng = np.random.default_rng(937211 + n)
            basis = rng.normal(size=(n, 8))
            basis /= np.sqrt(np.sum(basis * basis, axis=1))[:, None]
            strength = (.65, .65, .95)[kind]
            correlation = strength * (basis @ basis.T) + (1 - strength) * np.eye(n)
            sigma = np.exp(np.linspace(np.log(.125), np.log(3.), n))
            v = sigma * sigma
            covariance = correlation * np.outer(sigma, sigma)
            mu = sigma * np.linspace(-6., 6., n)
            skew, kurtosis = ((0., 0.), (.03, .03), (.1, .12))[kind]
            k21 = skew * (v[:, None] * sigma[None, :]) * (.5 * correlation + .1 * np.eye(n))
            k4 = kurtosis * v * v
            k22 = kurtosis * (np.outer(v, v) + 2 * covariance * covariance) / 3
            return mu, covariance, k21, k4, k22

        def run(fn, arrays):
            args = tuple(fnp.asarray(x, dtype=fnp.float64) for x in arrays)
            start = time.monotonic()
            with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=120, quiet=True) as budget:
                outputs = fn(*args, metric="covariance")
            elapsed = time.monotonic() - start
            live = dict(mean=np.array(outputs[0], copy=True), covariance=np.array(outputs[1], copy=True))
            live.update({"aux/" + k: np.array(v, copy=True) for k, v in outputs[4].items()})
            require(all(np.isfinite(x).all() for x in live.values()), "Nonfinite live output")
            require(rss() < 2 * 1024**3, "2GiB synthetic RSS cap")
            return live, int(budget.flops_used), elapsed

        with threadpool_limits(limits=1):
            for n, kind in [(n, k) for n in (32, 64) for k in (0, 1, 2)] + [(1024, 1)]:
                arrays = fixture(n, kind)
                parent, fa, ta = run(namespace["_nonlinear_pair_batch"], arrays)
                changed, fb, tb = run(namespace["_nonlinear_terminal_live"], arrays)
                exact = parent.keys() == changed.keys() and all(
                    a.shape == changed[k].shape and a.dtype == changed[k].dtype
                    and a.tobytes() == changed[k].tobytes() for k, a in parent.items())
                record = dict(n=n, kind=kind, live_byte_exact=bool(exact), parent_function_flops=fa,
                              terminal_function_flops=fb, saving_flops=fa-fb,
                              parent_wall_s=ta, terminal_wall_s=tb)
                result["records"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                require(exact and fb < fa, "Fixed byte-exact/strict-cost gate failed")
        after = {p: digest(p) for p in before}
        require(before == after, "Source changed during synthetic qualification")
        result.update(passed=True, sources_unchanged=True)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
    result.update(elapsed_s=time.monotonic()-begin, peak_rss_bytes=rss(),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / "result.json", result)
    write(OUT / "completion.json", dict(passed=result["passed"], result_sha256=digest(OUT / "result.json")))
    print(json.dumps(dict(passed=result["passed"], elapsed_s=result["elapsed_s"],
                          peak_rss_bytes=result["peak_rss_bytes"], error=result.get("error"))), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
