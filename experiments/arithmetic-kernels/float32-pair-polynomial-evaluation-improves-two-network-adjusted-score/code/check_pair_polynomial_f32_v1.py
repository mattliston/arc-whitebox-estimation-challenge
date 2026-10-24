"""Fixed synthetic fidelity/function-price gate; inert without --run."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "pair_polynomial_f32_preflight_v1"
ENV = HERE.parent / "data_gate/.venv"
PINS = {
    "candidate_response_rows_current_runtime_v1.py": "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172",
    "PAIR_POLYNOMIAL_F32_COMPONENT_PREFLIGHT_V1.md": "213185c374ec7179cfdf1602d31d6ccba6da47062cae3ed02fe94e65bf57c2d2",
    "pair_polynomial_f32_source_v1.py": "fb758f0e3e36a0ee29cb7f69490af83ece4c7e12424cb82c5122ee4e00113198",
}
RETAINED_COMPLETE_F = 472413324017.2143


def require(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def main():
    if sys.argv[1:] != ["--run"]:
        print("Inert. Fixed synthetic preflight requires --run.")
        return
    require(not OUT.exists(), "Refusing existing preflight directory")
    OUT.mkdir()
    began = time.monotonic()
    records = []

    def guard():
        require(time.monotonic()-began <= 120., "120s diagnostic wall limit")
        require(rss() <= 2*1024**3, "2GiB diagnostic RSS limit")

    try:
        require(Path(sys.prefix).resolve() == ENV.resolve(), "Use data_gate/.venv only")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice >=10 required")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            require(os.environ.get(name) == "1", f"Set {name}=1")
        require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
        before = {name: digest(HERE/name) for name in (*PINS, Path(__file__).name)}
        require(all(before[name] == value for name, value in PINS.items()), "Source pin mismatch")
        require(importlib.metadata.version("flopscope") == "0.12.0", "Native FlopScope0.12 required")
        write(OUT/"started.json", dict(utc=utc(), sources=before, executable=sys.executable,
                                       pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                                       network_and_truth_inputs=False,
                                       random_seed_rule="937211+n; each class restarts that seed",
                                       small_shapes=[32, 64], classes=[0, 1, 2],
                                       large_price_fixture=dict(n=1024, fixture_class=1),
                                       score_or_full_bill=False))
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits
        from pair_polynomial_f32_source_v1 import functions, sources

        flops.configure(symmetry_warnings=False)
        parent, changed = functions(flops, fnp)
        literal, transformed = sources()
        transformed_sha = sha256(transformed.encode()).hexdigest()

        def fixture(n, kind):
            rng = np.random.default_rng(937211+n)
            basis = rng.normal(size=(n, 8))
            basis /= np.sqrt(np.sum(basis*basis, axis=1))[:, None]
            strength = (0.65, 0.65, 0.95)[kind]
            correlation = strength*(basis@basis.T)+(1-strength)*np.eye(n)
            sigma = np.exp(np.linspace(np.log(.125), np.log(3.), n))
            v = sigma*sigma
            covariance = correlation*np.outer(sigma, sigma)
            alpha = np.linspace(-6., 6., n)
            mu = sigma*alpha
            skew, kurtosis = ((0., 0.), (.03, .03), (.1, .12))[kind]
            k21 = skew*(v[:, None]*sigma[None, :])*(.5*correlation+.1*np.eye(n))
            k4 = kurtosis*v*v
            k22 = kurtosis*(np.outer(v, v)+2*covariance*covariance)/3
            return (mu, covariance, k21, k4, k22), sigma

        def run(function, arrays):
            # Shared diagnostic ingress is outside this FUNCTION-only bill.
            args = tuple(fnp.asarray(array, dtype=fnp.float64) for array in arrays)
            start = time.monotonic()
            with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=120, quiet=True) as budget:
                result = function(*args, metric="covariance")
            elapsed = time.monotonic()-start
            out = tuple(np.array(value, copy=True) for value in result[:4])
            aux = {key: np.array(value, copy=True) for key, value in result[4].items()}
            require(all(np.isfinite(value).all() for value in (*out, *aux.values())),
                    "Nonfinite synthetic result")
            guard()
            return out, aux, int(budget.flops_used), elapsed

        def pair(n, kind):
            arrays, sigma = fixture(n, kind)
            a, ax, fa, ta = run(parent, arrays)
            b, bx, fb, tb = run(changed, arrays)
            mean_exact = np.array_equal(a[0], b[0]) and a[0].dtype == b[0].dtype
            aux_exact = ax.keys() == bx.keys() and all(
                np.array_equal(ax[key], bx[key]) and ax[key].dtype == bx[key].dtype for key in ax)
            diag_exact = all(np.array_equal(np.diag(a[index]), np.diag(b[index])) for index in (1, 2))
            errors = dict(cov_scaled_max=float(np.max(np.abs(a[1]-b[1])/np.outer(sigma, sigma))),
                          k21_scaled_max=float(np.max(np.abs(a[2]-b[2])/(sigma[:, None]**2*sigma[None, :]))),
                          radial_abs=float(np.abs(a[3]-b[3])))
            passed = mean_exact and aux_exact and diag_exact and errors["cov_scaled_max"] <= 1e-6 \
                and errors["k21_scaled_max"] <= 5e-6 and errors["radial_abs"] <= 5e-6
            rec = dict(n=n, fixture_class=kind, mean_exact=mean_exact, aux_exact=aux_exact,
                       diagonals_exact=diag_exact, errors=errors, numerical_pass=bool(passed),
                       parent_function_flops=fa, fp32poly_function_flops=fb,
                       parent_function_wall_s=ta, fp32poly_function_wall_s=tb)
            records.append(rec)
            print(json.dumps(rec, sort_keys=True, allow_nan=False), flush=True)
            return rec

        with threadpool_limits(limits=1):
            for n in (32, 64):
                for kind in (0, 1, 2):
                    pair(n, kind)
            small_pass = all(row["numerical_pass"] for row in records)
            big = pair(1024, 1) if small_pass else None
        after = {name: digest(HERE/name) for name in before}
        require(before == after, "Source changed during preflight")
        guard()
        saving = big["parent_function_flops"]-big["fp32poly_function_flops"] if big else None
        projection = 15*saving/RETAINED_COMPLETE_F if big else None
        advance = bool(small_pass and big and big["numerical_pass"] and saving > 0 and projection >= .005)
        result = dict(utc=utc(), decision="SYNTHETIC_COMPONENT_GATE_PASS" if advance else "STOP_FIXED_PARTITION",
                      records=records, small_numerical_gate_pass=small_pass,
                      function_saving_flops=saving, projected_15_call_saving_fraction=projection,
                      parent_complete_bill_reference=RETAINED_COMPLETE_F,
                      projection_scope="Unchanged-geometry planning only; not a measured whole-candidate bill",
                      raw_or_adjusted_measured=False, network_or_truth_arrays=False,
                      no_full_candidate_admission=True, sources_unchanged=True,
                      transformed_function_sha256=transformed_sha,
                      started_sha256=digest(OUT/"started.json"),
                      wall_seconds=time.monotonic()-began, peak_rss_bytes=rss(),
                      flopscope_version=importlib.metadata.version("flopscope"), numpy_version=np.__version__)
        write(OUT/"result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    except BaseException:
        write(OUT/"failure.json", dict(utc=utc(), error=traceback.format_exc(), records=records,
                                       wall_seconds=time.monotonic()-began, peak_rss_bytes=rss(),
                                       decision="STOP; no qualifying preflight"))
        raise


if __name__ == "__main__":
    main()
