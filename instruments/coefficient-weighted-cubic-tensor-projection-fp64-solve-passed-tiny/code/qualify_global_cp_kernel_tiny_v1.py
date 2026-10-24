"""Fixed synthetic qualification; no network, label, head, or candidate reads."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp


HERE = Path(__file__).resolve().parent
OUT = HERE / "global_cp_kernel_tiny_v1.json"
PLAN = HERE / "GLOBAL_CP_KERNEL_PREFLIGHT_V1.md"


def tensor(p, weights):
    p, weights = np.asarray(p, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    return np.einsum("ir,jr,kr,r->ijk", p, p, p, weights)


def star_tensor(star):
    star = np.asarray(star, dtype=np.float64)
    n = len(star)
    eye = np.eye(n)
    return (np.einsum("ij,ki->ijk", eye, star)
            + np.einsum("ik,ji->ijk", eye, star)
            + np.einsum("jk,ij->ijk", eye, star)) / 3


def relative(error, reference):
    return float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-300))


def fixtures():
    rng = np.random.default_rng(719)
    n, a, k = 8, 6, 6
    triples = list(itertools.combinations(range(a), 3))[:k]
    first = np.zeros((a, k))
    for j, triple in enumerate(triples):
        first[list(triple), j] = 1
    second = rng.normal(size=(a, k)) / np.sqrt(a)
    second += .5 * rng.normal(size=(a, 1))
    for name, base in (("orthogonal_feature", first), ("correlated_signed_pair", second)):
        p = np.zeros((n, 2*k), dtype=np.float32)
        p[:a, :k] = base
        p[:a, k:] = base + .1 * rng.normal(size=(a, k))
        positive = rng.uniform(.5, 1.5, size=k).astype(np.float32)
        weights = np.concatenate((positive, -positive)).astype(np.float32)
        star = (.05 * rng.normal(size=(n, n))).astype(np.float32)
        query = rng.normal(size=(n, 5)).astype(np.float64) / np.sqrt(n)
        yield name, a, p, weights, star, query


def qualify(name, a, p0, lam0, star0, query):
    n, m = p0.shape
    k = m // 2
    selected = np.arange(k)
    stages = {}
    with flops.BudgetContext(flop_budget=2**28, quiet=True) as ctx:
        def stage(label, before):
            stages[label] = int(ctx.flops_used) - before

        before = int(ctx.flops_used)
        p = fnp.asarray(p0, dtype=fnp.float32)
        lam = fnp.asarray(lam0, dtype=fnp.float32)
        old_star = fnp.asarray(star0, dtype=fnp.float32)
        z = p[:, :k]
        p1, z1 = p[:a], z[:a]
        p2, z2 = p1*p1, z1*z1
        p3, z3 = p2*p1, z2*z1
        stage("inputs_and_powers", before)
        before = int(ctx.flops_used)
        s1, s2, s3 = z1.T @ p1, z2.T @ p2, z3.T @ p3
        stage("three_fp32_cross_grams", before)
        before = int(ctx.flops_used)
        s1 = s1.astype(fnp.float64)
        s2 = s2.astype(fnp.float64)
        s3 = s3.astype(fnp.float64)
        kernel = s1*s1*s1 - 3*s1*s2 + 2*s3
        gram = kernel[:, :k]
        gram = (gram + gram.T) / 2
        rhs = kernel @ lam.astype(fnp.float64)
        stage("polynomial_gram_and_rhs", before)
        before = int(ctx.flops_used)
        coefficients = fnp.linalg.solve(gram, rhs)
        stage("fp64_solve", before)
        before = int(ctx.flops_used)
        stored_coefficients = coefficients.astype(fnp.float32)
        old_plane = ((p*p) @ (p*lam).T).astype(fnp.float64)
        new_plane = ((z*z) @ (z*stored_coefficients).T).astype(fnp.float64)
        rest = old_plane-new_plane
        correction = 3*rest.T
        fnp.fill_diagonal(correction, fnp.diag(rest))
        new_star = (old_star.astype(fnp.float64) + correction).astype(fnp.float32)
        stage("both_actual_planes_and_star_restoration", before)
        total = int(ctx.flops_used)

    # Independent explicit-tensor reference work, outside the measured circuit.
    index = np.indices((n,n,n))
    distinct = (index[0] != index[1]) & (index[0] != index[2]) & (index[1] != index[2])
    repeated = ~distinct
    atom_features = np.einsum("ir,jr,kr->ijkr", p0.astype(np.float64),
                              p0.astype(np.float64), p0.astype(np.float64))[distinct]
    exact_kernel = atom_features[:, selected].T @ atom_features
    exact_gram = exact_kernel[:, selected]
    exact_rhs = exact_kernel @ lam0.astype(np.float64)
    exact_coefficients = np.linalg.solve(exact_gram, exact_rhs)
    c = np.asarray(coefficients)
    old = tensor(p0, lam0) + star_tensor(star0)
    new = tensor(p0[:, selected], np.asarray(stored_coefficients)) + star_tensor(np.asarray(new_star))
    residual = old-new
    distinct_residual = np.where(distinct, residual, 0)
    actual_query = np.einsum("ijk,iu,jv,kw->uvw", residual, query, query, query)
    distinct_query = np.einsum("ijk,iu,jv,kw->uvw", distinct_residual, query, query, query)
    old_query = np.einsum("ijk,iu,jv,kw->uvw", old, query, query, query)
    normal_residual = relative(np.asarray(gram) @ c - np.asarray(rhs), np.asarray(rhs))
    checks = dict(
        kernel_relative=relative(np.asarray(kernel)-exact_kernel, exact_kernel),
        coefficient_relative=relative(c-exact_coefficients, exact_coefficients),
        normal_equation_relative=normal_residual,
        repeated_tensor_relative=relative(residual[repeated], old),
        contracted_repeated_relative=relative(actual_query-distinct_query, old_query),
        finite=bool(np.isfinite(c).all() and np.isfinite(new).all()),
    )
    expected_grams = 3*(2*a*k*m-k*m)
    expected_solve = 2*(2*k**3//3+2*k*k)
    gates = dict(
        kernel=checks["kernel_relative"] <= 1e-5,
        coefficients=checks["coefficient_relative"] <= 1e-3,
        normal_equation=checks["normal_equation_relative"] <= 1e-10,
        repeated=checks["repeated_tensor_relative"] <= 3e-6,
        contracted_repeated=checks["contracted_repeated_relative"] <= 1e-5,
        cross_gram_bill=stages["three_fp32_cross_grams"] == expected_grams,
        solve_bill=stages["fp64_solve"] == expected_solve,
        finite=checks["finite"],
    )
    return dict(fixture=name, n=n, active=a, old_atoms=m, selected_atoms=k,
                passed=bool(all(gates.values())), gates=gates, errors=checks,
                metered_stages=stages, total_native_flops=total,
                expected_cross_gram_flops=expected_grams, expected_solve_flops=expected_solve,
                exact_gram_condition=float(np.linalg.cond(exact_gram)),
                coefficient_absolute_max=float(np.max(np.abs(c))),
                synthetic_distinct_error_ratio=relative(distinct_residual, old),
                input_sha256=hashlib.sha256(p0.tobytes()+lam0.tobytes()+star0.tobytes()).hexdigest())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.time()
    env = {key: os.environ.get(key) for key in
           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    assert set(env.values()) == {"1"}, env
    assert nice >= 10, nice
    tests = []
    failure = None
    try:
        for args in fixtures():
            result = qualify(*args)
            tests.append(result)
            print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    result = dict(passed=bool(failure is None and len(tests)==2 and all(t["passed"] for t in tests)),
                  tests=tests, failure=failure, blas_env=env, nice=nice,
                  wall_s=time.time()-started,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                  qualification="Tiny synthetic algebra and native API bills only. No P2 network, targets, learned heads, current-bank bill, raw error, candidate or score evidence.")
    with OUT.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
