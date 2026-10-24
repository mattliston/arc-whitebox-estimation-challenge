#!/usr/bin/env python3
"""Inert until --run: one n=4 tree/Wick certificate, not a raw-score test.

Run only after external source/resource review. No estimator imports, network
weights, targets, MPS, quadrature, source tuning, or candidate-file creation.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build.py"
OUT = HERE / "tiny_v1"
SEED = 9309101
RELATIVE = 1e-11
ABSOLUTE = 1e-12
REF_TINY = 1e-12
MAX_RSS = 1_000_000_000
MAX_WALL = 30.0
BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def alarm_handler(signum, frame):
    raise TimeoutError("fixed tiny certificate 30-second limit")


def trees():
    """All labeled trees via length-two Pruefer words; no closed-form rays."""
    found = []
    for word in itertools.product(range(4), repeat=2):
        degree = [1 + word.count(i) for i in range(4)]
        remaining = degree.copy()
        edges = []
        for vertex in word:
            leaf = next(i for i, d in enumerate(remaining) if d == 1)
            edges.append(tuple(sorted((leaf, vertex))))
            remaining[leaf] -= 1
            remaining[vertex] -= 1
        leaves = [i for i, d in enumerate(remaining) if d == 1]
        edges.append(tuple(leaves))
        found.append((tuple(sorted(edges)), tuple(degree)))
    require(len({edges for edges, degree in found}) == 16, "16 unique trees")
    require(sum(sorted(degree) == [1, 1, 2, 2] for edges, degree in found) == 12,
            "12 path multiplicities")
    require(sum(sorted(degree) == [1, 1, 1, 3] for edges, degree in found) == 4,
            "4 star multiplicities")
    return found


def pairings(stubs):
    if not stubs:
        yield ()
        return
    first = stubs[0]
    for index in range(1, len(stubs)):
        for tail in pairings(stubs[1:index] + stubs[index + 1:]):
            yield ((first, stubs[index]),) + tail


def wick_multiplier(edges, degree):
    # Six distinct stubs, but each stores its vertex owner. Normal ordering
    # forbids intra-vertex pairs. Only this connected three-edge tree is kept;
    # higher-edge sectors of the cubic-Hermite cumulant are NOT included.
    stubs = tuple((vertex, j) for vertex, count in enumerate(degree) for j in range(count))
    count = 0
    for pairing in pairings(stubs):
        induced = tuple(sorted(tuple(sorted((a[0], b[0]))) for a, b in pairing))
        if induced == edges:
            count += 1
    denominator = math.prod(math.factorial(d) for d in degree)
    require(count == denominator, "Wick factorial cancellation for a labeled tree")
    return count / denominator, count


def explicit_tensor(np, C, g, d, e, tree_list):
    n = len(g)
    S = np.zeros((n, n, n, n), dtype=np.float64)
    response = {1: g, 2: d, 3: e}
    multipliers = [wick_multiplier(edges, degree)[0] for edges, degree in tree_list]
    for indices in itertools.product(range(n), repeat=4):
        terms = []
        for (edges, degree), factor in zip(tree_list, multipliers):
            coefficient = math.prod(float(response[degree[k]][indices[k]]) for k in range(4))
            bonds = math.prod(float(C[indices[a], indices[b]]) for a, b in edges)
            terms.append(factor * coefficient * bonds)
        S[indices] = math.fsum(terms)
    return S


def compare(np, records, name, actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    require(actual.shape == reference.shape and np.all(np.isfinite(actual))
            and np.all(np.isfinite(reference)), name + ": finite shape")
    error = float(np.linalg.norm((actual - reference).ravel()))
    norm = float(np.linalg.norm(reference.ravel()))
    limit = ABSOLUTE if norm <= REF_TINY else RELATIVE * norm
    item = {"name": name, "absolute_l2_error": error, "reference_l2_norm": norm,
            "relative_l2_error": error / norm if norm > 0 else None,
            "absolute_max_error": float(np.max(np.abs(actual - reference))),
            "limit": limit, "tiny_reference": norm <= REF_TINY,
            "passed": error <= limit}
    records.append(item)
    require(item["passed"], name + ": fixed algebra tolerance")


def fixture(np):
    rng = np.random.Generator(np.random.PCG64(SEED))
    A = rng.standard_normal((4, 4))
    C = A @ A.T / 4 + np.diag(np.array([0.5, 0.75, 1.0, 1.25]))
    C = (C + C.T) / 2
    W = rng.standard_normal((4, 4)) / 2
    v = np.diag(C).copy()
    sigma = np.sqrt(v)
    alpha = np.array([-0.75, -0.25, 0.5, 1.25], dtype=np.float64) / sigma
    pdf = np.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    g = np.array([0.5 * (1 + math.erf(float(a) / math.sqrt(2))) for a in alpha])
    aux = dict(gate=g, pdf=pdf, sigma=sigma, alpha=alpha, pre_v=v)
    # Distinct fixed metrics test PRE-P normalization versus actual final vv.
    # These two matrices are algebraic metric fixtures, not claimed ReLU laws.
    Cpost = C / 2 + np.diag(np.array([0.25, 0.375, 0.5, 0.625]))
    MpostP = 0.9 * Cpost + np.diag(np.array([0.03, 0.04, 0.05, 0.06]))
    vv = np.diag(W.T @ MpostP @ W).copy()
    s15 = np.sqrt(vv)
    a15 = np.array([-0.5, 0.0, 0.75, 1.5], dtype=np.float64) / s15
    aux15 = dict(alpha=a15, pdf=np.exp(-a15 * a15 / 2) / math.sqrt(2 * math.pi),
                 pre_v=vv.copy(), sigma=s15)
    return C, Cpost, W, aux, vv, aux15


def certify(np, helper, result):
    C, Cpost, W, aux, vv, aux15 = fixture(np)
    tree_list = trees()
    result["wick_counts"] = [wick_multiplier(edges, degree)[1] for edges, degree in tree_list]
    result["fixture_arrays"] = {}
    for key, array in {"C": C, "Cpost_preP": Cpost, "W": W, "vv15": vv,
                       **{"aux14_" + k: v for k, v in aux.items()},
                       **{"aux15_" + k: v for k, v in aux15.items()}}.items():
        result["fixture_arrays"][key] = {"dtype": str(array.dtype), "shape": list(array.shape),
             "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    checks = result["checks"]
    denominator = math.fsum(float(Cpost[i, i]) for i in range(4)) ** 2
    denominator += 2 * math.fsum(float(x) ** 2 for x in Cpost.ravel())
    for name in ("gaussian_three_edge", "zero_source"):
        current = dict(aux)
        if name == "zero_source":
            current["pdf"] = np.zeros(4, dtype=np.float64)
        g = current["gate"]
        d = current["pdf"] / current["sigma"]
        e = -current["alpha"] * current["pdf"] / current["pre_v"]
        S = explicit_tensor(np, C, g, d, e, tree_list)
        rays = np.array([math.fsum(float(S[i, j, k, l]) * float(W[i, a]) * float(W[j, a])
                    * float(W[k, a]) * float(W[l, a])
                    for i, j, k, l in itertools.product(range(4), repeat=4)) for a in range(4)])
        trace = math.fsum(float(S[i, i, j, j]) for i in range(4) for j in range(4))
        rho = 3 * trace / denominator
        actual_rays, actual_rho = helper["_terminal_tree_source"](C, Cpost, current, W)
        compare(np, checks, name + ": all rays", actual_rays, rays)
        compare(np, checks, name + ": double trace", actual_rho * denominator / 3, trace)
        compare(np, checks, name + ": pre-P radial coefficient", actual_rho, rho)
        # E_z[S[z,z,z,z]]=3*sum_ij S_iijj by the three auxiliary Wick pairings.
        auxiliary_wick = math.fsum(float(S[i, j, k, l]) *
            ((i == j and k == l) + (i == k and j == l) + (i == l and j == k))
            for i, j, k, l in itertools.product(range(4), repeat=4))
        compare(np, checks, name + ": auxiliary Wick trace", auxiliary_wick, 3 * trace)
        for permutation in itertools.permutations(range(4)):
            compare(np, checks, name + ": symmetry " + str(permutation), S.transpose(permutation), S)
        expected_delta = (rays - rho * vv * vv) * (aux15["alpha"] ** 2 - 1) * aux15["pdf"]
        expected_delta = expected_delta / (24 * aux15["pre_v"] * aux15["sigma"])
        actual_delta = helper["_terminal_tree_mean_delta"](actual_rays, actual_rho, vv, aux15)
        compare(np, checks, name + ": final response and actual vv", actual_delta, expected_delta)
        if name == "zero_source":
            require(np.all(actual_rays == 0) and actual_rho == 0 and np.all(actual_delta == 0),
                    "zero source must be algebraically zero")
            host = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
            require(np.maximum(host + actual_delta, 0).astype(np.float32).tobytes() ==
                    np.maximum(host, 0).astype(np.float32).tobytes(), "zero served bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--builder-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    start = time.monotonic()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL)
    require(sha(__file__) == args.source_sha256, "reviewed checker digest")
    require(sha(BUILDER) == args.builder_sha256, "reviewed builder digest")
    spec = importlib.util.spec_from_file_location("terminal_tree_source_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    pins = builder.authenticate(args.builder_sha256, args.protocol_sha256)
    pins[str(Path(__file__).resolve())] = args.source_sha256
    require(all(os.environ.get(k) == "1" for k in BLAS), "three BLAS launch variables must be one")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice10 required")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "bytecode writes disabled")
    require(peak_rss() < MAX_RSS, "initial RSS limit")
    OUT.mkdir(exist_ok=False)
    result = {"schema": "terminal_full_fourth_tree_tiny_v1", "passed": False, "completed": False,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_bindings": pins,
              "artifacts": {}, "seed": SEED, "n": 4, "trees": 16, "paths": 12, "stars": 4,
              "relative_tolerance": RELATIVE, "absolute_tolerance": ABSOLUTE,
              "tiny_reference_threshold": REF_TINY, "checks": [], "truth_accessed": False,
              "model_calls": 0, "candidate_files_created": False, "MPS_used": False,
              "algebra_only_not_score_evidence": True, "full_cumulant_claimed": False,
              "numerical_run_or_score_admitted": False}
    try:
        # Source inverse first; only helper functions are compiled numerically.
        parent = builder.PARENT.read_bytes().decode("utf-8")
        for enabled in (False, True):
            source = builder.candidate_source(parent, enabled)
            require(builder.restore(source, enabled) == parent, "source inverse")
        del source, parent
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        result["environment"] = {"python": sys.version, "numpy": np.__version__,
                                  "executable": sys.executable,
                                  "blas_launch": {k: os.environ[k] for k in BLAS}}
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            require(isinstance(pools, list) and all(p.get("num_threads") == 1 for p in pools),
                    "reported backend threads must equal one")
            result["environment"]["threadpools"] = pools
            result["environment"]["empty_inventory_means_unknown_instrumentation"] = not bool(pools)
            helper = {"fnp": np}
            exec(compile(builder.HELPERS, "<reviewed-terminal-tree-helper>", "exec"), helper)
            certify(np, helper, result)
        for path, digest in pins.items():
            require(sha(path) == digest, "source changed: " + path)
        require(time.monotonic() - start < MAX_WALL and peak_rss() < MAX_RSS, "final resource limits")
        result.update(passed=True, completed=True, hashes_unchanged=True,
                      sources_unchanged=True, exact_source_inverse=True,
                      wick_factorials_verified=True, zero_source_passed=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(wall_s=time.monotonic() - start, peak_rss_bytes=peak_rss())
        if result["wall_s"] >= MAX_WALL or result["peak_rss_bytes"] >= MAX_RSS:
            result.update(passed=False, completed=False, resource_gate_passed=False)
        else:
            result["resource_gate_passed"] = True
        write(OUT / "result.json", result)
        write(OUT / "completion.json", {"passed": result["passed"], "completed": result["completed"],
                                       "result_sha256": sha(OUT / "result.json")})
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
