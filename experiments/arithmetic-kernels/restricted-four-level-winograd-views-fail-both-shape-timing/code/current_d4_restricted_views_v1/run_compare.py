"""Fixed D4 interval-restricted view test; not an estimator or score."""

import ast
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
from importlib import metadata
import json
from math import prod
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ENV = PARENT.parent / "data_gate" / ".venv"
VERSIONS = {"numpy": "2.4.6", "flopscope": "0.12.0", "whestbench": "0.16.0", "threadpoolctl": "3.6.0"}
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
assert Path(sys.prefix).resolve() == ENV.resolve()
assert "anaconda3" not in sys.executable and "anaconda3" not in sys.prefix
assert {name: metadata.version(name) for name in VERSIONS} == VERSIONS
assert all(os.environ.get(name) == "1" for name in BLAS_ENV)

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info, threadpool_limits

SOURCE = PARENT / "unused_materializations_v1" / "estimator.py"
SOURCE_SHA = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
LAYOUT = PARENT / "packed_arena_generic.py"
LAYOUT_SHA = "4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e"
NAMES = ("_cp3_bind_arena_views", "_packed_frame", "_packed_product_cached")
SHAPES = ((1024, 832, 8192), (832, 8192, 832))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def restricted_specs(original, m, k, n, d=4):
    shapes = []
    for level in range(1, d + 1):
        shapes.extend(((7,) * level + (m // 2**level, k // 2**level),
                       (7,) * level + (k // 2**level, n // 2**level)))
    for rows, columns in ((m, k), (k, n)):
        for level in range(1, d + 1):
            shapes.append((4,) + (7,) * (level - 1)
                          + (rows // 2**level, columns // 2**level))
    shapes.append((7,) * d + (m // 2**d, n // 2**d))
    for level in range(d - 1, 0, -1):
        shapes.append((7,) * level + (m // 2**level, n // 2**level))
    for level in range(d, 0, -1):
        shapes.append((3,) + (7,) * (level - 1)
                      + (m // 2**level, n // 2**level))
    assert len(original) == len(shapes) == 6 * d
    result = []
    for (start, limit, capacity_shape), active_shape in zip(original, shapes):
        assert len(capacity_shape) == len(active_shape)
        assert all(a <= c for a, c in zip(active_shape, capacity_shape))
        stop = start + prod(active_shape)
        assert stop <= limit
        result.append((start, stop, active_shape))
    return tuple(result)


def price(m, k, n):
    q = Fraction(7, 4)
    value = 7**4 * (m // 16) * (n // 16) * (2 * k // 16 - 1)
    value += q * (m*k + k*n + m*n) * sum(q**j for j in range(4))
    assert value.denominator == 1
    return int(value)


def run():
    started = time.perf_counter()
    receipt = {
        "status": "RUNNING", "began_utc": utc(), "pid": os.getpid(),
        "executable": sys.executable, "python": sys.version,
        "sys_prefix": sys.prefix, "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "source_sha256": sha(SOURCE), "layout_sha256": sha(LAYOUT),
        "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "versions": {name: metadata.version(name) for name in VERSIONS},
        "blas_env": {name: os.environ.get(name) for name in BLAS_ENV},
        "records": [],
        "scope": "Fixed native synthetic D4 view-layout diagnostic, not complete estimator cost, raw/adjusted score or remote/official qualification.",
    }
    assert not (HERE / "result.json").exists()
    assert not (HERE / "started.json").exists()
    assert receipt["source_sha256"] == SOURCE_SHA
    assert receipt["layout_sha256"] == LAYOUT_SHA
    assert receipt["nice"] == 10
    save(HERE / "started.json", receipt)
    try:
        tree = ast.parse(SOURCE.read_text())
        nodes = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in NAMES}
        assert tuple(nodes) == NAMES
        env = {"fnp": fnp, "_PACKED_FRAME_CACHE": {}}
        for name in NAMES:
            unit = ast.Module(body=[nodes[name]], type_ignores=[])
            exec(compile(unit, str(SOURCE), "exec"), env)
        layout_env = {}
        exec(compile(LAYOUT.read_text(), str(LAYOUT), "exec"), layout_env)
        rng = np.random.default_rng(52907)
        all_timing_gates = []
        with threadpool_limits(limits=1):
            receipt["threadpools"] = threadpool_info()
            pools = [p for p in receipt["threadpools"] if p.get("user_api") == "blas"]
            assert all(p["num_threads"] == 1 for p in pools)
            receipt["thread_attestation"] = (
                "Every reported BLAS pool and environment guard is one-thread."
                if pools else "No discoverable BLAS pool; only three one-thread environment guards are attested."
            )
            for m, k, n in SHAPES:
                aa = rng.standard_normal((m, k), dtype=np.float32) / np.float32(np.sqrt(k))
                bb = rng.standard_normal((k, n), dtype=np.float32)
                capacity = (1024, 1024, 8192) if n > 1024 else (1024, 8192, 1024)
                size, specs, _ = layout_env["layout"](*capacity, 4)
                rspecs = restricted_specs(specs, m, k, n)
                descriptors = {"capacity": specs, "restricted": rspecs}
                with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=30, quiet=True) as ingress:
                    a, b = fnp.asarray(aa), fnp.asarray(bb)
                    arena = fnp.empty((size,), dtype=fnp.float32)
                    output = fnp.empty((m, n), dtype=fnp.float32)
                record = {"shape": [m, k, n], "capacity": list(capacity), "depth": 4,
                          "arena_owner_bytes": size*4, "kernel_formula_flops": price(m, k, n),
                          "ingress": ingress.summary_dict(), "setup": {}, "calls": [],
                          "descriptor_sha256": {name: hashlib.sha256(repr(value).encode()).hexdigest()
                                                for name, value in descriptors.items()}}
                receipt["records"].append(record)
                views = {}
                for name, desc in descriptors.items():
                    with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=30, quiet=True) as setup:
                        views[name] = env["_cp3_bind_arena_views"](arena, desc)
                    summary = setup.summary_dict()
                    expected = sum(prod(shape) for _, _, shape in desc)
                    assert setup.flops_used == expected
                    assert sum(op["flop_cost"] for op in summary["operations"].values()) == expected
                    assert summary["operations"]["reshape"]["calls"] == 24
                    record["setup"][name] = {"summary": summary, "predicted_flops": expected}
                first_digest = None
                sequence = (("capacity", True), ("restricted", True),
                            ("capacity", False), ("restricted", False),
                            ("restricted", False), ("capacity", False))
                for index, (name, cold) in enumerate(sequence):
                    outer_start = time.perf_counter()
                    with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=30, quiet=True) as ctx:
                        env["_packed_product_cached"](a, b, output, views[name], 4)
                    outer_wall = time.perf_counter() - outer_start
                    summary = ctx.summary_dict()
                    trial = {"index": index, "kind": name, "cold_frame": cold,
                             "outer_wall_s": outer_wall, "summary": summary}
                    record["calls"].append(trial)
                    assert sum(op["flop_cost"] for op in summary["operations"].values()) == ctx.flops_used
                    assert ctx.flops_used == price(m, k, n)
                    expected = {"add": 28, "subtract": 32, "stack": 8, "copyto": 4, "matmul": 1}
                    assert all(summary["operations"][key]["calls"] == value for key, value in expected.items())
                    array = np.frombuffer(output.tobytes(), dtype=np.float32).reshape(m, n)
                    assert str(output.dtype) == "float32" and np.isfinite(array).all()
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    if first_digest is None:
                        first_digest = digest
                    assert digest == first_digest, "Output bytes changed; no tolerance rescue."
                    trial["output_bytes_sha256"] = digest
                    print(json.dumps({"shape": [m, k, n], "kind": name, "index": index,
                                      "flops": ctx.flops_used, "outer_wall_s": outer_wall}), flush=True)
                means = {}
                for name in descriptors:
                    warm = [c for c in record["calls"] if c["kind"] == name and not c["cold_frame"]]
                    assert len(warm) == 2
                    means[name] = {"outer_wall_s": sum(c["outer_wall_s"] for c in warm)/2,
                                   "backend_wall_s": sum(c["summary"]["flopscope_backend_time_s"] for c in warm)/2}
                ratios = {key: means["restricted"][key]/means["capacity"][key]
                          for key in ("outer_wall_s", "backend_wall_s")}
                record["warm_means"] = means
                record["restricted_over_capacity"] = ratios
                record["runtime_gate_pass"] = all(value <= .90 for value in ratios.values())
                all_timing_gates.append(record["runtime_gate_pass"])
                env["_PACKED_FRAME_CACHE"].clear()
                del a, b, aa, bb, arena, output, views, array
                gc.collect()
            assert all(p["num_threads"] == 1 for p in threadpool_info() if p.get("user_api") == "blas")
        assert sha(SOURCE) == SOURCE_SHA and sha(LAYOUT) == LAYOUT_SHA
        assert sha(Path(__file__)) == receipt["runner_sha256"]
        assert sha(HERE / "PROTOCOL.md") == receipt["protocol_sha256"]
        receipt["wall_s"] = time.perf_counter() - started
        receipt["wall_scope"] = "run() wall excludes interpreter/module-import startup; includes fixtures, all contexts and byte inspection."
        receipt["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert receipt["wall_s"] <= 60 and receipt["peak_rss_bytes"] < 4_000_000_000
        receipt["numerical_and_bill_consistency_pass"] = True
        receipt["runtime_component_gate_pass"] = all(all_timing_gates)
        receipt["status"] = ("FIXED_RUNTIME_GATE_PASS_NOT_CANDIDATE_QUALIFICATION"
                             if all(all_timing_gates) else "FIXED_RUNTIME_GATE_FAILED_CLOSED_NO_AUTOMATIC_RETRY")
    except Exception:
        receipt["status"] = "FAILED_PRESERVED_NO_AUTOMATIC_RETRY"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["ended_utc"] = utc()
        save(HERE / "result.json", receipt)


if __name__ == "__main__":
    run()
