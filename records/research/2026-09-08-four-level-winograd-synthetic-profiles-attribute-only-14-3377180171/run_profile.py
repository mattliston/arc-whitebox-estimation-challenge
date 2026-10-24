"""Fixed operation-category profile of the UNCHANGED qualified D4 helper."""

import ast
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
from importlib import metadata
import json
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
LAYOUT_SHA = "4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e"
NAMES = ("_cp3_bind_arena_views", "_packed_frame", "_packed_product_cached")
SHAPES = ((1024, 832, 8192), (832, 8192, 832))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_exclusive(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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
        "source_sha256": sha(SOURCE), "runner_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "versions": {name: metadata.version(name) for name in VERSIONS},
        "blas_env": {name: os.environ.get(name) for name in BLAS_ENV},
        "sys_prefix": sys.prefix, "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "records": [],
        "scope": "Unchanged D4 synthetic native kernel profile; not complete estimator cost, raw score, remote or official qualification.",
    }
    assert not (HERE / "result.json").exists()
    assert not (HERE / "started.json").exists()
    assert receipt["source_sha256"] == SOURCE_SHA
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10
    save_exclusive(HERE / "started.json", receipt)
    try:
        source = SOURCE.read_text()
        tree = ast.parse(source)
        nodes = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in NAMES}
        assert tuple(nodes) == NAMES
        env = {"fnp": fnp, "_PACKED_FRAME_CACHE": {}}
        for name in NAMES:
            # Compile the original AST node; no body substitution or instrumentation.
            unit = ast.Module(body=[nodes[name]], type_ignores=[])
            exec(compile(unit, str(SOURCE), "exec"), env)
        layout_path = PARENT / "packed_arena_generic.py"
        receipt["layout_sha256"] = sha(layout_path)
        assert receipt["layout_sha256"] == LAYOUT_SHA
        layout_env = {}
        exec(compile(layout_path.read_text(), str(layout_path), "exec"), layout_env)
        layout = layout_env["layout"]
        rng = np.random.default_rng(52907)
        with threadpool_limits(limits=1):
            receipt["threadpools"] = threadpool_info()
            blas = [p for p in receipt["threadpools"] if p.get("user_api") == "blas"]
            assert all(p["num_threads"] == 1 for p in blas)
            receipt["thread_attestation"] = (
                "Every reported BLAS pool is one-thread; environment guards are one."
                if blas else
                "No discoverable BLAS pool: only the three one-thread environment guards are attested, as in qualified OFFPY native receipts."
            )
            for m, k, n in SHAPES:
                aa = rng.standard_normal((m, k), dtype=np.float32) / np.float32(np.sqrt(k))
                bb = rng.standard_normal((k, n), dtype=np.float32)
                capacity = (1024, 1024, 8192) if n > 1024 else (1024, 8192, 1024)
                size, specs, _ = layout(*capacity, 4)
                with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=30, quiet=True) as setup:
                    a, b = fnp.asarray(aa), fnp.asarray(bb)
                    arena = fnp.empty((size,), dtype=fnp.float32)
                    views = env["_cp3_bind_arena_views"](arena, specs)
                    output = fnp.empty((m, n), dtype=fnp.float32)
                record = {
                    "shape": [m, k, n], "capacity": list(capacity), "depth": 4,
                    "arena_bytes": size*4, "warm_formula_flops": price(m, k, n),
                    "setup": setup.summary_dict(), "calls": [],
                }
                receipt["records"].append(record)
                first_digest = None
                for call in range(3):
                    outer_start = time.perf_counter()
                    with flops.BudgetContext(flop_budget=2**38, wall_time_limit_s=30, quiet=True) as ctx:
                        env["_packed_product_cached"](a, b, output, views, 4)
                    outer_wall = time.perf_counter() - outer_start
                    summary = ctx.summary_dict()
                    trial = {"call": call, "cold_frame": call == 0,
                             "outer_wall_s": outer_wall, "summary": summary}
                    record["calls"].append(trial)
                    total = sum(op["flop_cost"] for op in summary["operations"].values())
                    assert total == ctx.flops_used
                    # Read only after the timing context, with an explicit base-array
                    # materialization. No ndarray-subclass serialization is used.
                    array = np.frombuffer(output.tobytes(), dtype=np.float32).reshape(m, n)
                    assert str(output.dtype) == "float32" and np.isfinite(array).all()
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    if first_digest is None:
                        first_digest = digest
                    assert digest == first_digest
                    if call:
                        assert ctx.flops_used == price(m, k, n)
                    ops = summary["operations"]
                    expected = {"add": 28, "subtract": 32, "stack": 8, "copyto": 4, "matmul": 1}
                    assert all(ops[key]["calls"] == value for key, value in expected.items())
                    trial["output_bytes_sha256"] = digest
                    print(json.dumps({"shape": [m, k, n], "call": call,
                                      "flops": ctx.flops_used, "outer_wall_s": outer_wall}), flush=True)
                env["_PACKED_FRAME_CACHE"].clear()
                del a, b, aa, bb, arena, views, output, array
                gc.collect()
            assert all(p["num_threads"] == 1 for p in threadpool_info()
                       if p.get("user_api") == "blas")
        assert sha(SOURCE) == SOURCE_SHA
        assert sha(layout_path) == LAYOUT_SHA
        assert sha(Path(__file__)) == receipt["runner_sha256"]
        assert sha(HERE / "PROTOCOL.md") == receipt["protocol_sha256"]
        receipt["wall_s"] = time.perf_counter() - started
        receipt["wall_scope"] = "run() wall; excludes Python/module-import startup, includes fixture generation, setup, kernels and output inspection."
        receipt["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert receipt["wall_s"] < 60
        assert receipt["peak_rss_bytes"] < 4_000_000_000
        receipt["status"] = "PROFILE_CONSISTENCY_PASS_NOT_CANDIDATE_QUALIFICATION"
    except Exception:
        receipt["status"] = "FAILED_PRESERVED_NO_AUTOMATIC_RETRY"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["ended_utc"] = utc()
        save_exclusive(HERE / "result.json", receipt)


if __name__ == "__main__":
    run()
