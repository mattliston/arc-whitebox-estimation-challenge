#!/usr/bin/env python3
"""Run a target-blind Phase 2 whest/flopscope meter and retain provenance.

This driver deliberately has no dataset argument.  It invokes the official
dataset-less ``whest run`` path, whose synthetic weights and Monte-Carlo truth
are both determined by ``--seed``.  It then recreates that fixture through the
installed whestbench API and stores hashes (not target values) in the receipt.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shlex
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WIDTH = 1024
DEPTH = 16
FLOP_BUDGET = 2**41
SCORE_FLOOR_NUMERATOR = 1
SCORE_FLOOR_DENOMINATOR = 10
SETUP_TIMEOUT_S = 5.0
PREDICT_RESPONSE_TIMEOUT_S = 30.0
WALL_TIME_LIMIT_S = 120.0
RESIDUAL_WALL_TIME_LIMIT_S = 0.4
MEMORY_LIMIT_MB = 8192
LAMBDA_FLOPS_PER_SECOND = 0.0
MANAGED_FILENAMES = (
    "invocation.json",
    "raw_whest_stdout.json",
    "raw_whest_stderr.txt",
    "raw_process_tree_rss.jsonl",
    "receipt.json",
)
RSS_SAMPLE_INTERVAL_S = 0.01


def _workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("could not locate workspace root above meter driver")


def _phase2_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


class _ProcessTreeRSSSampler:
    """Sample the live RSS of a subprocess and every descendant.

    macOS uses libproc directly because the workspace sandbox denies ``ps``.
    Linux uses procfs.  Every sample is retained as JSONL, so the summary peak
    can be independently recomputed instead of being a bare assertion.
    """

    def __init__(self, root_pid: int, *, interval_s: float) -> None:
        self.root_pid = int(root_pid)
        self.interval_s = float(interval_s)
        self.started_monotonic = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="phase2-process-tree-rss-sampler",
        )
        self._path_cache: dict[int, str | None] = {}
        if sys.platform == "darwin":
            self.backend = "macos-libproc-proc_pidinfo"
            libproc_path = ctypes.util.find_library("proc")
            if libproc_path is None:
                raise RuntimeError("macOS libproc was not found")
            self._libproc = ctypes.CDLL(libproc_path)
            self._libproc.proc_listchildpids.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._libproc.proc_listchildpids.restype = ctypes.c_int
            self._libproc.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            self._libproc.proc_pidinfo.restype = ctypes.c_int
            self._libproc.proc_pidpath.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            self._libproc.proc_pidpath.restype = ctypes.c_int
        elif sys.platform.startswith("linux"):
            self.backend = "linux-procfs"
            self._libproc = None
        else:
            raise RuntimeError(f"unsupported RSS sampling platform: {sys.platform}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, 10.0 * self.interval_s))
        if self._thread.is_alive():
            self.errors.append("sampler thread did not stop within its join timeout")

    def _macos_children(self, pid: int) -> list[int]:
        capacity = 4096
        buffer = (ctypes.c_int * capacity)()
        count = int(
            self._libproc.proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
        )
        if count < 0:
            return []
        if count >= capacity:
            self.errors.append(f"child pid buffer saturated for pid {pid}")
        return [int(value) for value in buffer[: min(count, capacity)] if value > 0]

    def _macos_rss(self, pid: int) -> int | None:
        # PROC_PIDTASKINFO=4. proc_taskinfo starts with virtual_size (uint64)
        # then resident_size (uint64), so RSS is the second field at offset 8.
        buffer = ctypes.create_string_buffer(256)
        written = int(self._libproc.proc_pidinfo(pid, 4, 0, buffer, len(buffer)))
        if written < 16:
            return None
        return int(struct.unpack_from("=Q", buffer.raw, 8)[0])

    def _macos_path(self, pid: int) -> str | None:
        buffer = ctypes.create_string_buffer(4096)
        written = int(self._libproc.proc_pidpath(pid, buffer, len(buffer)))
        return os.fsdecode(buffer.value) if written > 0 else None

    @staticmethod
    def _linux_children(pid: int) -> list[int]:
        try:
            payload = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return []
        return [int(value) for value in payload.split()] if payload else []

    @staticmethod
    def _linux_rss(pid: int) -> int | None:
        try:
            fields = Path(f"/proc/{pid}/statm").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        if len(fields) < 2:
            return None
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))

    @staticmethod
    def _linux_path(pid: int) -> str | None:
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None

    def _children(self, pid: int) -> list[int]:
        return self._macos_children(pid) if self._libproc is not None else self._linux_children(pid)

    def _rss(self, pid: int) -> int | None:
        return self._macos_rss(pid) if self._libproc is not None else self._linux_rss(pid)

    def _path(self, pid: int) -> str | None:
        if pid not in self._path_cache:
            self._path_cache[pid] = (
                self._macos_path(pid) if self._libproc is not None else self._linux_path(pid)
            )
        return self._path_cache[pid]

    def _snapshot(self) -> dict[str, Any] | None:
        queue: list[tuple[int, int, int | None]] = [(self.root_pid, 0, None)]
        seen: set[int] = set()
        processes: list[dict[str, Any]] = []
        while queue:
            pid, depth, parent_pid = queue.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            rss = self._rss(pid)
            if rss is not None:
                processes.append(
                    {
                        "pid": pid,
                        "parent_pid": parent_pid,
                        "depth": depth,
                        "rss_bytes": rss,
                        "executable": self._path(pid),
                    }
                )
            for child_pid in self._children(pid):
                queue.append((child_pid, depth + 1, pid))
        if not processes:
            return None
        root_rss = sum(record["rss_bytes"] for record in processes if record["depth"] == 0)
        direct_rss = sum(record["rss_bytes"] for record in processes if record["depth"] == 1)
        descendant_rss = sum(record["rss_bytes"] for record in processes if record["depth"] > 0)
        return {
            "elapsed_s": time.monotonic() - self.started_monotonic,
            "root_rss_bytes": root_rss,
            "direct_children_aggregate_rss_bytes": direct_rss,
            "descendants_aggregate_rss_bytes": descendant_rss,
            "tree_aggregate_rss_bytes": root_rss + descendant_rss,
            "processes": processes,
        }

    def _run(self) -> None:
        while True:
            try:
                snapshot = self._snapshot()
                if snapshot is not None:
                    self.samples.append(snapshot)
            except Exception as exc:  # noqa: BLE001 - retain sampler failure as evidence
                if len(self.errors) < 20:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            if self._stop_event.wait(self.interval_s):
                break

    def raw_jsonl(self) -> bytes:
        return b"".join(
            (json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for sample in self.samples
        )

    def summary(self) -> dict[str, Any]:
        def peak(field: str) -> int:
            return max((int(sample[field]) for sample in self.samples), default=0)

        peak_descendant_sample = max(
            self.samples,
            key=lambda sample: int(sample["descendants_aggregate_rss_bytes"]),
            default=None,
        )
        peak_tree_sample = max(
            self.samples,
            key=lambda sample: int(sample["tree_aggregate_rss_bytes"]),
            default=None,
        )
        observed_pids = sorted(
            {int(process["pid"]) for sample in self.samples for process in sample["processes"]}
        )
        return {
            "backend": self.backend,
            "root_pid": self.root_pid,
            "requested_sample_interval_s": self.interval_s,
            "sample_count": len(self.samples),
            "sampling_errors": self.errors,
            "observed_pids": observed_pids,
            "peak_root_rss_bytes": peak("root_rss_bytes"),
            "peak_direct_children_aggregate_rss_bytes": peak(
                "direct_children_aggregate_rss_bytes"
            ),
            "peak_descendants_aggregate_rss_bytes": peak("descendants_aggregate_rss_bytes"),
            "peak_tree_aggregate_rss_bytes": peak("tree_aggregate_rss_bytes"),
            "peak_descendant_snapshot": peak_descendant_sample,
            "peak_tree_snapshot": peak_tree_sample,
        }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _distribution_fingerprint(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for relative in sorted(distribution.files or (), key=str):
        if str(relative).endswith((".pyc", ".pyo")):
            continue
        resolved = Path(distribution.locate_file(relative))
        if not resolved.is_file():
            continue
        payload = resolved.read_bytes()
        encoded_name = str(relative).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        file_count += 1
        total_bytes += len(payload)
    return {
        "version": distribution.version,
        "installed_distribution_sha256": digest.hexdigest(),
        "hashed_file_count": file_count,
        "hashed_bytes": total_bytes,
    }


def _installed_source_hashes() -> dict[str, str]:
    site_root = Path(importlib.metadata.distribution("whestbench").locate_file(""))
    relative_paths = (
        "whestbench/cli.py",
        "whestbench/generation.py",
        "whestbench/scoring.py",
        "whestbench/simulation.py",
        "whestbench/seeds.py",
        "flopscope/__init__.py",
        "flopscope/accounting.py",
        "flopscope/numpy/__init__.py",
        "flopscope/_einsum.py",
        "flopscope/_symmetric.py",
    )
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        source = site_root / relative
        if not source.is_file():
            raise RuntimeError(f"required installed source is missing: {source}")
        hashes[relative] = _sha256_file(source)
    return hashes


def _array_digest(array: Any) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return {
        "sha256": digest.hexdigest(),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "nbytes": int(contiguous.nbytes),
    }


def _combine_named_digests(items: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("ascii")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(value_bytes)
    return digest.hexdigest()


def _synthetic_fixture_fingerprint(
    *, seed: int, n_mlps: int, n_samples: int, max_threads: int
) -> dict[str, Any]:
    """Recreate whest's deterministic synthetic fixture and hash its arrays."""

    import numpy as np
    from threadpoolctl import threadpool_limits
    from whestbench.scoring import ContestSpec, make_contest
    from whestbench.simulation import _pick_chunk_size

    spec = ContestSpec(
        width=WIDTH,
        depth=DEPTH,
        n_mlps=n_mlps,
        flop_budget=FLOP_BUDGET,
        ground_truth_samples=n_samples,
        setup_timeout_s=SETUP_TIMEOUT_S,
        predict_timeout_s=PREDICT_RESPONSE_TIMEOUT_S,
        memory_limit_mb=MEMORY_LIMIT_MB,
        wall_time_limit_s=WALL_TIME_LIMIT_S,
        residual_wall_time_limit_s=RESIDUAL_WALL_TIME_LIMIT_S,
        seed=seed,
        lambda_flops_per_second=LAMBDA_FLOPS_PER_SECOND,
    )
    with threadpool_limits(limits=max_threads):
        contest = make_contest(spec)

    streams = np.random.SeedSequence(seed).spawn(3 * n_mlps)
    scale = float(np.sqrt(2.0 / WIDTH))
    chunk_size = int(_pick_chunk_size(WIDTH))
    per_mlp: list[dict[str, Any]] = []
    aggregate_weight_digests: list[tuple[str, str]] = []
    aggregate_input_digests: list[tuple[str, str]] = []
    aggregate_truth_digests: list[tuple[str, str]] = []

    for mlp_index, mlp in enumerate(contest.mlps):
        weight_rng = np.random.default_rng(streams[3 * mlp_index])
        layer_records: list[dict[str, Any]] = []
        exact_weight_reconstruction = True
        for layer_index, official_weight in enumerate(mlp.weights):
            official_array = np.asarray(official_weight)
            recreated = (weight_rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
            exact_weight_reconstruction &= bool(np.array_equal(official_array, recreated))
            layer_digest = _array_digest(official_array)
            layer_records.append({"layer": layer_index, **layer_digest})
            aggregate_weight_digests.append(
                (f"mlp{mlp_index}/layer{layer_index}", layer_digest["sha256"])
            )

        sample_rng = np.random.default_rng(streams[3 * mlp_index + 1])
        sample_digest = hashlib.sha256()
        sample_digest.update(np.dtype(np.float32).str.encode("ascii"))
        sample_digest.update(
            json.dumps([n_samples, WIDTH], separators=(",", ":")).encode("ascii")
        )
        for start in range(0, n_samples, chunk_size):
            count = min(chunk_size, n_samples - start)
            sample_chunk = sample_rng.standard_normal((count, WIDTH), dtype=np.float32)
            sample_digest.update(memoryview(np.ascontiguousarray(sample_chunk)).cast("B"))
        input_sha256 = sample_digest.hexdigest()
        aggregate_input_digests.append((f"mlp{mlp_index}", input_sha256))

        all_layer_truth = _array_digest(contest.all_layer_targets[mlp_index])
        final_truth = _array_digest(contest.final_targets[mlp_index])
        aggregate_truth_digests.append(
            (f"mlp{mlp_index}/all_layers", all_layer_truth["sha256"])
        )
        aggregate_truth_digests.append((f"mlp{mlp_index}/final", final_truth["sha256"]))
        per_mlp.append(
            {
                "mlp_index": mlp_index,
                "mlp_name": mlp.name,
                "estimator_seed": int(mlp.seed),
                "weight_layers": layer_records,
                "exact_numpy_weight_reconstruction": exact_weight_reconstruction,
                "sampled_input_sha256": input_sha256,
                "sampled_input_dtype": "float32",
                "sampled_input_shape": [n_samples, WIDTH],
                "all_layer_truth": all_layer_truth,
                "final_layer_truth": final_truth,
                "average_final_layer_variance": float(contest.avg_variances[mlp_index]),
            }
        )

    return {
        "authority": "installed whestbench.scoring.make_contest",
        "dataset_argument": None,
        "public_target_values_accessed": 0,
        "seed": seed,
        "seed_tree": "numpy.random.SeedSequence(seed).spawn(3 * n_mlps)",
        "stream_assignment": ["weights", "synthetic_truth_inputs", "estimator_seed"],
        "weight_generation": "float32(N(0,1) float64 * sqrt(2/width))",
        "truth_input_generation": "numpy Generator.standard_normal(..., dtype=float32)",
        "truth_forward_pass": "x = maximum(x @ W, 0) after each of 16 matrices",
        "truth_reduction": "per-layer float64 sums divided by n_samples, cast to float32",
        "width": WIDTH,
        "depth": DEPTH,
        "n_mlps": n_mlps,
        "truth_samples": n_samples,
        "truth_chunk_size": chunk_size,
        "max_blas_threads": max_threads,
        "aggregate_weights_sha256": _combine_named_digests(aggregate_weight_digests),
        "aggregate_sampled_inputs_sha256": _combine_named_digests(aggregate_input_digests),
        "aggregate_truth_sha256": _combine_named_digests(aggregate_truth_digests),
        "all_weights_reconstruct_exactly": all(
            record["exact_numpy_weight_reconstruction"] for record in per_mlp
        ),
        "per_mlp": per_mlp,
    }


def _operation_totals(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}
    estimator_breakdown = report["results"].get("breakdowns", {}).get("estimator", {})
    for namespace in estimator_breakdown.get("by_namespace", {}).values():
        for name, record in namespace.get("operations", {}).items():
            destination = totals.setdefault(name, {"flop_cost": 0, "calls": 0})
            destination["flop_cost"] = int(destination["flop_cost"]) + int(
                record.get("flop_cost", 0)
            )
            destination["calls"] = int(destination["calls"]) + int(record.get("calls", 0))
    return dict(sorted(totals.items()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True, help="Path to estimator.py")
    parser.add_argument("--class", dest="class_name", required=True, help="Estimator class")
    parser.add_argument("--label", required=True, help="Stable output directory label")
    parser.add_argument("--n-mlps", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--process-timeout", type=float, default=300.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four managed artifacts in an existing label directory",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.label):
        raise SystemExit("--label must be a 1-128 character filename-safe token")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.class_name):
        raise SystemExit("--class must be a Python identifier")
    if args.n_mlps <= 0 or args.n_samples <= 0 or args.max_threads <= 0:
        raise SystemExit("--n-mlps, --n-samples, and --max-threads must be positive")

    workspace = _workspace_root()
    phase2_root = _phase2_root()
    estimator = Path(args.estimator)
    if not estimator.is_absolute():
        estimator = (workspace / estimator).resolve()
    if not estimator.is_file():
        raise SystemExit(f"estimator does not exist: {estimator}")

    expected_venv = phase2_root / "data_gate" / ".venv"
    whest = expected_venv / "bin" / "whest"
    if not whest.is_file():
        raise SystemExit(f"isolated whest executable is missing: {whest}")
    if Path(sys.prefix).resolve() != expected_venv.resolve():
        raise SystemExit(
            "run this driver with the isolated interpreter: "
            f"{expected_venv / 'bin' / 'python'}"
        )

    output_dir = Path(__file__).resolve().parent / "runs" / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in MANAGED_FILENAMES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        paths = ", ".join(path.name for path in existing)
        raise SystemExit(f"managed artifacts already exist ({paths}); pass --overwrite explicitly")

    estimator_argument = _relative_or_absolute(estimator, workspace)
    # Run the installed console-script body with the isolated interpreter
    # explicitly.  The generated macOS console script is a /bin/sh/Python
    # polyglot; bypassing that transient shell makes the sampled root exactly
    # the whest evaluator and its descendants exactly the solution process tree.
    whest_argv = [
        str(expected_venv / "bin" / "python"),
        str(whest),
        "run",
        "--estimator",
        estimator_argument,
        "--class",
        args.class_name,
        "--runner",
        "subprocess",
        "--n-mlps",
        str(args.n_mlps),
        "--n-samples",
        str(args.n_samples),
        "--seed",
        str(args.seed),
        "--max-threads",
        str(args.max_threads),
        "--flop-budget",
        str(FLOP_BUDGET),
        "--lambda-flops-per-second",
        str(LAMBDA_FLOPS_PER_SECOND),
        "--setup-timeout",
        str(SETUP_TIMEOUT_S),
        "--wall-time-limit",
        str(WALL_TIME_LIMIT_S),
        "--residual-wall-time-limit",
        str(RESIDUAL_WALL_TIME_LIMIT_S),
        "--profile",
        "--detail",
        "full",
        "--format",
        "json",
    ]
    deterministic_env = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(args.max_threads),
        "OPENBLAS_NUM_THREADS": str(args.max_threads),
        "MKL_NUM_THREADS": str(args.max_threads),
        "VECLIB_MAXIMUM_THREADS": str(args.max_threads),
    }
    child_env = os.environ.copy()
    child_env.update(deterministic_env)

    requirements_lock = phase2_root / "data_gate" / "requirements.lock.txt"
    invocation = {
        "schema": "phase2-target-blind-synthetic-meter-invocation-v1",
        "driver_argv": list(sys.argv),
        "driver_shell_command": shlex.join([sys.executable, *sys.argv]),
        "whest_argv": whest_argv,
        "whest_shell_command": shlex.join(whest_argv),
        "cli_launch": "isolated Python interpreter executing installed whest console script",
        "cwd": str(workspace),
        "environment_overrides": deterministic_env,
        "external_process_timeout_s": args.process_timeout,
        "dataset_argument": None,
        "memory_sampling": {
            "scope": "whest CLI root process plus every observed descendant",
            "requested_interval_s": RSS_SAMPLE_INTERVAL_S,
            "raw_samples_retained": "raw_process_tree_rss.jsonl",
        },
        "config": {
            "width": WIDTH,
            "depth": DEPTH,
            "n_mlps": args.n_mlps,
            "truth_samples": args.n_samples,
            "seed": args.seed,
            "flop_budget": FLOP_BUDGET,
            "lambda_flops_per_second": LAMBDA_FLOPS_PER_SECOND,
            "setup_timeout_s": SETUP_TIMEOUT_S,
            "predict_response_timeout_s_implicit_default": PREDICT_RESPONSE_TIMEOUT_S,
            "wall_time_limit_s": WALL_TIME_LIMIT_S,
            "residual_wall_time_limit_s": RESIDUAL_WALL_TIME_LIMIT_S,
            "memory_limit_mb_implicit_default": MEMORY_LIMIT_MB,
            "cpu_time_limit_s_implicit_default": None,
            "runner": "subprocess",
            "profile": True,
            "detail": "full",
            "format": "json",
        },
    }
    _write_json(output_dir / "invocation.json", invocation)

    started_at = datetime.now(timezone.utc)
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    process = subprocess.Popen(
        whest_argv,
        cwd=workspace,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rss_sampler = _ProcessTreeRSSSampler(process.pid, interval_s=RSS_SAMPLE_INTERVAL_S)
    rss_sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=args.process_timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout, stderr = process.communicate()
        rss_sampler.stop()
        _atomic_write(output_dir / "raw_whest_stdout.json", stdout)
        _atomic_write(output_dir / "raw_whest_stderr.txt", stderr)
        _atomic_write(output_dir / "raw_process_tree_rss.jsonl", rss_sampler.raw_jsonl())
        raise SystemExit(f"whest process exceeded {args.process_timeout}s") from exc
    rss_sampler.stop()
    completed = subprocess.CompletedProcess(
        args=whest_argv,
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
    )
    finished_at = datetime.now(timezone.utc)
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    rusage_unit = "bytes" if sys.platform == "darwin" else "KiB"
    rusage_children_peak_bytes = int(
        rusage_after if sys.platform == "darwin" else rusage_after * 1024
    )
    _atomic_write(output_dir / "raw_whest_stdout.json", completed.stdout)
    _atomic_write(output_dir / "raw_whest_stderr.txt", completed.stderr)
    _atomic_write(output_dir / "raw_process_tree_rss.jsonl", rss_sampler.raw_jsonl())
    if completed.returncode != 0:
        raise SystemExit(
            f"whest returned {completed.returncode}; raw stdout/stderr retained in {output_dir}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"whest stdout was not JSON; raw bytes retained in {output_dir}") from exc

    run_config = report.get("run_config", {})
    results = report.get("results", {})
    per_mlp = results.get("per_mlp", [])
    fixture = _synthetic_fixture_fingerprint(
        seed=args.seed,
        n_mlps=args.n_mlps,
        n_samples=args.n_samples,
        max_threads=args.max_threads,
    )
    expected_config = {
        "width": WIDTH,
        "depth": DEPTH,
        "n_mlps": args.n_mlps,
        "seed": args.seed,
        "flop_budget": FLOP_BUDGET,
        "setup_timeout_s": SETUP_TIMEOUT_S,
        "wall_time_limit_s": WALL_TIME_LIMIT_S,
        "residual_wall_time_limit_s": RESIDUAL_WALL_TIME_LIMIT_S,
        "lambda_flops_per_second": LAMBDA_FLOPS_PER_SECOND,
    }
    config_matches = all(run_config.get(key) == value for key, value in expected_config.items())
    names_match = [record.get("mlp_name") for record in per_mlp] == [
        record["mlp_name"] for record in fixture["per_mlp"]
    ]
    no_failures = int(results.get("n_failed_mlps", -1)) == 0
    software = {
        name: _distribution_fingerprint(name)
        for name in ("numpy", "flopscope", "flopscope-server", "whestbench")
    }
    raw_host = report.get("run_meta", {}).get("host", {})
    rss_summary = rss_sampler.summary()
    memory_limit_bytes = MEMORY_LIMIT_MB * 1024 * 1024
    sampled_descendant_peak_bytes = int(rss_summary["peak_descendants_aggregate_rss_bytes"])
    assertions = {
        "whest_returncode_zero": completed.returncode == 0,
        "run_config_matches_invocation": config_matches,
        "one_result_per_synthetic_mlp": len(per_mlp) == args.n_mlps,
        "synthetic_mlp_names_match_recreated_fixture": names_match,
        "all_weights_reconstruct_exactly": fixture["all_weights_reconstruct_exactly"],
        "no_failed_mlps": no_failures,
        "no_dataset_argument": "--dataset" not in whest_argv,
        "no_public_target_values_accessed": fixture["public_target_values_accessed"] == 0,
        "raw_whestbench_version_matches_installed": report.get("whestbench_version")
        == software["whestbench"]["version"],
        "raw_numpy_version_matches_installed": raw_host.get("numpy_version")
        == software["numpy"]["version"],
        "raw_flopscope_version_matches_installed": raw_host.get("flopscope_version")
        == software["flopscope"]["version"],
        "process_tree_rss_samples_retained": int(rss_summary["sample_count"]) > 0,
        "solution_descendant_rss_observed": sampled_descendant_peak_bytes > 0,
        "process_tree_rss_sampler_error_free": not rss_summary["sampling_errors"],
        "sampled_solution_descendant_rss_below_memory_limit": sampled_descendant_peak_bytes
        < memory_limit_bytes,
    }
    passed = all(assertions.values())

    estimator_flops = [int(record.get("flops_used", 0)) for record in per_mlp]
    score_floor_max_flops = (
        FLOP_BUDGET * SCORE_FLOOR_NUMERATOR // SCORE_FLOOR_DENOMINATOR
    )
    floor_classification = [
        {
            "mlp_index": index,
            "flops_used": flops_used,
            "within_0p1_score_floor": (
                SCORE_FLOOR_DENOMINATOR * flops_used
                <= SCORE_FLOOR_NUMERATOR * FLOP_BUDGET
            ),
            "integer_flop_margin_to_floor_max": score_floor_max_flops - flops_used,
        }
        for index, flops_used in enumerate(estimator_flops)
    ]
    receipt = {
        "schema": "phase2-target-blind-synthetic-meter-receipt-v1",
        "status": (
            "PASS_PHASE2_SUBPROCESS_INTERFACE_AND_METER__ACCURACY_UNSCORED"
            if passed
            else "FAIL_PHASE2_SYNTHETIC_METER_ASSERTION"
        ),
        "created_at_utc": finished_at.isoformat(),
        "elapsed_external_process_s": (finished_at - started_at).total_seconds(),
        "estimator": {
            "path": estimator_argument,
            "class": args.class_name,
            "sha256": _sha256_file(estimator),
        },
        "provenance": {
            "meter_driver_path": _relative_or_absolute(Path(__file__).resolve(), workspace),
            "meter_driver_sha256": _sha256_file(Path(__file__).resolve()),
            "receipt_verifier_path": _relative_or_absolute(
                Path(__file__).resolve().parent / "verify_meter_run.py", workspace
            ),
            "receipt_verifier_sha256": _sha256_file(
                Path(__file__).resolve().parent / "verify_meter_run.py"
            ),
            "requirements_lock_path": _relative_or_absolute(requirements_lock, workspace),
            "requirements_lock_sha256": _sha256_file(requirements_lock),
            "installed_source_sha256": _installed_source_hashes(),
        },
        "software": software,
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "invocation": invocation,
        "deterministic_synthetic_fixture": fixture,
        "raw_artifacts": {
            name: {
                "sha256": _sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in (
                "invocation.json",
                "raw_whest_stdout.json",
                "raw_whest_stderr.txt",
                "raw_process_tree_rss.jsonl",
            )
        },
        "measured": {
            "n_mlps": args.n_mlps,
            "width": WIDTH,
            "depth": DEPTH,
            "truth_samples": args.n_samples,
            "output_contract": {
                "shape": [DEPTH, WIDTH],
                "materialized_scored_dtype": "float32",
                "all_values_finite": no_failures,
                "authority": (
                    "whestbench.scoring.materialise_predictions and validate_predictions; "
                    "a shape or finiteness violation is recorded as an MLP failure"
                ),
            },
            "flops_used_per_mlp": estimator_flops,
            "mean_flops_used": (
                float(sum(estimator_flops) / len(estimator_flops)) if estimator_flops else 0.0
            ),
            "mean_effective_compute": float(results.get("mean_effective_compute", 0.0)),
            "mean_compute_utilization": float(results.get("mean_compute_utilization", 0.0)),
            "mean_score_multiplier": float(results.get("mean_score_multiplier", 0.0)),
            "score_floor": {
                "multiplier_numerator": SCORE_FLOOR_NUMERATOR,
                "multiplier_denominator": SCORE_FLOOR_DENOMINATOR,
                "budget_flops": FLOP_BUDGET,
                "exact_threshold_expression": "flops_used * 10 <= 2^41",
                "max_integer_flops_inclusive": score_floor_max_flops,
                "per_mlp": floor_classification,
                "all_mlps_within_floor": all(
                    record["within_0p1_score_floor"] for record in floor_classification
                ),
            },
            "wall_time_s_per_mlp": [float(record.get("wall_time_s", 0.0)) for record in per_mlp],
            "flopscope_backend_time_s_per_mlp": [
                float(record.get("flopscope_backend_time_s", 0.0)) for record in per_mlp
            ],
            "flopscope_overhead_time_s_per_mlp": [
                float(record.get("flopscope_overhead_time_s", 0.0)) for record in per_mlp
            ],
            "residual_wall_time_s_per_mlp": [
                float(record.get("residual_wall_time_s", 0.0)) for record in per_mlp
            ],
            "operation_totals": _operation_totals(report),
            "n_failed_mlps": int(results.get("n_failed_mlps", -1)),
            "memory": {
                "measurement": "sampled live resident set size",
                "scope": (
                    "root is the whest evaluator CLI; descendants are the isolated "
                    "whestbench solution process tree. The admission comparison uses the "
                    "peak aggregate RSS of descendants only, excluding the evaluator."
                ),
                "memory_limit_mb": MEMORY_LIMIT_MB,
                "memory_limit_bytes": memory_limit_bytes,
                "sampled_descendant_peak_below_memory_limit": sampled_descendant_peak_bytes
                < memory_limit_bytes,
                "sampler": rss_summary,
                "os_rusage_children": {
                    "ru_maxrss_before_raw": rusage_before,
                    "ru_maxrss_after_raw": rusage_after,
                    "raw_unit": rusage_unit,
                    "normalized_peak_bytes": rusage_children_peak_bytes,
                    "note": (
                        "OS child high-water cross-check; unlike raw JSONL samples, this is "
                        "not an aggregate simultaneous process-tree measurement."
                    ),
                },
                "interpretation_limit": (
                    "RSS measures resident physical pages and can miss a sub-interval spike; "
                    "it is not a measurement of virtual address space. The official "
                    "subprocess memory limit also completed without failure."
                ),
            },
        },
        "assertions": assertions,
        "scope": {
            "accuracy_scored": False,
            "accuracy_note": (
                "Dataset-free Monte-Carlo truth is a deterministic plumbing fixture, not a "
                "scientific Phase 2 accuracy measurement. Its MSE must not be compared with "
                "the public 1e9-sample targets."
            ),
            "public_phase2_target_or_truth_values_accessed": 0,
            "teacher_states_accessed": 0,
            "prior_scores_accessed": 0,
            "phase1_checkpoints_accessed": 0,
        },
    }
    _write_json(output_dir / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "run_dir": _relative_or_absolute(output_dir, workspace),
                "estimator_sha256": receipt["estimator"]["sha256"],
                "raw_whest_stdout_sha256": receipt["raw_artifacts"][
                    "raw_whest_stdout.json"
                ]["sha256"],
                "mean_flops_used": receipt["measured"]["mean_flops_used"],
                "mean_compute_utilization": receipt["measured"][
                    "mean_compute_utilization"
                ],
                "n_failed_mlps": receipt["measured"]["n_failed_mlps"],
                "all_mlps_within_0p1_score_floor": receipt["measured"]["score_floor"][
                    "all_mlps_within_floor"
                ],
                "sampled_solution_descendant_peak_rss_bytes": receipt["measured"]["memory"][
                    "sampler"
                ]["peak_descendants_aggregate_rss_bytes"],
                "sampled_solution_descendant_rss_below_8gib": receipt["measured"]["memory"][
                    "sampled_descendant_peak_below_memory_limit"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
