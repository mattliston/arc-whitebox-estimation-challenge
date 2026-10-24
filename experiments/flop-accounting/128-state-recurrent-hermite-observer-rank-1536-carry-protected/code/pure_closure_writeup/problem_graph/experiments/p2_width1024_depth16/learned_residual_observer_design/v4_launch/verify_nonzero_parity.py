#!/usr/bin/env python3
"""Orchestrate target-free nonzero Torch/Flopscope V4 parity.

All generated checkpoints and traces live in one fresh TemporaryDirectory
below /private/tmp and are deleted on exit.  The verifier never imports Torch,
Flopscope, a target reader, or an optimizer in its own process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
TORCH_PYTHON = Path("/Users/matt/anaconda3/bin/python")
FLOPSCOPE_PYTHON = DATA_GATE / ".venv" / "bin" / "python"
CONFIGS = {
    "primary_cp1536_h128_c256": {"cap": 1536, "hidden": 128, "cell_width": 256},
    "control_cp1792_h64_c96": {"cap": 1792, "hidden": 64, "cell_width": 96},
}
FIXTURES = (
    "identity",
    "official_seed20260823_mlp0",
    "mini_ordinal_0",
    "mini_ordinal_1",
)
EXPECTED_FROZEN = {
    V3 / "observer_model.py": "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5",
    V3 / "estimator_generalized.py": "0b18dba56b6790f21640314f5904c7cd91d165558d90bc5c265898bb632a9310",
    V3 / "estimator.py": "f8d48e5fc05d4de53173302e6fd8334c46908dd659a39baaebf2b534f5bdbea6",
    V3 / "PARITY_FIXTURES_V3.json": "8e2314f958a4d7925ccd6a8eb5b6368be36fa77135650100d05344837f163252",
    DATA_GATE / "target_blind_mini_v2.py": "6778a3216377a4a7e588df71a08fb0e182dc7a14d42ab393c2ac80200cccf0e1",
}
EXPECTED_ENVIRONMENTS = {
    "torch": {
        "python": "3.11.5",
        "numpy": "1.26.4",
        "torch": "2.7.0",
    },
    "flopscope": {
        "python": "3.11.5",
        "numpy": "2.4.6",
        "flopscope": "0.12.0",
        "whestbench": "0.16.0",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=_environment(),
    )
    return json.loads(completed.stdout)


def _versions() -> dict[str, dict[str, str]]:
    probe = (
        "import importlib.metadata as m, json, numpy, platform; "
        "d={'python':platform.python_version(),'numpy':numpy.__version__}; "
        "[(d.__setitem__(n,m.version(n))) for n in ('torch','flopscope','whestbench') "
        "if __import__('importlib').util.find_spec(n) is not None]; print(json.dumps(d))"
    )
    observed = {
        "torch": _run_json([str(TORCH_PYTHON), "-c", probe]),
        "flopscope": _run_json([str(FLOPSCOPE_PYTHON), "-c", probe]),
    }
    if observed != EXPECTED_ENVIRONMENTS:
        raise AssertionError(f"parity environment changed: {observed!r}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        parent = args.output.parent.resolve().as_posix()
        if args.output.exists() or not (
            parent == "/private/tmp" or parent.startswith("/private/tmp/")
        ):
            raise SystemExit("optional receipt must be a new path below /private/tmp")
    for path, expected in EXPECTED_FROZEN.items():
        if sha256_file(path) != expected:
            raise AssertionError(f"frozen V3 parity dependency changed: {path.name}")
    versions = _versions()
    runs: dict[str, Any] = {}
    fixture_bindings: dict[str, tuple[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="p2-observer-v4-parity-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        for config_name in args.configs:
            config = CONFIGS[config_name]
            bundle = root / f"{config_name}-bundle"
            export = _run_json(
                [
                    str(TORCH_PYTHON), str(HERE / "torch_nonzero_worker.py"), "export",
                    "--bundle", str(bundle), "--hidden", str(config["hidden"]),
                    "--cell-width", str(config["cell_width"]),
                ]
            )
            if export.get("status") != "PASS_CLOSED_NONZERO_TORCH_CHECKPOINT_EXPORT":
                raise AssertionError("closed nonzero export failed")
            config_runs: dict[str, Any] = {}
            for fixture in args.fixtures:
                trace_dir = root / f"{config_name}-{fixture}"
                flops = _run_json(
                    [
                        str(FLOPSCOPE_PYTHON), str(HERE / "flopscope_nonzero_worker.py"),
                        "--fixture", fixture, "--cap", str(config["cap"]),
                        "--bundle", str(bundle), "--output", str(trace_dir),
                    ]
                )
                if flops.get("status") != "PASS_TARGET_FREE_FLOPSCOPE_NONZERO_TRACE_WORKER":
                    raise AssertionError("Flopscope parity worker failed")
                binding = (
                    flops["fixture"]["weights_raw_sha256"],
                    flops["fixture"]["weights_aggregate_sha256"],
                )
                if fixture in fixture_bindings and fixture_bindings[fixture] != binding:
                    raise AssertionError("two host caps consumed different weight bytes")
                fixture_bindings[fixture] = binding
                comparisons: dict[str, Any] = {}
                for arm in ("local_no_recurrence", "recurrent"):
                    comparison = _run_json(
                        [
                            str(TORCH_PYTHON), str(HERE / "torch_nonzero_worker.py"), "compare",
                            "--bundle", str(bundle), "--trace-dir", str(trace_dir),
                            "--arm", arm,
                        ]
                    )
                    if comparison.get("status") != "PASS_TARGET_FREE_TORCH_FLOPSCOPE_INTERNAL_STATE_PARITY":
                        raise AssertionError("clean internal-state parity failed")
                    comparisons[arm] = comparison
                fault = _run_json(
                    [
                        str(TORCH_PYTHON), str(HERE / "torch_nonzero_worker.py"), "compare",
                        "--bundle", str(bundle), "--trace-dir", str(trace_dir),
                        "--arm", "recurrent", "--fault-layer0", "2",
                    ]
                )
                if fault.get("latch") != [True, True] + [False] * 14:
                    raise AssertionError("permanent latch-off replay failed")
                config_runs[fixture] = {
                    "fixture": flops["fixture"],
                    "host": flops["host"],
                    "flops_used_host_plus_three_observer_traces": flops[
                        "flops_used_host_plus_three_observer_traces"
                    ],
                    "comparisons": comparisons,
                    "fault": fault,
                }
            runs[config_name] = {
                "config": config,
                "checkpoint": {
                    "manifest_sha256": export["manifest_sha256"],
                    "tensor_bundle_sha256": export["torch_tensor_bundle_sha256"],
                    "parameter_order": export["parameter_order"],
                },
                "fixtures": config_runs,
            }
    report = {
        "schema": "phase2-observer-v4-target-free-nonzero-parity-receipt-v1",
        "status": "PASS_TARGET_FREE_NONZERO_CONVERTER_INTERNAL_STATE_CP_LATCH_PARITY",
        "environments": versions,
        "frozen_dependencies": {str(path.relative_to(PHASE2)): digest for path, digest in EXPECTED_FROZEN.items()},
        "runs": runs,
        "role_parquet_files_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "mps_used": False,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    print(payload, end="")


if __name__ == "__main__":
    main()
