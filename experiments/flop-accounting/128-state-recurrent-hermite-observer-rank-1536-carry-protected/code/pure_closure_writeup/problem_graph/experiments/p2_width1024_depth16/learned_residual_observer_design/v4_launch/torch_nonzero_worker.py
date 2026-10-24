#!/usr/bin/env python3
"""Deterministic CPU Torch side of the V4 nonzero parity replay.

The export mode creates only a closed synthetic checkpoint.  The compare mode
consumes a target-free host tape emitted by the Flopscope worker and never
imports a dataset reader, target reader, optimizer, or training routine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
sys.path.insert(0, str(HERE))

from checkpoint_converter import (  # noqa: E402
    PARAMETER_ORDER,
    export_parity_bundle,
    load_numpy_bundle,
)


TOLERANCES = {
    "standardized": {"max_abs": 2.0e-6, "relative_l2": 2.0e-6},
    "messages": {"max_abs": 2.0e-4, "relative_l2": 2.0e-4},
    "log_rms": {"max_abs": 2.0e-4, "relative_l2": 2.0e-4},
    "hidden": {"max_abs": 2.0e-4, "relative_l2": 2.0e-4},
    "unit": {"max_abs": 5.0e-5, "relative_l2": 2.0e-4},
    "physical": {"max_abs": 5.0e-6, "relative_l2": 2.0e-4},
    "predictions": {"max_abs": 5.0e-6, "relative_l2": 2.0e-4, "mse": 1.0e-10},
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBSERVER = _load("v4_frozen_torch_observer", V3 / "observer_model.py")


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape:
        raise AssertionError(f"parity shape mismatch: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise AssertionError("clean parity metric received nonfinite values")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return {
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(denominator, 1.0e-30)),
        "mse": float(np.mean(difference * difference, dtype=np.float64)),
    }


def _assert_metric(name: str, metric: dict[str, float]) -> None:
    for key, ceiling in TOLERANCES[name].items():
        if metric[key] > ceiling:
            raise AssertionError(f"{name} {key}={metric[key]} exceeds {ceiling}")


def _copy_bundle(model: Any, bundle_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    arrays, centers, scales, manifest = load_numpy_bundle(bundle_dir)
    named = dict(model.named_parameters())
    if set(named) != set(PARAMETER_ORDER):
        raise AssertionError("Torch observer parameter membership changed")
    with torch.no_grad():
        for name in PARAMETER_ORDER:
            named[name].copy_(torch.from_numpy(arrays[name]))
    if OBSERVER.model_sha256(model) != manifest["torch_tensor_bundle_sha256"]:
        raise AssertionError("Torch checkpoint differs after converter roundtrip")
    return (
        torch.from_numpy(np.ascontiguousarray(centers, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(scales, dtype=np.float64)),
        manifest,
    )


def compare(
    *,
    bundle_dir: Path,
    trace_dir: Path,
    arm: str,
    fault_layer0: int | None,
) -> dict[str, Any]:
    OBSERVER.configure_cpu_determinism()
    _, _, _, bundle_manifest = load_numpy_bundle(bundle_dir)
    hidden = int(bundle_manifest["hidden_channels"])
    cell_width = int(bundle_manifest["cell_width"])
    model = OBSERVER.ResidualObserver(
        arm=arm,
        init_seed=OBSERVER.PARITY_SEED,
        parity_nonzero_mean_head=True,
        hidden_channels=hidden,
        cell_width=cell_width,
    ).cpu()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("parity model escaped the CPU authority")
    centers, scales, manifest = _copy_bundle(model, bundle_dir)
    tape_path = trace_dir / "host-tape.npz"
    reference_name = (
        "flops-recurrent-fault-l3.npz"
        if fault_layer0 is not None
        else f"flops-{arm}.npz"
    )
    with np.load(tape_path, allow_pickle=False) as tape_file:
        tape = {name: np.array(tape_file[name], copy=True) for name in tape_file.files}
    with np.load(trace_dir / reference_name, allow_pickle=False) as reference_file:
        reference = {name: np.array(reference_file[name], copy=True) for name in reference_file.files}
    worker_report = json.loads((trace_dir / "worker.json").read_text(encoding="utf-8"))
    observed_tape_hashes = {
        "raw32_sha256": _array_hash(tape["raw32"]),
        "edges_sha256": _array_hash(tape["edges"]),
        "host_sha256": _array_hash(tape["host"]),
        "sigma_sha256": _array_hash(tape["sigma"]),
    }
    for key, value in observed_tape_hashes.items():
        if worker_report["host"].get(key) != value:
            raise AssertionError(f"Torch received a changed host tape: {key}")
    entering_cp_hashes = [str(value) for value in tape["entering_cp_hashes"]]
    entering_cp_chain = hashlib.sha256(
        "\n".join(entering_cp_hashes).encode("ascii")
    ).hexdigest()
    if entering_cp_hashes != worker_report["host"].get("entering_cp_state_hashes"):
        raise AssertionError("Torch received a changed entering-CP state chain")
    if entering_cp_chain != worker_report["host"].get("entering_cp_chain_sha256"):
        raise AssertionError("entering-CP state chain digest changed")

    raw = torch.from_numpy(np.ascontiguousarray(tape["raw32"], dtype=np.float32))
    standardized = OBSERVER.standardize_raw_features(raw, centers, scales)
    if fault_layer0 is not None:
        standardized = standardized.clone()
        standardized[fault_layer0, 0, 0] = float("nan")
    trace = model(
        standardized,
        torch.from_numpy(np.ascontiguousarray(tape["edges"], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(tape["host"], dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(tape["sigma"], dtype=np.float32)),
        return_trace=True,
    )
    actual = {
        "standardized": standardized.detach().numpy(),
        "messages": trace["messages"].detach().numpy(),
        "log_rms": trace["log_rms"].detach().numpy(),
        "hidden": trace["hidden"].detach().numpy(),
        "unit": trace["unit_correction"].detach().numpy(),
        "physical": trace["physical_correction"].detach().numpy(),
        "predictions": trace["predictions"].detach().numpy(),
        "latch": np.asarray(trace["latch"], dtype=np.bool_),
    }
    if not np.array_equal(actual["latch"], reference["latch"]):
        raise AssertionError("Torch/Flopscope latch traces differ")
    if not np.array_equal(actual["physical"][0], np.zeros(1024, dtype=np.float32)):
        raise AssertionError("Torch layer-1 correction is not structurally zero")
    if not np.array_equal(reference["physical"][0], np.zeros(1024, dtype=np.float32)):
        raise AssertionError("Flopscope layer-1 correction is not structurally zero")

    metrics: dict[str, dict[str, float]] = {}
    if fault_layer0 is None:
        for name in TOLERANCES:
            metrics[name] = _metric(reference[name], actual[name])
            _assert_metric(name, metrics[name])
        if not bool(actual["latch"].all()):
            raise AssertionError("clean nonzero replay unexpectedly latched off")
    else:
        expected_latch = np.asarray(
            [layer0 < fault_layer0 for layer0 in range(16)], dtype=np.bool_
        )
        if not np.array_equal(actual["latch"], expected_latch):
            raise AssertionError("fault did not permanently latch off at injected layer")
        if not np.array_equal(
            np.isnan(actual["standardized"]), np.isnan(reference["standardized"])
        ):
            raise AssertionError("fault nonfinite masks differ")
        safe_start = fault_layer0
        for name in ("hidden", "unit", "physical", "predictions"):
            if not np.array_equal(actual[name][safe_start:], reference[name][safe_start:]):
                raise AssertionError(f"latched host-fallback bytes differ: {name}")

    return {
        "status": "PASS_TARGET_FREE_TORCH_FLOPSCOPE_INTERNAL_STATE_PARITY",
        "arm": arm,
        "fault_layer0": fault_layer0,
        "hidden_channels": hidden,
        "cell_width": cell_width,
        "tensor_bundle_sha256": manifest["torch_tensor_bundle_sha256"],
        "host_tape_hashes": observed_tape_hashes,
        "entering_cp_state_hashes": entering_cp_hashes,
        "entering_cp_chain_sha256": entering_cp_chain,
        "metrics": metrics,
        "latch": [bool(value) for value in actual["latch"]],
        "torch_prediction_sha256": _array_hash(actual["predictions"]),
        "flopscope_prediction_sha256": _array_hash(reference["predictions"]),
        "layer1_structural_zero": True,
        "device": "cpu",
        "mps_used": False,
        "target_values_decoded": 0,
        "optimizer_constructed": False,
        "fits": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    exporter = subparsers.add_parser("export")
    exporter.add_argument("--bundle", type=Path, required=True)
    exporter.add_argument("--hidden", type=int, required=True)
    exporter.add_argument("--cell-width", type=int, required=True)
    comparer = subparsers.add_parser("compare")
    comparer.add_argument("--bundle", type=Path, required=True)
    comparer.add_argument("--trace-dir", type=Path, required=True)
    comparer.add_argument("--arm", choices=("local_no_recurrence", "recurrent"), required=True)
    comparer.add_argument("--fault-layer0", type=int)
    args = parser.parse_args()
    if args.mode == "export":
        report = export_parity_bundle(
            observer_model_path=V3 / "observer_model.py",
            output_dir=args.bundle,
            hidden_channels=args.hidden,
            cell_width=args.cell_width,
        )
        report["status"] = "PASS_CLOSED_NONZERO_TORCH_CHECKPOINT_EXPORT"
        report["device"] = "cpu"
        report["mps_used"] = False
    else:
        report = compare(
            bundle_dir=args.bundle,
            trace_dir=args.trace_dir,
            arm=args.arm,
            fault_layer0=args.fault_layer0,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
