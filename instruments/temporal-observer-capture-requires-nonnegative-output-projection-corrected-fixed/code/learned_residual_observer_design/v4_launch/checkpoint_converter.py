#!/usr/bin/env python3
"""Deterministic Torch-to-deployment tensor converter for V4 parity.

Torch is imported only inside export functions.  Deployment-side loading uses
NumPy and validates every raw tensor byte.  Verifiers write bundles only below
fresh /private/tmp directories; no target, checkpoint, or production path is
opened by module import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_ORDER = (
    "core1.weight",
    "core1.bias",
    "core2.weight",
    "core2.bias",
    "candidate_head.weight",
    "candidate_head.bias",
    "gate_head.weight",
    "gate_head.bias",
    "mean_head.weight",
    "mean_head.bias",
    "source_gate_logits",
)
PARITY_SEED = 501163896429966340
EXPECTED_OBSERVER_MODEL_SHA256 = (
    "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _tensor_bundle_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in PARAMETER_ORDER:
        array = np.ascontiguousarray(arrays[name])
        metadata = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        for part in (name.encode("utf-8"), metadata, memoryview(array).cast("B")):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def expected_shapes(hidden_channels: int, cell_width: int) -> dict[str, tuple[int, ...]]:
    h, c = int(hidden_channels), int(cell_width)
    if h <= 0 or h % 4 or c <= 0:
        raise ValueError("hidden channels must be positive/divisible by four")
    inputs = 24 + h
    return {
        "core1.weight": (c, inputs),
        "core1.bias": (c,),
        "core2.weight": (c, c),
        "core2.bias": (c,),
        "candidate_head.weight": (h, c),
        "candidate_head.bias": (h,),
        "gate_head.weight": (h, c),
        "gate_head.bias": (h,),
        "mean_head.weight": (1, c),
        "mean_head.bias": (1,),
        "source_gate_logits": (8,),
    }


def parity_normalization() -> tuple[np.ndarray, np.ndarray]:
    layers = np.arange(16, dtype=np.float64)[:, None]
    features = np.arange(19, dtype=np.float64)[None, :]
    centers = np.ascontiguousarray((layers - 7.5) * 1.0e-3 + (features - 9.0) * 1.0e-4)
    scales = np.ascontiguousarray(1.0 + layers * 2.0e-3 + features * 3.0e-4)
    return centers, scales


def export_model_bundle(
    *,
    model: Any,
    centers: np.ndarray,
    scales: np.ndarray,
    output_dir: Path,
    hidden_channels: int,
    cell_width: int,
    provenance: str,
) -> dict[str, Any]:
    """Export exact CPU tensors to exclusive raw files plus canonical manifest."""

    shapes = expected_shapes(hidden_channels, cell_width)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("converter output path already exists")
    output_dir.mkdir(mode=0o700)
    named = dict(model.named_parameters())
    if set(named) != set(PARAMETER_ORDER):
        raise AssertionError("Torch parameter membership changed")
    arrays: dict[str, np.ndarray] = {}
    tensors: dict[str, Any] = {}
    for index, name in enumerate(PARAMETER_ORDER):
        tensor = named[name]
        if str(tensor.device) != "cpu" or str(tensor.dtype) != "torch.float32":
            raise AssertionError("converter accepts CPU float32 parameters only")
        array = np.ascontiguousarray(tensor.detach().numpy(), dtype=np.float32)
        if array.shape != shapes[name]:
            raise AssertionError(f"parameter shape changed: {name}")
        file_name = f"{index:02d}-{name.replace('.', '_')}.f32le"
        path = output_dir / file_name
        payload = memoryview(array.astype("<f4", copy=False)).cast("B")
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        arrays[name] = array
        tensors[name] = {
            "path": file_name,
            "dtype": "<f4",
            "shape": list(array.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    stats: dict[str, Any] = {}
    for name, value in (("centers", centers), ("scales", scales)):
        array = np.ascontiguousarray(value, dtype="<f8")
        if array.shape != (16, 19) or not np.isfinite(array).all():
            raise AssertionError("normalization statistic changed")
        if name == "scales" and not (array > 0.0).all():
            raise AssertionError("normalization scale is nonpositive")
        path = output_dir / f"{name}.f64le"
        payload = memoryview(array).cast("B")
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        stats[name] = {
            "path": path.name,
            "dtype": "<f8",
            "shape": [16, 19],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest = {
        "schema": "phase2-observer-v4-deterministic-tensor-bundle-v1",
        "provenance": provenance,
        "hidden_channels": int(hidden_channels),
        "cell_width": int(cell_width),
        "parameter_order": list(PARAMETER_ORDER),
        "torch_tensor_bundle_sha256": _tensor_bundle_sha256(arrays),
        "tensors": tensors,
        "normalization": stats,
        "target_values_decoded": 0,
        "fit_performed": False,
    }
    manifest_path = output_dir / "manifest.json"
    payload = _canonical_json(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {**manifest, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def export_parity_bundle(
    *,
    observer_model_path: Path,
    output_dir: Path,
    hidden_channels: int,
    cell_width: int,
) -> dict[str, Any]:
    """Create the closed nonzero checkpoint used only for converter parity."""

    if sha256_file(observer_model_path) != EXPECTED_OBSERVER_MODEL_SHA256:
        raise AssertionError("frozen Torch observer model changed")
    spec = importlib.util.spec_from_file_location("v4_frozen_observer_model", observer_model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen observer model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ResidualObserver(
        arm="recurrent",
        init_seed=PARITY_SEED,
        parity_nonzero_mean_head=True,
        hidden_channels=hidden_channels,
        cell_width=cell_width,
    )
    centers, scales = parity_normalization()
    report = export_model_bundle(
        model=model,
        centers=centers,
        scales=scales,
        output_dir=output_dir,
        hidden_channels=hidden_channels,
        cell_width=cell_width,
        provenance="closed_target_free_parity_initialization_seed_501163896429966340",
    )
    report["observer_model_sha256"] = EXPECTED_OBSERVER_MODEL_SHA256
    report["parity_seed"] = PARITY_SEED
    return report


def load_numpy_bundle(bundle_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase2-observer-v4-deterministic-tensor-bundle-v1":
        raise AssertionError("wrong converter bundle schema")
    order = tuple(manifest.get("parameter_order", ()))
    if order != PARAMETER_ORDER:
        raise AssertionError("converter parameter order changed")
    shapes = expected_shapes(manifest["hidden_channels"], manifest["cell_width"])
    arrays: dict[str, np.ndarray] = {}
    for name in PARAMETER_ORDER:
        record = manifest["tensors"][name]
        path = bundle_dir / record["path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise AssertionError(f"converted parameter hash changed: {name}")
        array = np.frombuffer(payload, dtype="<f4").reshape(shapes[name]).copy()
        arrays[name] = np.ascontiguousarray(array, dtype=np.float32)
    if _tensor_bundle_sha256(arrays) != manifest["torch_tensor_bundle_sha256"]:
        raise AssertionError("converted tensor bundle hash changed")
    stats_arrays: dict[str, np.ndarray] = {}
    for name in ("centers", "scales"):
        record = manifest["normalization"][name]
        path = bundle_dir / record["path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise AssertionError(f"normalization hash changed: {name}")
        stats_arrays[name] = np.frombuffer(payload, dtype="<f8").reshape(16, 19).copy()
    return arrays, stats_arrays["centers"], stats_arrays["scales"], manifest
