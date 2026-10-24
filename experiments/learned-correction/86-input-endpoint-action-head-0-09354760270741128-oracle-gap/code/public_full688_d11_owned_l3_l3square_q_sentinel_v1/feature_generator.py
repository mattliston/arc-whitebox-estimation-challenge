"""Adapter for the frozen four owned L3/L3^2 transport fields.

This module is target blind.  It only adapts the separately frozen generator
to the exact API used by the pinned fold-0 q-head runner.  No feature is
persisted: every payload is returned in memory for layers 10 through 15.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FROZEN_SOURCE = (
    P2
    / "public_q24_owned_l3_l3square_forward_fields_v1"
    / "feature_generator.py"
)
FROZEN_SOURCE_SHA256 = (
    "5f4ed4a777f5ab80141bc2e3d1ced73b973f7a60b9f694dd330ca461b545c086"
)
LAYERS = tuple(range(10, 16))
RAW_PHI = 5
FIELD_ORDER = (
    "mean_l3",
    "preactivation_variance_l3",
    "mean_l3_squared",
    "preactivation_variance_l3_squared",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen():
    if not FROZEN_SOURCE.is_file() or _sha256(FROZEN_SOURCE) != FROZEN_SOURCE_SHA256:
        raise AssertionError("frozen four-field generator changed")
    name = "owned_l3_l3square_frozen_generator"
    specification = importlib.util.spec_from_file_location(name, FROZEN_SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(FROZEN_SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_FROZEN = _load_frozen()


@dataclass(frozen=True)
class AdaptedFeatures:
    layers: tuple[int, ...]
    field_order: tuple[str, ...]
    appended_features: np.ndarray
    used_resident_squared_weights: bool


def build_features(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    layer_rms: np.ndarray,
    *,
    first_layer: int = 10,
    last_layer: int = 15,
) -> AdaptedFeatures:
    """Return the exact four-field suffix payload expected by the q runner."""

    if (int(first_layer), int(last_layer)) != (10, 15):
        raise ValueError("the frozen arm is exactly the layer-10--15 suffix")
    if not isinstance(layer_rms, np.ndarray) or layer_rms.dtype != np.float32:
        raise TypeError("layer_rms must be a float32 ndarray")
    if layer_rms.shape != (raw20.shape[0],):
        raise ValueError("layer_rms shape mismatch")
    if not np.isfinite(layer_rms).all() or np.any(layer_rms <= 0.0):
        raise ValueError("layer_rms must be finite and positive")

    built = _FROZEN.build_features(
        raw20,
        sigma,
        weights,
        squared_weights=None,
        target_layers=LAYERS,
    )
    if built.target_layers != LAYERS or built.field_order != FIELD_ORDER:
        raise AssertionError("frozen four-field contract changed")
    values = np.ascontiguousarray(built.features, dtype=np.float32)
    if values.shape != (len(LAYERS), raw20.shape[1], len(FIELD_ORDER)):
        raise AssertionError("adapted four-field shape changed")
    return AdaptedFeatures(
        layers=LAYERS,
        field_order=FIELD_ORDER,
        appended_features=values,
        used_resident_squared_weights=built.used_resident_squared_weights,
    )

