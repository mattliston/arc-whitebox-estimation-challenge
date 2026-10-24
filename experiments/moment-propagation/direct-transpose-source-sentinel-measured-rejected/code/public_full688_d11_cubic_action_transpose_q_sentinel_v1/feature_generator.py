"""Target-blind in-memory D11 cubic-action transpose features.

This module performs no file I/O and knows no target path.  It consumes one
already-loaded native row and returns only in-memory float32 arrays.  The
physical cubic source is old-owned by D11; callers may use these arrays only as
predictor coordinates, never as a direct additive correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_FIRST_LAYER = 5
DEFAULT_LAST_LAYER = 15
RAW_ALPHA = 3
RAW_PHI = 5
RAW_PDF = 6
RAW_STANDARDIZED_K3 = 10
CLIP = np.float32(8.0)
DENOMINATOR_FLOOR = np.float32(2.0**-30)
NEGATIVE_SIXTH = np.float32(-1.0 / 6.0)


@dataclass(frozen=True)
class CubicActionTransposeFeatures:
    """Selected-layer physical source, transpose scan, and normalized fields."""

    layers: tuple[int, ...]
    cubic_action: np.ndarray
    transpose_action: np.ndarray
    appended_features: np.ndarray


def _validate_native_row(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    first_layer: int,
    last_layer: int,
) -> tuple[int, int, tuple[int, ...]]:
    if not isinstance(raw20, np.ndarray) or raw20.dtype != np.float32:
        raise TypeError("raw20 must be a float32 ndarray")
    if not isinstance(sigma, np.ndarray) or sigma.dtype != np.float32:
        raise TypeError("sigma must be a float32 ndarray")
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
        raise TypeError("weights must be a float32 ndarray")
    if raw20.ndim != 3 or raw20.shape[2] <= RAW_STANDARDIZED_K3:
        raise ValueError("raw20 must have shape [depth,width,features>=11]")
    depth, width, _ = raw20.shape
    if sigma.shape != (depth, width):
        raise ValueError("sigma shape mismatch")
    if weights.shape != (depth, width, width):
        raise ValueError("weights shape mismatch")
    if not 0 <= int(first_layer) <= int(last_layer) < depth:
        raise ValueError("invalid selected-layer interval")
    layers = tuple(range(int(first_layer), int(last_layer) + 1))
    return depth, width, layers


def cubic_action_transpose(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    *,
    first_layer: int = DEFAULT_FIRST_LAYER,
    last_layer: int = DEFAULT_LAST_LAYER,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Return ``(layers,c,b)`` using the frozen float32 operation order.

    For a selected layer ``l``, ``c`` is the D11 cubic mean action

        c_l = -(sigma_l * alpha_l * pdf_l * standardized_k3_l) / 6.

    The reverse scan uses the exact adjoint of the native row-vector tangent:

        b_last = c_last
        b_l = c_l + W_(l+1) @ (phi_(l+1) * b_(l+1)).
    """

    _, width, layers = _validate_native_row(
        raw20, sigma, weights, first_layer, last_layer
    )
    count = len(layers)
    cubic = np.empty((count, width), dtype=np.float32)
    transpose = np.empty_like(cubic)

    for slot, layer in enumerate(layers):
        np.multiply(sigma[layer], raw20[layer, :, RAW_ALPHA], out=cubic[slot])
        np.multiply(cubic[slot], raw20[layer, :, RAW_PDF], out=cubic[slot])
        np.multiply(
            cubic[slot],
            raw20[layer, :, RAW_STANDARDIZED_K3],
            out=cubic[slot],
        )
        np.multiply(cubic[slot], NEGATIVE_SIXTH, out=cubic[slot])

    np.copyto(transpose[-1], cubic[-1])
    scratch = np.empty(width, dtype=np.float32)
    for slot in range(count - 2, -1, -1):
        next_layer = layers[slot + 1]
        np.multiply(
            raw20[next_layer, :, RAW_PHI], transpose[slot + 1], out=scratch
        )
        np.matmul(weights[next_layer], scratch, out=transpose[slot])
        np.add(transpose[slot], cubic[slot], out=transpose[slot])
    return layers, cubic, transpose


def normalized_appended_features(
    layers: tuple[int, ...],
    cubic_action: np.ndarray,
    transpose_action: np.ndarray,
    sigma: np.ndarray,
    layer_rms: np.ndarray,
) -> np.ndarray:
    """Return the three frozen predictor coordinates in ``[-8,8]``."""

    if not isinstance(sigma, np.ndarray) or sigma.dtype != np.float32:
        raise TypeError("sigma must be a float32 ndarray")
    if not isinstance(layer_rms, np.ndarray) or layer_rms.dtype != np.float32:
        raise TypeError("layer_rms must be a float32 ndarray")
    if sigma.ndim != 2 or layer_rms.shape != (sigma.shape[0],):
        raise ValueError("sigma/layer_rms shape mismatch")
    count = len(layers)
    width = sigma.shape[1]
    expected = (count, width)
    if (
        cubic_action.dtype != np.float32
        or transpose_action.dtype != np.float32
        or cubic_action.shape != expected
        or transpose_action.shape != expected
    ):
        raise ValueError("cubic/transpose feature shape or dtype mismatch")
    if any(layer < 0 or layer >= sigma.shape[0] for layer in layers):
        raise ValueError("layer index outside sigma")
    if any(not np.isfinite(layer_rms[layer]) or layer_rms[layer] <= 0 for layer in layers):
        raise ValueError("selected layer RMS values must be finite and positive")

    features = np.empty((count, width, 3), dtype=np.float32)
    denominator = np.empty(width, dtype=np.float32)
    for slot, layer in enumerate(layers):
        rms = layer_rms[layer]
        np.divide(cubic_action[slot], rms, out=features[slot, :, 0])
        np.divide(transpose_action[slot], rms, out=features[slot, :, 1])
        np.multiply(sigma[layer], rms, out=denominator)
        np.maximum(denominator, DENOMINATOR_FLOOR, out=denominator)
        np.divide(transpose_action[slot], denominator, out=features[slot, :, 2])
    np.clip(features, -CLIP, CLIP, out=features)
    return features


def build_features(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    layer_rms: np.ndarray,
    *,
    first_layer: int = DEFAULT_FIRST_LAYER,
    last_layer: int = DEFAULT_LAST_LAYER,
) -> CubicActionTransposeFeatures:
    """Build the complete frozen in-memory feature payload for one network."""

    layers, cubic, transpose = cubic_action_transpose(
        raw20,
        sigma,
        weights,
        first_layer=first_layer,
        last_layer=last_layer,
    )
    appended = normalized_appended_features(
        layers, cubic, transpose, sigma, layer_rms
    )
    return CubicActionTransposeFeatures(layers, cubic, transpose, appended)


def operation_census(
    *, width: int = 1024, selected_layers: int = 11, itemsize: int = 4
) -> dict[str, int]:
    """Return the exact declared arithmetic and NumPy-owned live-array census."""

    width = int(width)
    selected_layers = int(selected_layers)
    itemsize = int(itemsize)
    if width <= 0 or selected_layers <= 0 or itemsize <= 0:
        raise ValueError("census dimensions must be positive")
    transitions = selected_layers - 1
    cubic_arithmetic = 4 * selected_layers * width
    reverse_matvec = transitions * (2 * width * width - width)
    reverse_elementwise = transitions * 2 * width
    normalization_arithmetic = 4 * selected_layers * width
    comparison_operations = 7 * selected_layers * width
    arithmetic_flops = (
        cubic_arithmetic
        + reverse_matvec
        + reverse_elementwise
        + normalization_arithmetic
    )
    c_and_b_bytes = 2 * selected_layers * width * itemsize
    appended_feature_bytes = 3 * selected_layers * width * itemsize
    scratch_and_denominator_bytes = 2 * width * itemsize
    return {
        "cubic_arithmetic_flops": cubic_arithmetic,
        "reverse_matvec_flops": reverse_matvec,
        "reverse_elementwise_flops": reverse_elementwise,
        "normalization_arithmetic_flops": normalization_arithmetic,
        "arithmetic_flops": arithmetic_flops,
        "comparison_operations": comparison_operations,
        "arithmetic_plus_comparison_operations": arithmetic_flops
        + comparison_operations,
        "c_and_b_bytes": c_and_b_bytes,
        "appended_feature_bytes": appended_feature_bytes,
        "scratch_and_denominator_bytes": scratch_and_denominator_bytes,
        "numpy_owned_peak_live_bytes": c_and_b_bytes
        + appended_feature_bytes
        + scratch_and_denominator_bytes,
    }
