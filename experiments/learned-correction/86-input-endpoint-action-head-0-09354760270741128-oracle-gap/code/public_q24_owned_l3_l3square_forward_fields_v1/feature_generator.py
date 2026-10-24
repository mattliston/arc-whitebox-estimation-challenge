"""Target-blind owned L3/L3^2 product-before-transport fields.

This module performs no file I/O and has no target surface.  It consumes one
already-loaded native D11 row and emits four dimensionless, full-width fields
on the frozen layer-10--15 suffix.  The local reset terms are already owned by
D11; only forming them before the weight action is the proposed predictor
information.  The returned fields must never be added directly to an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_TARGET_LAYERS = tuple(range(10, 16))
RAW_ALPHA = 3
RAW_GATE = 5
RAW_PDF = 6
RAW_GAUSSIAN_MEAN_OVER_SIGMA = 7
RAW_STANDARDIZED_K3 = 10
FIELD_ORDER = (
    "mean_l3",
    "preactivation_variance_l3",
    "mean_l3_squared",
    "preactivation_variance_l3_squared",
)

NEGATIVE_SIXTH = np.float32(-1.0 / 6.0)
ONE_THIRD = np.float32(1.0 / 3.0)
ONE_SEVENTY_SECOND = np.float32(1.0 / 72.0)
NEGATIVE_ONE_THIRTY_SIXTH = np.float32(-1.0 / 36.0)


@dataclass(frozen=True)
class OwnedProductTransportFeatures:
    """Frozen field payload for one network.

    ``features`` has shape ``[len(target_layers), width, 4]`` and uses
    ``FIELD_ORDER``.  Mean providers are divided by target ``sigma``;
    incoming diagonal-variance providers are divided by target ``sigma^2``.
    No corpus statistic, fold statistic, target, clipping, or learned
    quantity enters this transform.
    """

    target_layers: tuple[int, ...]
    field_order: tuple[str, ...]
    features: np.ndarray
    used_resident_squared_weights: bool


def _validate_native_row(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    squared_weights: np.ndarray | None,
    target_layers: Sequence[int],
) -> tuple[int, int, tuple[int, ...]]:
    for name, value in (("raw20", raw20), ("sigma", sigma), ("weights", weights)):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise TypeError(f"{name} must be a float32 ndarray")
    if raw20.ndim != 3 or raw20.shape[2] <= RAW_STANDARDIZED_K3:
        raise ValueError("raw20 must have shape [depth,width,features>=11]")
    depth, width, _ = raw20.shape
    if sigma.shape != (depth, width):
        raise ValueError("sigma shape mismatch")
    if weights.shape != (depth, width, width):
        raise ValueError("weights shape mismatch")
    if squared_weights is not None:
        if not isinstance(squared_weights, np.ndarray) or squared_weights.dtype != np.float32:
            raise TypeError("squared_weights must be None or a float32 ndarray")
        if squared_weights.shape != weights.shape:
            raise ValueError("squared_weights shape mismatch")

    layers = tuple(int(layer) for layer in target_layers)
    if not layers or layers != tuple(sorted(set(layers))):
        raise ValueError("target_layers must be a nonempty strictly increasing sequence")
    if layers[0] < 1 or layers[-1] >= depth:
        raise ValueError("target layer outside native row")
    for layer in sorted(set(layers) | {layer - 1 for layer in layers}):
        layer_sigma = sigma[layer]
        if not np.isfinite(layer_sigma).all() or np.any(layer_sigma <= 0.0):
            raise ValueError("selected sigma values must be finite and positive")
    return depth, width, layers


def _owned_source_actions(
    raw: np.ndarray,
    sigma: np.ndarray,
    mean_out: np.ndarray,
    variance_out: np.ndarray,
    scratch: np.ndarray,
) -> None:
    """Write the exact D11-owned L3 and L3^2 local actions.

    Columns of ``mean_out`` are ``(dmu3,dmu33)``.  Columns of
    ``variance_out`` are ``(dv3,dv33)``.  ``dv*`` are centered-variance
    actions at the source reset, before their squared-edge push.
    """

    alpha = raw[:, RAW_ALPHA]
    pdf = raw[:, RAW_PDF]
    gaussian_mean_over_sigma = raw[:, RAW_GAUSSIAN_MEAN_OVER_SIGMA]
    gamma3 = raw[:, RAW_STANDARDIZED_K3]
    sigma2 = scratch[:, 0]
    gamma3_squared = scratch[:, 1]
    alpha_squared = scratch[:, 2]
    work = scratch[:, 3]

    # dmu3 = -sigma * alpha * pdf * gamma3 / 6.
    np.multiply(sigma, alpha, out=mean_out[:, 0])
    np.multiply(mean_out[:, 0], pdf, out=mean_out[:, 0])
    np.multiply(mean_out[:, 0], gamma3, out=mean_out[:, 0])
    np.multiply(mean_out[:, 0], NEGATIVE_SIXTH, out=mean_out[:, 0])

    # dv3 = sigma^2 * pdf * gamma3 * (1 + alpha*h) / 3.
    np.multiply(sigma, sigma, out=sigma2)
    np.multiply(sigma2, pdf, out=variance_out[:, 0])
    np.multiply(variance_out[:, 0], gamma3, out=variance_out[:, 0])
    np.multiply(alpha, gaussian_mean_over_sigma, out=work)
    np.add(work, np.float32(1.0), out=work)
    np.multiply(variance_out[:, 0], work, out=variance_out[:, 0])
    np.multiply(variance_out[:, 0], ONE_THIRD, out=variance_out[:, 0])

    np.multiply(gamma3, gamma3, out=gamma3_squared)
    np.multiply(alpha, alpha, out=alpha_squared)

    # H4(alpha) = alpha^4 - 6 alpha^2 + 3.
    np.multiply(alpha_squared, alpha_squared, out=work)
    np.multiply(alpha_squared, np.float32(6.0), out=mean_out[:, 1])
    np.subtract(work, mean_out[:, 1], out=work)
    np.add(work, np.float32(3.0), out=work)

    # dmu33 = sigma * gamma3^2 * pdf * H4(alpha) / 72.
    np.multiply(sigma, gamma3_squared, out=mean_out[:, 1])
    np.multiply(mean_out[:, 1], pdf, out=mean_out[:, 1])
    np.multiply(mean_out[:, 1], work, out=mean_out[:, 1])
    np.multiply(mean_out[:, 1], ONE_SEVENTY_SECOND, out=mean_out[:, 1])

    # H3(alpha) = alpha^3 - 3 alpha, followed by dm233.
    np.multiply(alpha_squared, alpha, out=work)
    np.multiply(alpha, np.float32(3.0), out=variance_out[:, 1])
    np.subtract(work, variance_out[:, 1], out=work)
    np.multiply(sigma2, gamma3_squared, out=variance_out[:, 1])
    np.multiply(variance_out[:, 1], pdf, out=variance_out[:, 1])
    np.multiply(variance_out[:, 1], work, out=variance_out[:, 1])
    np.multiply(
        variance_out[:, 1],
        NEGATIVE_ONE_THIRTY_SIXTH,
        out=variance_out[:, 1],
    )

    # dv33 = dm233 - 2*(sigma*h)*dmu33 - dmu3^2.
    np.multiply(sigma, gaussian_mean_over_sigma, out=work)
    np.multiply(work, mean_out[:, 1], out=work)
    np.multiply(work, np.float32(2.0), out=work)
    np.subtract(variance_out[:, 1], work, out=variance_out[:, 1])
    np.multiply(mean_out[:, 0], mean_out[:, 0], out=work)
    np.subtract(variance_out[:, 1], work, out=variance_out[:, 1])


def build_features(
    raw20: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    *,
    squared_weights: np.ndarray | None = None,
    target_layers: Sequence[int] = DEFAULT_TARGET_LAYERS,
) -> OwnedProductTransportFeatures:
    """Build the frozen in-memory four-field payload for one network.

    At target layer ``l``, the generator forms the nonlinear source actions at
    ``l-1`` *before* transport, then computes

    ``gate_l * W_l.T @ (dmu3,dmu33)`` and
    ``(W_l^2).T @ (dv3,dv33)``.

    Supplying an already-resident exact float32 ``squared_weights`` array
    realizes the frozen 24/1024 dense-product cost.  If it is omitted, the
    function lawfully forms one layer-sized square scratch and the resource
    census must use ``resident_squared_weights=False``.
    """

    _, width, layers = _validate_native_row(
        raw20, sigma, weights, squared_weights, target_layers
    )
    count = len(layers)
    features = np.empty((count, width, len(FIELD_ORDER)), dtype=np.float32)
    source_mean = np.empty((width, 2), dtype=np.float32)
    source_variance = np.empty((width, 2), dtype=np.float32)
    transported = np.empty((width, 2), dtype=np.float32)
    scratch = np.empty((width, 4), dtype=np.float32)
    weight_square_scratch = (
        None
        if squared_weights is not None
        else np.empty((width, width), dtype=np.float32)
    )

    for slot, target_layer in enumerate(layers):
        source_layer = target_layer - 1
        _owned_source_actions(
            raw20[source_layer],
            sigma[source_layer],
            source_mean,
            source_variance,
            scratch,
        )

        np.matmul(weights[target_layer].T, source_mean, out=transported)
        np.multiply(
            transported,
            raw20[target_layer, :, RAW_GATE, None],
            out=transported,
        )
        np.divide(transported[:, 0], sigma[target_layer], out=features[slot, :, 0])
        np.divide(transported[:, 1], sigma[target_layer], out=features[slot, :, 2])

        if squared_weights is None:
            assert weight_square_scratch is not None
            np.multiply(
                weights[target_layer],
                weights[target_layer],
                out=weight_square_scratch,
            )
            weight2 = weight_square_scratch
        else:
            weight2 = squared_weights[target_layer]
        np.matmul(weight2.T, source_variance, out=transported)
        np.multiply(sigma[target_layer], sigma[target_layer], out=scratch[:, 0])
        np.divide(transported[:, 0], scratch[:, 0], out=features[slot, :, 1])
        np.divide(transported[:, 1], scratch[:, 0], out=features[slot, :, 3])

    if not np.isfinite(features).all():
        raise FloatingPointError("nonfinite owned product-transport field")
    return OwnedProductTransportFeatures(
        target_layers=layers,
        field_order=FIELD_ORDER,
        features=features,
        used_resident_squared_weights=squared_weights is not None,
    )


def operation_census(
    *,
    width: int = 1024,
    seams: int = 6,
    itemsize: int = 4,
    resident_squared_weights: bool = True,
) -> dict[str, int | float | bool]:
    """Return the exact declared arithmetic and explicit-array byte census.

    FLOPs count each scalar add/subtract/multiply/divide as one and an
    ``n``-term dot product as ``2n-1``.  One reference dense square GEMM is
    therefore ``n^2(2n-1)`` FLOPs.  Bytes cover arrays explicitly owned by
    this module, plus the transient output-finiteness Boolean array; caller
    inputs and implementation-dependent BLAS workspace are excluded.
    """

    width = int(width)
    seams = int(seams)
    itemsize = int(itemsize)
    if width <= 0 or seams <= 0 or itemsize <= 0:
        raise ValueError("census dimensions must be positive")

    source_action_flops = 34 * seams * width
    one_two_column_product_flops = 2 * width * (2 * width - 1)
    dense_product_flops = 2 * seams * one_two_column_product_flops
    gate_and_standardization_flops = 7 * seams * width
    squared_weight_formation_flops = (
        0 if resident_squared_weights else seams * width * width
    )
    arithmetic_flops = (
        source_action_flops
        + dense_product_flops
        + gate_and_standardization_flops
        + squared_weight_formation_flops
    )
    reference_dense_gemm_flops = width * width * (2 * width - 1)

    output_bytes = seams * width * len(FIELD_ORDER) * itemsize
    source_work_and_scratch_bytes = 10 * width * itemsize
    weight_square_scratch_bytes = (
        0 if resident_squared_weights else width * width * itemsize
    )
    finiteness_boolean_bytes = seams * width * len(FIELD_ORDER)
    explicit_peak_live_bytes = (
        output_bytes
        + source_work_and_scratch_bytes
        + weight_square_scratch_bytes
        + finiteness_boolean_bytes
    )
    return {
        "width": width,
        "seams": seams,
        "resident_squared_weights": bool(resident_squared_weights),
        "source_action_flops": source_action_flops,
        "dense_product_flops": dense_product_flops,
        "gate_and_standardization_flops": gate_and_standardization_flops,
        "squared_weight_formation_flops": squared_weight_formation_flops,
        "arithmetic_flops": arithmetic_flops,
        "reference_dense_gemm_flops": reference_dense_gemm_flops,
        "dense_product_gemm_equivalents": dense_product_flops
        / reference_dense_gemm_flops,
        "all_arithmetic_gemm_equivalents": arithmetic_flops
        / reference_dense_gemm_flops,
        "output_bytes": output_bytes,
        "source_work_and_scratch_bytes": source_work_and_scratch_bytes,
        "weight_square_scratch_bytes": weight_square_scratch_bytes,
        "finiteness_boolean_bytes": finiteness_boolean_bytes,
        "explicit_peak_live_bytes": explicit_peak_live_bytes,
    }
