#!/usr/bin/env python3
"""Pure target-free science for the fast CP1792 residual ridge.

This module has no dataset, reader, feature-store, filesystem, or optimizer
dependency.  A later claim-owning launcher may pass already authorized arrays
to these functions after independently validating every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEPTH = 16
WIDTH = 1024
TAIL_FEATURES = 19
SCALE_FLOOR = np.float64(2.0**-20)
UNIT_CLIP = np.float64(0.25)
RIDGE_LAMBDAS = tuple(np.float64(value) for value in (2.0**-24, 2.0**-18, 2.0**-12, 2.0**-6, 1.0))
BASIS_ORDER = ("linear20", "context77")
BASIS_WIDTH = {"linear20": 20, "context77": 77}


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} is nonfinite")
    return array


@dataclass
class NormalizationStatistics:
    """Exact V4 fit-only scalar-visitation Welford statistics."""

    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls) -> "NormalizationStatistics":
        return cls(
            count=np.zeros(DEPTH, dtype=np.int64),
            mean=np.zeros((DEPTH, TAIL_FEATURES), dtype=np.float64),
            m2=np.zeros((DEPTH, TAIL_FEATURES), dtype=np.float64),
        )

    def add_raw_network(self, raw_row: np.ndarray) -> None:
        raw = _finite("raw_row", raw_row)
        if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, 20) or not raw.flags.c_contiguous:
            raise ValueError("raw_row must be contiguous float32[16,1024,20]")
        for layer0 in range(DEPTH):
            for neuron in range(WIDTH):
                self.count[layer0] += 1
                sample = raw[layer0, neuron, 1:].astype(np.float64)
                delta = sample - self.mean[layer0]
                self.mean[layer0] += delta / np.float64(self.count[layer0])
                self.m2[layer0] += delta * (sample - self.mean[layer0])

    def finalize(self, expected_networks: int) -> tuple[np.ndarray, np.ndarray]:
        expected = np.full(DEPTH, expected_networks * WIDTH, dtype=np.int64)
        if expected_networks <= 0 or not np.array_equal(self.count, expected):
            raise ValueError("normalization observation counts changed")
        scales = np.maximum(
            np.sqrt(self.m2 / self.count[:, None].astype(np.float64)), SCALE_FLOOR
        )
        if not np.isfinite(self.mean).all() or not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise FloatingPointError("normalization statistics are invalid")
        return np.ascontiguousarray(self.mean), np.ascontiguousarray(scales)


def standardize_raw_network(
    raw_row: np.ndarray, centers: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Apply fit-only V4 normalization; x01 is the exact depth bypass."""

    raw = _finite("raw_row", raw_row)
    center = _finite("centers", centers)
    scale = _finite("scales", scales)
    if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, 20) or not raw.flags.c_contiguous:
        raise ValueError("raw_row must be contiguous float32[16,1024,20]")
    if center.dtype != np.float64 or scale.dtype != np.float64:
        raise ValueError("normalization centers/scales must be float64")
    if center.shape != (DEPTH, TAIL_FEATURES) or scale.shape != (DEPTH, TAIL_FEATURES):
        raise ValueError("normalization geometry changed")
    if np.any(scale <= 0.0):
        raise FloatingPointError("normalization scale is not positive")
    tail = np.clip(
        (raw[..., 1:].astype(np.float64) - center[:, None, :]) / scale[:, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)
    depth = np.asarray(
        [np.float32(2.0 * layer0 / 15.0 - 1.0) for layer0 in range(DEPTH)],
        dtype=np.float32,
    )[:, None, None]
    result = np.ascontiguousarray(
        np.concatenate((np.broadcast_to(depth, (DEPTH, WIDTH, 1)), tail), axis=2)
    )
    if result.shape != (DEPTH, WIDTH, 20) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise AssertionError("standardized row schema changed")
    return result


def basis_for_layer(standardized_row: np.ndarray, layer0: int, basis: str) -> np.ndarray:
    """Return the fixed float64 width-equivariant basis for one layer."""

    row = _finite("standardized_row", standardized_row)
    if row.dtype != np.float32 or row.shape != (DEPTH, WIDTH, 20) or not row.flags.c_contiguous:
        raise ValueError("standardized_row must be contiguous float32[16,1024,20]")
    if not 0 <= layer0 < DEPTH or basis not in BASIS_ORDER:
        raise ValueError("invalid layer or basis")
    z = np.ascontiguousarray(row[layer0, :, 1:].astype(np.float64))
    if np.any(z < -8.0) or np.any(z > 8.0):
        raise ValueError("standardized tail is outside the frozen clip")
    ones = np.ones((WIDTH, 1), dtype=np.float64)
    if basis == "linear20":
        result = np.concatenate((ones, z), axis=1)
    else:
        mean = np.mean(z, axis=0, dtype=np.float64)
        rms = np.sqrt(np.mean(z * z, axis=0, dtype=np.float64))
        result = np.concatenate(
            (
                ones,
                z,
                z * z - np.float64(1.0),
                np.broadcast_to(mean, (WIDTH, TAIL_FEATURES)),
                np.broadcast_to(rms - np.float64(1.0), (WIDTH, TAIL_FEATURES)),
            ),
            axis=1,
        )
    result = np.ascontiguousarray(result, dtype=np.float64)
    if result.shape != (WIDTH, BASIS_WIDTH[basis]) or not np.isfinite(result).all():
        raise AssertionError("ridge basis construction failed")
    return result


@dataclass
class SufficientStatistics:
    basis: str
    gram: np.ndarray
    rhs: np.ndarray
    observations: np.ndarray

    @classmethod
    def empty(cls, basis: str) -> "SufficientStatistics":
        if basis not in BASIS_ORDER:
            raise ValueError("unknown ridge basis")
        p = BASIS_WIDTH[basis]
        return cls(
            basis=basis,
            gram=np.zeros((DEPTH, p, p), dtype=np.float64),
            rhs=np.zeros((DEPTH, p), dtype=np.float64),
            observations=np.zeros(DEPTH, dtype=np.int64),
        )

    def add_network(
        self,
        standardized_row: np.ndarray,
        host_rows: np.ndarray,
        sigma: np.ndarray,
        target: np.ndarray,
    ) -> None:
        host = _finite("host_rows", host_rows)
        scale = _finite("sigma", sigma)
        truth = _finite("target", target)
        for name, value in (("host_rows", host), ("sigma", scale), ("target", truth)):
            if value.dtype != np.float32 or value.shape != (DEPTH, WIDTH) or not value.flags.c_contiguous:
                raise ValueError(f"{name} must be contiguous float32[16,1024]")
        if np.any(host < 0.0) or np.any(scale <= 0.0) or np.any(truth < 0.0):
            raise FloatingPointError("host, sigma, and target sign contract failed")
        for layer0 in range(1, DEPTH):
            phi = basis_for_layer(standardized_row, layer0, self.basis)
            sigma64 = scale[layer0].astype(np.float64)
            residual = truth[layer0].astype(np.float64) - host[layer0].astype(np.float64)
            weighted = sigma64[:, None] * phi
            update = weighted.T @ weighted
            # Make symmetry a byte-level invariant instead of depending on a
            # particular BLAS kernel's two triangular accumulation orders.
            update = (update + update.T) * np.float64(0.5)
            self.gram[layer0] += update
            self.rhs[layer0] += weighted.T @ residual
            self.observations[layer0] += WIDTH


def solve_ladder(stats: SufficientStatistics) -> dict[str, np.ndarray]:
    """Solve every frozen ridge value; layer-1 coefficients remain zero."""

    if stats.basis not in BASIS_ORDER:
        raise ValueError("statistics basis changed")
    p = BASIS_WIDTH[stats.basis]
    if stats.gram.shape != (DEPTH, p, p) or stats.rhs.shape != (DEPTH, p):
        raise ValueError("statistics geometry changed")
    if np.any(stats.observations[1:] <= 0) or stats.observations[0] != 0:
        raise ValueError("statistics observation counts changed")
    result: dict[str, np.ndarray] = {}
    penalty = np.ones(p, dtype=np.float64)
    penalty[0] = np.float64(0.0)
    for ladder_index, ridge in enumerate(RIDGE_LAMBDAS):
        coefficients = np.zeros((DEPTH, p), dtype=np.float64)
        for layer0 in range(1, DEPTH):
            gram = np.asarray(stats.gram[layer0], dtype=np.float64)
            if not np.array_equal(gram, gram.T):
                raise AssertionError("streamed Gram matrix lost exact symmetry")
            gram_scale = np.trace(gram, dtype=np.float64) / np.float64(p)
            if not np.isfinite(gram_scale) or gram_scale <= 0.0:
                raise FloatingPointError("ridge Gram scale is not positive finite")
            regularized = gram + np.diag(ridge * gram_scale * penalty)
            regularized[0, 0] += np.float64(2.0**-52) * gram_scale
            coefficients[layer0] = np.linalg.solve(regularized, stats.rhs[layer0])
        if not np.isfinite(coefficients).all() or np.any(coefficients[0] != 0.0):
            raise FloatingPointError("ridge solve produced an invalid coefficient array")
        result[f"lambda_{ladder_index}"] = np.ascontiguousarray(coefficients)
    return result


def predict_network(
    standardized_row: np.ndarray,
    host_rows: np.ndarray,
    sigma: np.ndarray,
    basis: str,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply one fitted local head while preserving the analytic layer 1."""

    host = _finite("host_rows", host_rows)
    scale = _finite("sigma", sigma)
    p = BASIS_WIDTH.get(basis)
    beta = _finite("coefficients", coefficients)
    if p is None or beta.dtype != np.float64 or beta.shape != (DEPTH, p):
        raise ValueError("coefficient schema changed")
    if host.dtype != np.float32 or scale.dtype != np.float32:
        raise ValueError("host and sigma must be float32")
    if host.shape != (DEPTH, WIDTH) or scale.shape != (DEPTH, WIDTH):
        raise ValueError("host and sigma geometry changed")
    if np.any(host < 0.0) or np.any(scale <= 0.0):
        raise FloatingPointError("host/sigma sign contract failed")
    prediction = np.array(host, dtype=np.float32, order="C", copy=True)
    for layer0 in range(1, DEPTH):
        phi = basis_for_layer(standardized_row, layer0, basis)
        unit = np.clip(phi @ beta[layer0], -UNIT_CLIP, UNIT_CLIP)
        physical = scale[layer0].astype(np.float64) * unit
        prediction[layer0] = np.maximum(
            host[layer0].astype(np.float64) + physical, np.float64(0.0)
        ).astype(np.float32)
    if not np.array_equal(prediction[0], host[0]):
        raise AssertionError("layer-1 ridge correction is not structurally zero")
    if not np.isfinite(prediction).all() or np.any(prediction < 0.0):
        raise FloatingPointError("ridge prediction is invalid")
    return prediction


def ordered_layer_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Deterministic float64 MSE in network, layer, neuron order."""

    pred = _finite("prediction", prediction)
    truth = _finite("target", target)
    if pred.dtype != np.float32 or truth.dtype != np.float32 or pred.shape != truth.shape:
        raise ValueError("prediction/target schema mismatch")
    if pred.ndim != 3 or pred.shape[1:] != (DEPTH, WIDTH):
        raise ValueError("prediction/target must be float32[N,16,1024]")
    totals = np.zeros(DEPTH, dtype=np.float64)
    for position in range(pred.shape[0]):
        for layer0 in range(DEPTH):
            error = pred[position, layer0].astype(np.float64) - truth[position, layer0].astype(np.float64)
            totals[layer0] += np.dot(error, error)
    return totals / np.float64(pred.shape[0] * WIDTH)


def select_once(
    host: np.ndarray,
    candidates: Iterable[tuple[str, np.ndarray]],
    target: np.ndarray,
) -> dict[str, object]:
    """Frozen host-first, L2-safe, strict-final-improvement selection."""

    host_mse = ordered_layer_mse(host, target)
    rows: list[dict[str, object]] = [
        {
            "name": "host",
            "layer2_mse": float(host_mse[1]),
            "final_mse": float(host_mse[15]),
            "layer2_non_regression": True,
        }
    ]
    winner_name = "host"
    winner_final = host_mse[15]
    seen = {"host"}
    for name, prediction in candidates:
        if name in seen:
            raise ValueError("duplicate selection candidate")
        seen.add(name)
        mse = ordered_layer_mse(prediction, target)
        eligible = bool(mse[1] <= host_mse[1])
        rows.append(
            {
                "name": name,
                "layer2_mse": float(mse[1]),
                "final_mse": float(mse[15]),
                "layer2_non_regression": eligible,
            }
        )
        if eligible and mse[15] < winner_final:
            winner_name = name
            winner_final = mse[15]
    return {
        "winner": winner_name,
        "winner_final_mse": float(winner_final),
        "learned_candidate_selected": winner_name != "host",
        "rows": rows,
    }


if __name__ == "__main__":
    raise SystemExit("pure science library; no standalone data or fit entry point")
