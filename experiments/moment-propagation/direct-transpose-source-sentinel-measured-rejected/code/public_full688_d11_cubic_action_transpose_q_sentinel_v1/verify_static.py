#!/usr/bin/env python3
"""Tiny no-I/O algebra and equivariance checks for feature_generator.py."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import feature_generator as generator


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "feature_generator.py"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
DESIGN_FREEZE = HERE / "FREEZE_RECEIPT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_io_or_target_surface() -> dict[str, object]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imports: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    forbidden_imports = {
        "os",
        "pathlib",
        "pyarrow",
        "subprocess",
        "torch",
    }
    forbidden_calls = {
        "load",
        "save",
        "open",
        "open_memmap",
        "read_bytes",
        "read_text",
        "tofile",
        "write_bytes",
        "write_text",
    }
    if imports & forbidden_imports:
        raise AssertionError(sorted(imports & forbidden_imports))
    if calls & forbidden_calls:
        raise AssertionError(sorted(calls & forbidden_calls))
    source_text = SOURCE.read_text(encoding="utf-8").lower()
    for forbidden_word in (
        "observer_fit.parquet",
        "all_layer_means",
        "patched_error",
        "target_bundle",
    ):
        if forbidden_word in source_text:
            raise AssertionError(forbidden_word)
    public_functions = sorted(
        name
        for name, value in vars(generator).items()
        if inspect.isfunction(value) and value.__module__ == generator.__name__
    )
    return {
        "imports": sorted(imports),
        "forbidden_import_intersection": [],
        "forbidden_call_intersection": [],
        "public_functions": public_functions,
    }


def synthetic_row(seed: int = 0xC0B1C) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    depth, width, features = 4, 5, 20
    raw = rng.normal(0.0, 0.4, size=(depth, width, features)).astype(np.float32)
    raw[:, :, generator.RAW_PHI] = rng.uniform(
        0.05, 0.95, size=(depth, width)
    ).astype(np.float32)
    raw[:, :, generator.RAW_PDF] = rng.uniform(
        0.05, 0.4, size=(depth, width)
    ).astype(np.float32)
    sigma = rng.uniform(0.2, 1.4, size=(depth, width)).astype(np.float32)
    weights = rng.normal(0.0, 0.3, size=(depth, width, width)).astype(np.float32)
    layer_rms = rng.uniform(0.3, 1.1, size=depth).astype(np.float32)
    return raw, sigma, weights, layer_rms


def explicit_float64(
    raw: np.ndarray, sigma: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c = -(
        sigma.astype(np.float64)
        * raw[:, :, generator.RAW_ALPHA].astype(np.float64)
        * raw[:, :, generator.RAW_PDF].astype(np.float64)
        * raw[:, :, generator.RAW_STANDARDIZED_K3].astype(np.float64)
    ) / 6.0
    b = np.empty_like(c)
    b[-1] = c[-1]
    for layer in range(len(c) - 2, -1, -1):
        b[layer] = c[layer] + weights[layer + 1].astype(np.float64) @ (
            raw[layer + 1, :, generator.RAW_PHI].astype(np.float64)
            * b[layer + 1]
        )
    return c, b


def algebra_check() -> dict[str, float]:
    raw, sigma, weights, layer_rms = synthetic_row()
    result = generator.build_features(
        raw, sigma, weights, layer_rms, first_layer=0, last_layer=3
    )
    expected_c, expected_b = explicit_float64(raw, sigma, weights)
    c_error = float(np.max(np.abs(result.cubic_action.astype(np.float64) - expected_c)))
    b_error = float(
        np.max(np.abs(result.transpose_action.astype(np.float64) - expected_b))
    )
    if c_error > 2.0e-7 or b_error > 2.0e-6:
        raise AssertionError((c_error, b_error))

    maximum_duality_error = 0.0
    rng = np.random.default_rng(0xD0A1)
    for layer in range(1, 4):
        x = rng.normal(size=5)
        y = rng.normal(size=5)
        w = weights[layer].astype(np.float64)
        phi = raw[layer, :, generator.RAW_PHI].astype(np.float64)
        forward = phi * (x @ w)
        reverse = w @ (phi * y)
        error = abs(float(forward @ y) - float(x @ reverse))
        maximum_duality_error = max(maximum_duality_error, error)
    if maximum_duality_error > 2.0e-15:
        raise AssertionError(maximum_duality_error)

    manual = np.empty_like(result.appended_features)
    for layer in range(4):
        manual[layer, :, 0] = result.cubic_action[layer] / layer_rms[layer]
        manual[layer, :, 1] = result.transpose_action[layer] / layer_rms[layer]
        manual[layer, :, 2] = result.transpose_action[layer] / np.maximum(
            sigma[layer] * layer_rms[layer], generator.DENOMINATOR_FLOOR
        )
    np.clip(manual, -8.0, 8.0, out=manual)
    normalization_error = float(np.max(np.abs(manual - result.appended_features)))
    if normalization_error != 0.0:
        raise AssertionError(normalization_error)
    return {
        "maximum_cubic_float32_vs_float64_error": c_error,
        "maximum_transpose_float32_vs_float64_error": b_error,
        "maximum_forward_reverse_duality_error": maximum_duality_error,
        "maximum_normalization_replay_error": normalization_error,
    }


def permutation_equivariance_check() -> dict[str, float]:
    raw, sigma, weights, layer_rms = synthetic_row(0xE011A)
    base = generator.build_features(
        raw, sigma, weights, layer_rms, first_layer=0, last_layer=3
    )
    rng = np.random.default_rng(0xB1AEC7)
    matrices = []
    for _ in range(4):
        permutation = rng.permutation(5)
        matrices.append(np.eye(5, dtype=np.float32)[permutation])

    raw_p = np.empty_like(raw)
    sigma_p = np.empty_like(sigma)
    weights_p = np.empty_like(weights)
    for layer, matrix in enumerate(matrices):
        raw_p[layer] = matrix @ raw[layer]
        sigma_p[layer] = matrix @ sigma[layer]
        left = np.eye(5, dtype=np.float32) if layer == 0 else matrices[layer - 1]
        weights_p[layer] = left @ weights[layer] @ matrix.T
    permuted = generator.build_features(
        raw_p, sigma_p, weights_p, layer_rms, first_layer=0, last_layer=3
    )

    c_error = 0.0
    b_error = 0.0
    feature_error = 0.0
    for layer, matrix in enumerate(matrices):
        c_error = max(
            c_error,
            float(np.max(np.abs(permuted.cubic_action[layer] - matrix @ base.cubic_action[layer]))),
        )
        b_error = max(
            b_error,
            float(
                np.max(
                    np.abs(
                        permuted.transpose_action[layer]
                        - matrix @ base.transpose_action[layer]
                    )
                )
            ),
        )
        feature_error = max(
            feature_error,
            float(
                np.max(
                    np.abs(
                        permuted.appended_features[layer]
                        - matrix @ base.appended_features[layer]
                    )
                )
            ),
        )
    if c_error != 0.0 or b_error > 2.0e-7 or feature_error > 5.0e-7:
        raise AssertionError((c_error, b_error, feature_error))
    return {
        "maximum_cubic_equivariance_error": c_error,
        "maximum_transpose_equivariance_error": b_error,
        "maximum_appended_feature_equivariance_error": feature_error,
    }


def validation_negative_checks() -> dict[str, bool]:
    raw, sigma, weights, layer_rms = synthetic_row(7)
    checks: dict[str, bool] = {}
    cases = {
        "raw_dtype": lambda: generator.build_features(
            raw.astype(np.float64), sigma, weights, layer_rms, first_layer=0, last_layer=3
        ),
        "sigma_shape": lambda: generator.build_features(
            raw, sigma[:, :-1], weights, layer_rms, first_layer=0, last_layer=3
        ),
        "weights_shape": lambda: generator.build_features(
            raw, sigma, weights[:, :-1], layer_rms, first_layer=0, last_layer=3
        ),
        "layer_interval": lambda: generator.build_features(
            raw, sigma, weights, layer_rms, first_layer=3, last_layer=2
        ),
        "nonpositive_rms": lambda: generator.build_features(
            raw,
            sigma,
            weights,
            np.zeros_like(layer_rms),
            first_layer=0,
            last_layer=3,
        ),
    }
    for name, operation in cases.items():
        try:
            operation()
        except (TypeError, ValueError):
            checks[name] = True
        else:
            raise AssertionError(f"negative check did not fail: {name}")
    return checks


def main() -> None:
    no_io = assert_no_io_or_target_surface()
    algebra = algebra_check()
    equivariance = permutation_equivariance_check()
    negative = validation_negative_checks()
    census = generator.operation_census()
    expected_census = {
        "cubic_arithmetic_flops": 45056,
        "reverse_matvec_flops": 20961280,
        "reverse_elementwise_flops": 20480,
        "normalization_arithmetic_flops": 45056,
        "arithmetic_flops": 21071872,
        "comparison_operations": 78848,
        "arithmetic_plus_comparison_operations": 21150720,
        "c_and_b_bytes": 90112,
        "appended_feature_bytes": 135168,
        "scratch_and_denominator_bytes": 8192,
        "numpy_owned_peak_live_bytes": 233472,
    }
    if census != expected_census:
        raise AssertionError(census)
    receipt = {
        "schema": "public-full688-d11-cubic-action-transpose-feature-static-verification-v1",
        "status": "PASS_TINY_SYNTHETIC_ALGEBRA_DUALITY_PERMUTATION_EQUIVARIANCE_AND_NO_IO_AUDIT",
        "source_sha256": sha256(SOURCE),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "design_freeze_sha256": sha256(DESIGN_FREEZE),
        "source_audit": no_io,
        "algebra": algebra,
        "permutation_equivariance": equivariance,
        "negative_validation_checks": negative,
        "resource_census": census,
        "access": {
            "native_688_rows_opened": 0,
            "targets_opened_or_decoded": 0,
            "derived_feature_caches_created": 0,
            "training_or_scoring_runs": 0,
            "estimator_calls": 0,
            "network_or_submission_calls": 0,
        },
        "authority": {
            "target": False,
            "cache": False,
            "execution": False,
            "training": False,
            "remaining_folds": False,
            "integration": False,
            "meter": False,
            "board": False,
            "submission": False,
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
