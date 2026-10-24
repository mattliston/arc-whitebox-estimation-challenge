#!/Users/matt/anaconda3/bin/python3
"""Target/cache-free static verification for the frozen fold-0 runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_fold0.py"
ADAPTER = HERE / "feature_generator.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    sys.path.insert(0, str(HERE))
    adapter = load("owned_four_adapter_static", ADAPTER)
    runner = load("owned_four_runner_static", RUNNER)
    rng = np.random.default_rng(120629887)
    depth, width = 16, 8
    raw = rng.normal(size=(depth, width, 20)).astype(np.float32)
    raw[:, :, 5] = np.clip(rng.uniform(size=(depth, width)), 0.05, 0.95)
    raw[:, :, 6] = np.abs(raw[:, :, 6]) + np.float32(0.02)
    sigma = (np.abs(rng.normal(size=(depth, width))) + 0.5).astype(np.float32)
    weights = rng.normal(size=(depth, width, width)).astype(np.float32)
    layer_rms = (np.abs(rng.normal(size=depth)) + 0.5).astype(np.float32)
    built = adapter.build_features(raw, sigma, weights, layer_rms)
    if built.layers != tuple(range(10, 16)) or built.appended_features.shape != (6, 8, 4):
        raise AssertionError("adapter layer/shape contract failed")

    # Independent source/target relabeling at every layer.
    permutations = [rng.permutation(width) for _ in range(depth)]
    raw_p = np.empty_like(raw)
    sigma_p = np.empty_like(sigma)
    weights_p = np.empty_like(weights)
    for layer in range(depth):
        raw_p[layer] = raw[layer, permutations[layer]]
        sigma_p[layer] = sigma[layer, permutations[layer]]
        if layer == 0:
            weights_p[layer] = weights[layer][permutations[layer]][:, permutations[layer]]
        else:
            weights_p[layer] = weights[layer][permutations[layer - 1]][:, permutations[layer]]
    built_p = adapter.build_features(raw_p, sigma_p, weights_p, layer_rms)
    permutation_error = max(
        float(
            np.max(
                np.abs(
                    built_p.appended_features[slot]
                    - built.appended_features[slot, permutations[layer]]
                )
            )
        )
        for slot, layer in enumerate(range(10, 16))
    )
    if permutation_error > 3.0e-6:
        raise AssertionError(f"permutation equivariance failed: {permutation_error}")

    base = type("DummyBase", (), {})()
    base.centers = torch.zeros(DEPTH := 16, 19, dtype=torch.float32)
    base.scales = torch.ones(DEPTH, 19, dtype=torch.float32)
    base.layer_rms = torch.ones(DEPTH, dtype=torch.float32)
    model = runner.MessageHead86(
        base,
        np.zeros((6, 4), dtype=np.float32),
        np.ones((6, 4), dtype=np.float32),
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 155_398:
        raise AssertionError(parameter_count)
    with torch.no_grad():
        model.head_weight[0].fill_(0.01)
        model.head_bias.zero_()
    batch = 11
    inputs = (
        torch.from_numpy(rng.normal(size=(batch, 20)).astype(np.float32)),
        torch.ones(batch, dtype=torch.float32),
        torch.from_numpy(rng.normal(size=batch).astype(np.float32)),
        torch.from_numpy(rng.normal(size=batch).astype(np.float32)),
    )
    parent_a = torch.from_numpy(rng.normal(size=(batch, 60)).astype(np.float32))
    parent_b = parent_a + 0.25
    four_a = torch.from_numpy(rng.normal(size=(batch, 4)).astype(np.float32))
    four_b = four_a + 0.5
    zero_a = model.source(*inputs, parent_a, four_a, 10, source_on=False)
    zero_b = model.source(*inputs, parent_a, four_b, 10, source_on=False)
    zero_parent_b = model.source(*inputs, parent_b, four_a, 10, source_on=False)
    on_a = model.source(*inputs, parent_a, four_a, 10, source_on=True)
    on_b = model.source(*inputs, parent_a, four_b, 10, source_on=True)
    if not torch.equal(zero_a, zero_b):
        raise AssertionError("zero arm did not zero only the normalized new4")
    if torch.equal(zero_a, zero_parent_b):
        raise AssertionError("zero arm incorrectly zeroed the sealed parent60")
    if torch.equal(on_a, on_b):
        raise AssertionError("source-on arm is insensitive to new4")

    threshold = runner.PARENT_FOLD0_ENDPOINT - runner.MIN_GAP_RECOVERY * (
        runner.PARENT_FOLD0_ENDPOINT - runner.PARENT_FOLD0_ORACLE
    )
    if threshold != runner.ENDPOINT_THRESHOLD:
        raise AssertionError("frozen threshold arithmetic changed")
    census = adapter._FROZEN.operation_census(
        width=1024, seams=6, resident_squared_weights=False
    )
    output = {
        "schema": "owned-l3-l3square-fold0-static-verification-v1",
        "status": "PASS_TARGET_AND_CACHE_FREE_STATIC_VERIFICATION",
        "bindings": {
            "runner_sha256": sha256(RUNNER),
            "feature_adapter_sha256": sha256(ADAPTER),
            "frozen_generator_sha256": adapter.FROZEN_SOURCE_SHA256,
        },
        "architecture": {
            "inputs": 86,
            "local22": 22,
            "sealed_parent_messages": 60,
            "appended_four_fields": 4,
            "hidden": 256,
            "hidden_layers": 3,
            "heads": 6,
            "parameters": parameter_count,
        },
        "checks": {
            "adapter_shape_and_order": True,
            "permutation_equivariance_max_abs": permutation_error,
            "zero_arm_ignores_only_new4_after_normalization": True,
            "zero_arm_retains_parent60": True,
            "source_on_uses_new4": True,
            "fixed_parent82_gap_threshold": threshold,
            "fallback_four_field_arithmetic_flops_per_network": census[
                "arithmetic_flops"
            ],
            "fallback_four_field_gemm_equivalents_per_network": census[
                "all_arithmetic_gemm_equivalents"
            ],
            "fallback_four_field_explicit_peak_bytes_per_network": census[
                "explicit_peak_live_bytes"
            ],
        },
        "access": {
            "target_values_opened_or_decoded": 0,
            "target_blind_cache_rows_opened": 0,
            "training_or_scoring_runs": 0,
            "derived_feature_caches_created": 0,
            "estimator_calls": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

