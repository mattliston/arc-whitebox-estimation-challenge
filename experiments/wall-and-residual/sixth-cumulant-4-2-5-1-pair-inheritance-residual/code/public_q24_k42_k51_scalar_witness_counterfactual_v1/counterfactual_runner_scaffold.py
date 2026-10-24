#!/usr/bin/env python3
"""Non-executing scaffold for the K42+K51 scalar-witness counterfactual.

There is intentionally no row loader, rank selector, release parser, worker,
Estimator call, or execution CLI in this file.  The only command emits a
static HOLD receipt or runs allocation-tiny sink invariants.  The terminal
rank-201 closure and canonical graph forbid treating this instrumentation
rewrite as a retry or live successor.  The dormant source itself fails closed
before every base import, installation, or prediction path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "k42_k51_scalar_witness_source.py"
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
VERIFIER = HERE / "verify_static.py"
STATIC_RESULT = HERE / "STATIC_VERIFICATION_V1.json"
FREEZE = HERE / "SOURCE_FREEZE_V1.json"
PARENT_PACKAGE = HERE.parent / "public_q24_k42_k51_pair_inheritance_v1"
PARENT_CLOSURE = PARENT_PACKAGE / "K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json"
CANONICAL_GRAPH = HERE.parents[2] / "graph.json"

PARENT_CLOSURE_SHA256 = "3334080f68af40fa02509639e7c9a6ebc87c8888c30b21aea399d9362bae793d"
CANONICAL_GRAPH_SHA256 = "84ff484740ec9390d61272e9807784396db9d944abcc84b2d2490e4e23f47e76"
SOURCE_SHA256 = "e0cfc3c20d2b48e0ea5345580f5f76410a92371d715ba6b61877dc83538b4496"

STATUS = "HOLD_USER_SUPERSESSION_REQUIRED__COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE"
SOURCE_GUARDED_ENTRYPOINTS = ("_load_base", "_install", "predict_with_pair_tape")
SOURCE_GUARD_MESSAGE = "COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE"
COORDINATE_SHAPE = (15, 1024)
PLANE_WITNESS_KEYS = {
    "observed_dtype",
    "shape",
    "minimum",
    "maximum",
    "diagonal_nonzero_count",
}
ACTION_WITNESS_KEYS = {
    "sector_observed_dtype",
    "combined_observed_dtype",
    "sector_shape",
    "combined_shape",
    "sector_minimum",
    "sector_maximum",
    "exact_uint64_duplicate_sum",
}
PUBLIC_COMPONENTS = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
ADDED_WITNESS_CALLS = {
    "min": 30,
    "max": 30,
    "diagonal": 15,
    "count_nonzero": 15,
    "add": 15,
    "array_equal": 15,
}
COUNTERFACTUAL_OPERATION_CALL_DELTA = {
    "empty": 16,
    "add": 180,
    "array_equal": 15,
    "concatenate": 30,
    "copy": 15,
    "count_nonzero": 15,
    "getitem": 43,
    "max": 30,
    "min": 30,
    "multiply": 1020,
    "reshape": 15,
    "stack": 45,
    "subtract": 210,
    "einsum": 29,
    "diagonal": 30,
    "fill_diagonal": 30,
    "transpose": 15,
    "copyto": 30,
    "matmul": 15,
    "sum": 15,
}
RETAINED_COMBINED_VECTOR_BANK_BYTES = 15 * 1024 * 8
FINAL_COMBINED_BANK_BYTES = 15 * 1024 * 8
SIMULTANEOUS_CLIENT_COMBINED_BANK_BYTES = (
    RETAINED_COMBINED_VECTOR_BANK_BYTES + FINAL_COMBINED_BANK_BYTES
)
PARENT_D4_PLUS_REMOTE_SCIENTIFIC_SCRATCH_BYTES = 7_043_162_112
COUNTERFACTUAL_PROJECTED_PEAK_RSS_BYTES = (
    PARENT_D4_PLUS_REMOTE_SCIENTIFIC_SCRATCH_BYTES
    + SIMULTANEOUS_CLIENT_COMBINED_BANK_BYTES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_content_sha256(array: Any) -> str:
    view = memoryview(array).cast("B")
    return hashlib.sha256(view).hexdigest()


def _exact_float_list(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"{label} is not a fixed scalar list")
    answer: list[float] = []
    for item in value:
        if type(item) not in (float, int):
            raise RuntimeError(f"{label} contains a non-scalar value")
        answer.append(float(item))
    return answer


def validate_plane_witness(value: Any) -> dict[str, Any]:
    """Reject arrays and derive the old hard gates from scalar extrema."""

    if not isinstance(value, dict) or set(value) != PLANE_WITNESS_KEYS:
        raise RuntimeError("plane witness schema changed")
    if value["observed_dtype"] != "<f8" or value["shape"] != [2, 1024, 1024]:
        raise RuntimeError("plane witness dtype/shape changed")
    minima = _exact_float_list(value["minimum"], 2, "plane minima")
    maxima = _exact_float_list(value["maximum"], 2, "plane maxima")
    diagonal_nonzero = value["diagonal_nonzero_count"]
    if type(diagonal_nonzero) is not int or diagonal_nonzero < 0:
        raise RuntimeError("plane diagonal witness is not a nonnegative scalar")
    finite = [math.isfinite(lo) and math.isfinite(hi) for lo, hi in zip(minima, maxima)]
    nonzero = [lo < 0.0 or hi > 0.0 for lo, hi in zip(minima, maxima)]
    return {
        "observed_dtype": "<f8",
        "shape": [2, 1024, 1024],
        "minimum": minima,
        "maximum": maxima,
        "finite_via_extrema": finite,
        "nonzero_via_extrema": nonzero,
        "diagonal_nonzero_count": diagonal_nonzero,
        "exact_two_zero_diagonals": diagonal_nonzero == 0,
    }


def validate_action_witness(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ACTION_WITNESS_KEYS:
        raise RuntimeError("action witness schema changed")
    if (
        value["sector_observed_dtype"] != "<f8"
        or value["combined_observed_dtype"] != "<f8"
        or value["sector_shape"] != [2, 1024]
        or value["combined_shape"] != [1024]
        or value["exact_uint64_duplicate_sum"] is not True
    ):
        raise RuntimeError("action witness dtype/shape/sum changed")
    minima = _exact_float_list(value["sector_minimum"], 2, "sector minima")
    maxima = _exact_float_list(value["sector_maximum"], 2, "sector maxima")
    return {
        **value,
        "sector_minimum": minima,
        "sector_maximum": maxima,
        "sector_finite_via_extrema": [
            math.isfinite(lo) and math.isfinite(hi)
            for lo, hi in zip(minima, maxima)
        ],
        "sector_nonzero_via_extrema": [
            lo < 0.0 or hi > 0.0 for lo, hi in zip(minima, maxima)
        ],
    }


class ScalarWitnessSink:
    """Retain scalar plane receipts and the one existing combined bank only."""

    def __init__(self) -> None:
        self.sources: dict[int, dict[str, Any]] = {}
        self.actions: dict[int, dict[str, Any]] = {}
        self.combined: dict[int, Any] = {}

    def capture_source(self, layer: int, witness: Any) -> None:
        if layer in self.sources:
            raise RuntimeError(f"duplicate source witness at h{layer}")
        self.sources[layer] = validate_plane_witness(witness)

    def capture_action(
        self,
        layer: int,
        _raw_by_sector_remote: Any,
        raw_total_remote: Any,
        witness: Any,
    ) -> None:
        import numpy as np

        if layer in self.actions or layer in self.combined:
            raise RuntimeError(f"duplicate action witness at h{layer}")
        action = validate_action_witness(witness)
        combined = np.asarray(raw_total_remote)
        if combined.dtype != np.dtype("<f8") or combined.shape != (1024,):
            raise RuntimeError("combined raw-k6 dtype/shape changed before copy")
        combined = np.ascontiguousarray(combined)
        action["combined_finite"] = bool(np.isfinite(combined).all())
        action["combined_nonzero"] = bool(np.count_nonzero(combined))
        action["combined_content_sha256"] = array_content_sha256(combined)
        self.actions[layer] = action
        self.combined[layer] = combined

    def finalize(self) -> tuple[list[dict[str, Any]], Any]:
        import numpy as np

        expected = set(range(1, 16))
        if set(self.sources) != expected or set(self.actions) != expected or set(self.combined) != expected:
            raise RuntimeError("scalar-witness layer set is not exactly h1:h15")
        bank = np.empty(COORDINATE_SHAPE, dtype=np.dtype("<f8"))
        layers: list[dict[str, Any]] = []
        for slot, layer in enumerate(range(1, 16)):
            bank[slot] = self.combined[layer]
            layers.append(
                {
                    "slot": slot,
                    "source_layer_one_based": layer,
                    "outgoing_weight_index_zero_based": layer,
                    "consumer_layer_one_based": layer + 1,
                    "source_witness": self.sources[layer],
                    "action_witness": self.actions[layer],
                }
            )
        self.sources.clear()
        self.actions.clear()
        self.combined.clear()
        # ``bank`` was allocated C-order by np.empty and is returned directly;
        # no third coordinate-bank copy can overlap the retained-vector bank.
        return layers, bank


def counterfactual_selftest() -> dict[str, bool]:
    plane = {
        "observed_dtype": "<f8",
        "shape": [2, 1024, 1024],
        "minimum": [-2.0, -3.0],
        "maximum": [4.0, 5.0],
        "diagonal_nonzero_count": 0,
    }
    action = {
        "sector_observed_dtype": "<f8",
        "combined_observed_dtype": "<f8",
        "sector_shape": [2, 1024],
        "combined_shape": [1024],
        "sector_minimum": [-1.0, -2.0],
        "sector_maximum": [3.0, 4.0],
        "exact_uint64_duplicate_sum": True,
    }
    plane_result = validate_plane_witness(plane)
    action_result = validate_action_witness(action)
    rejected_array = False
    try:
        validate_plane_witness([[0.0]])
    except RuntimeError:
        rejected_array = True
    return {
        "scalar_plane_witness_accepted": all(plane_result["finite_via_extrema"]),
        "scalar_plane_nonzero_equivalent": all(plane_result["nonzero_via_extrema"]),
        "zero_diagonal_equivalent": plane_result["exact_two_zero_diagonals"],
        "scalar_action_witness_accepted": all(action_result["sector_finite_via_extrema"]),
        "scalar_action_nonzero_equivalent": all(action_result["sector_nonzero_via_extrema"]),
        "full_plane_payload_rejected": rejected_array,
        "simultaneous_client_bank_lifetime_exact": (
            RETAINED_COMBINED_VECTOR_BANK_BYTES == 122_880
            and FINAL_COMBINED_BANK_BYTES == 122_880
            and SIMULTANEOUS_CLIENT_COMBINED_BANK_BYTES == 245_760
            and COUNTERFACTUAL_PROJECTED_PEAK_RSS_BYTES == 7_043_407_872
        ),
    }


def static_hold_payload() -> dict[str, Any]:
    if sha256_file(PARENT_CLOSURE) != PARENT_CLOSURE_SHA256:
        raise RuntimeError("parent terminal closure changed")
    if sha256_file(CANONICAL_GRAPH) != CANONICAL_GRAPH_SHA256:
        raise RuntimeError("canonical no-successor graph changed")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("counterfactual source changed")
    forbidden_runtime_artifacts = sorted(
        path.name
        for path in HERE.iterdir()
        if (
            "RELEASE" in path.name
            or "RUN.lock" in path.name
            or "BUNDLE" in path.name
            or "WORKER_CONSUMED" in path.name
        )
    )
    if forbidden_runtime_artifacts:
        raise RuntimeError(f"forbidden runtime artifact exists: {forbidden_runtime_artifacts}")
    return {
        "schema": "q24-k42-k51-scalar-witness-counterfactual-hold-v1",
        "status": STATUS,
        "parent_terminal_closure_sha256": PARENT_CLOSURE_SHA256,
        "canonical_graph_sha256": CANONICAL_GRAPH_SHA256,
        "source_sha256": SOURCE_SHA256,
        "added_witness_flops": 126_074_640,
        "counterfactual_incremental_flops": 151_951_499_024,
        "hard_cap_flops": 152_200_000_000,
        "cap_headroom_flops": 248_500_976,
        "added_witness_calls": ADDED_WITNESS_CALLS,
        "counterfactual_operation_call_delta": COUNTERFACTUAL_OPERATION_CALL_DELTA,
        "resource_projection": {
            "retained_combined_vector_bank_bytes": RETAINED_COMBINED_VECTOR_BANK_BYTES,
            "final_combined_bank_bytes": FINAL_COMBINED_BANK_BYTES,
            "simultaneous_client_combined_bank_bytes": SIMULTANEOUS_CLIENT_COMBINED_BANK_BYTES,
            "counterfactual_projected_peak_rss_bytes": COUNTERFACTUAL_PROJECTED_PEAK_RSS_BYTES,
        },
        "source_execution_guard_contract": {
            "entrypoints": list(SOURCE_GUARDED_ENTRYPOINTS),
            "exact_direct_runtime_error_message": SOURCE_GUARD_MESSAGE,
            "guard_precedes_base_import_install_and_predict": True,
            "verified_statically_not_executed": True,
        },
        "full_public_parity_contract": {
            "components": list(PUBLIC_COMPONENTS),
            "required_only_after_explicit_user_supersession": True,
            "A1_B_A2_byte_parity_required": True,
            "claimed_or_executed_here": False,
        },
        "rank_selection_law_present": False,
        "release_path_present": False,
        "execute_path_present": False,
        "forbidden_runtime_artifacts": forbidden_runtime_artifacts,
        "firewall": {
            "row_opened": False,
            "weights_opened": False,
            "estimator_imported": False,
            "estimator_called": False,
            "target_or_score_read": False,
            "release_created": False,
            "rank_selected": False,
            "graph_node_created": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    payload: Any = counterfactual_selftest() if args.selftest else static_hold_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
