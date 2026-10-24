#!/usr/bin/env python3
"""V1R2 exact-type pooled-H rank-hash receipt contract.

This module hash-verifies and reuses the immutable V1 reducer, while binding
the immutable V1R1 package and its durable exact-type HOLD.  It adds no
numerical law.  Import is inert; no future or protected path is opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import ModuleType
from typing import Any


HERE = Path(__file__).absolute().parent
P2 = HERE.parent
V1_ROOT = P2 / "public_q24_signed_pair_v4r1_all688_refit_deployment_prep_v1"
V1_CONTRACT = V1_ROOT / "deployment_contract.py"
V1_HOLD = (
    P2
    / "public_q24_signed_pair_v4r1_all688_refit_deployment_prep_review_v1"
    / "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"
)
V1R1_ROOT = P2 / "public_q24_signed_pair_v4r1_all688_refit_deployment_prep_v1r1"
V1R1_CONTRACT = V1R1_ROOT / "deployment_contract_v1r1.py"
V1R1_HOLD = (
    P2
    / "public_q24_signed_pair_v4r1_all688_refit_deployment_prep_review_v1r1"
    / "INDEPENDENT_STATIC_REVIEW_V1R1.json"
)
SCAFFOLD = HERE / "H_SELECTION_RECEIPT_SCAFFOLD_V1R2.json"

V1_CONTRACT_SHA256 = "4d2055350c0017448e34e3d9be83755a902c614a238ed57fef258ad9fa74b6ce"
V1_HOLD_SHA256 = "198f6d6c99145cbd4f7d60ed886477cec4d439da7c646780d05cdbe0281fd277"
V1_HOLD_STATUS = "HOLD_FROZEN_DEPLOYMENT_PREP__POOLED_H_RECEIPT_SCHEMA_INCOMPLETE"
V1R1_CONTRACT_SHA256 = "cc3bf2839ae4a3bcd471519f0fe27641c24a9643f78bceeea5e34376aa999ab8"
V1R1_HOLD_SHA256 = "40ce1a3ed6efc84037a7b8dc799169049532c058df0b0a3db9d3a37d5e2721ce"
V1R1_HOLD_STATUS = (
    "HOLD_FROZEN_POOLED_H_RANK_HASH_SCHEMA_REPAIR__NONEXACT_NUMERIC_TYPES_ACCEPTED"
)

HS = (16, 64, 256)
H_KEYS = {"16", "64", "256"}
FOLD_KEYS = {"0", "1", "2", "3"}
SCAFFOLD_SCHEMA = (
    "public-q24-signed-pair-v4r1-pooled-H-selection-receipt-scaffold-v1r2"
)
SCAFFOLD_STATUS = "HOLD_NULL_EXACT_TYPES_REVIEWED_OOF__H_NOT_SELECTED__NOT_A_RECEIPT"
MATERIALIZED_SCHEMA = "public-q24-signed-pair-v4r1-pooled-H-selection-receipt-v1r2"
MATERIALIZED_STATUS = (
    "EXACT_TYPES_H_SELECTED_FROM_REVIEWED_OOF_PASS__REFIT_REQUIRES_SEPARATE_AUTHORITY"
)
REDUCER = "CPython_3.11.5_math.fsum_in_fold_then_role_rank_order_divided_by_2064"

SCAFFOLD_KEYS = {
    "schema",
    "status",
    "authority",
    "candidate_H",
    "fold_order",
    "future_bindings",
    "observation_count_per_H",
    "pair_on_mse_vector_float64_le_sha256",
    "role_rank_int32_le_sha256",
    "pooled_pair_on_mse",
    "reducer",
    "role_rank_exact_multiplicity_per_H",
    "selected_H",
    "tie_tolerance",
}
MATERIALIZED_KEYS = (SCAFFOLD_KEYS - {"future_bindings"}) | {"bindings"}
BINDING_KEYS = {
    "oof_output_seal_sha256",
    "independent_oof_review_sha256",
    "selection_receipt_sha256",
}
AUTHORITY_KEYS = {"refit", "package", "submission"}
EVIDENCE_KEYS = {
    "candidate_H",
    "fold_order",
    "observation_count_per_H",
    "pair_on_mse_vector_float64_le_sha256",
    "role_rank_int32_le_sha256",
    "pooled_pair_on_mse",
    "role_rank_exact_multiplicity_per_H",
    "selected_H",
    "tie_tolerance",
}

_V1_MODULE: ModuleType | None = None


class RepairContractError(RuntimeError):
    """Fail-closed V1R2 receipt rejection."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_keys(value: Any, wanted: set[str], label: str) -> dict[str, Any]:
    if (
        type(value) is not dict
        or len(value) != len(wanted)
        or any(type(key) is not str for key in value)
        or set(value) != wanted
    ):
        raise RepairContractError(f"{label}: fields are not exact")
    return value


def exact_string(value: Any, wanted: str, label: str) -> str:
    if type(value) is not str or value != wanted:
        raise RepairContractError(f"{label}: must be exact builtin string")
    return value


def strict_int(value: Any, wanted: int, label: str) -> int:
    if type(value) is not int or value != wanted:
        raise RepairContractError(f"{label}: must be exact integer {wanted}")
    return value


def strict_int_list(value: Any, wanted: tuple[int, ...], label: str) -> list[int]:
    if (
        type(value) is not list
        or len(value) != len(wanted)
        or any(type(item) is not int for item in value)
        or value != list(wanted)
    ):
        raise RepairContractError(f"{label}: must be exact ordered integer list")
    return value


def strict_selected_h(value: Any, label: str) -> int:
    if type(value) is not int or value not in HS:
        raise RepairContractError(f"{label}: must be an exact candidate-H integer")
    return value


def strict_float(value: Any, label: str) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or (value == 0.0 and math.copysign(1.0, value) < 0.0)
    ):
        raise RepairContractError(
            f"{label}: must be an exact finite nonnegative builtin float"
        )
    return value


def same_binary64(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def strict_tie_tolerance(value: Any, label: str) -> float:
    item = strict_float(value, label)
    if not same_binary64(item, 0.0025):
        raise RepairContractError(f"{label}: must be exact binary64 0.0025")
    return item


def validate_pooled_map(
    value: Any,
    label: str,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, float]:
    mapping = exact_keys(value, H_KEYS, label)
    expected_mapping = None
    if expected is not None:
        expected_mapping = exact_keys(expected, H_KEYS, f"{label} expected")
    for h in HS:
        item = strict_float(mapping[str(h)], f"{label} H{h}")
        if expected_mapping is not None:
            wanted = strict_float(
                expected_mapping[str(h)], f"{label} expected H{h}"
            )
            if not same_binary64(item, wanted):
                raise RepairContractError(
                    f"{label}: H{h} differs at exact binary64 representation"
                )
    return mapping


def validate_selection_mapping(
    selections: Any,
) -> dict[int, dict[str, Any]]:
    if type(selections) is not dict or len(selections) != 4:
        raise RepairContractError("selection source must be an exact four-entry dict")
    if any(type(key) is not int for key in selections):
        raise RepairContractError("selection fold keys must be exact builtin integers")
    if set(selections) != {0, 1, 2, 3}:
        raise RepairContractError("selection source keys must be exactly folds 0..3")
    return selections


def validate_evidence_shape(value: Any, label: str) -> dict[str, Any]:
    evidence = exact_keys(value, EVIDENCE_KEYS, label)
    strict_int_list(evidence["candidate_H"], HS, f"{label} candidate H")
    strict_int_list(evidence["fold_order"], (0, 1, 2, 3), f"{label} fold order")
    strict_int(evidence["observation_count_per_H"], 2064, f"{label} count")
    strict_int(
        evidence["role_rank_exact_multiplicity_per_H"],
        3,
        f"{label} rank multiplicity",
    )
    strict_selected_h(evidence["selected_H"], f"{label} selected H")
    strict_tie_tolerance(evidence["tie_tolerance"], f"{label} tie tolerance")
    validate_sha_map(
        evidence["pair_on_mse_vector_float64_le_sha256"],
        f"{label} value-vector hashes",
        allow_null=False,
    )
    validate_sha_map(
        evidence["role_rank_int32_le_sha256"],
        f"{label} rank-vector hashes",
        allow_null=False,
    )
    validate_pooled_map(evidence["pooled_pair_on_mse"], f"{label} pooled MSE")
    return evidence


def validate_sha_map(
    value: Any, label: str, *, allow_null: bool
) -> dict[str, str | None]:
    mapping = exact_keys(value, H_KEYS, label)
    for h in HS:
        item = mapping[str(h)]
        if allow_null:
            if item is not None:
                raise RepairContractError(f"{label}: scaffold H{h} must be null")
        elif not is_sha256(item):
            raise RepairContractError(
                f"{label}: H{h} must be canonical lowercase SHA-256"
            )
    return mapping


def validate_selection_sha_map(
    value: Any, label: str, *, allow_null: bool
) -> dict[str, str | None]:
    mapping = exact_keys(value, FOLD_KEYS, label)
    for fold in range(4):
        item = mapping[str(fold)]
        if allow_null:
            if item is not None:
                raise RepairContractError(f"{label}: scaffold fold {fold} must be null")
        elif not is_sha256(item):
            raise RepairContractError(
                f"{label}: fold {fold} must be canonical lowercase SHA-256"
            )
    return mapping


def verify_immutable_predecessor() -> dict[str, str]:
    if sha256_file(V1_CONTRACT) != V1_CONTRACT_SHA256:
        raise RepairContractError("immutable V1 reducer contract changed")
    if sha256_file(V1_HOLD) != V1_HOLD_SHA256:
        raise RepairContractError("durable V1 HOLD changed")
    hold = json.loads(V1_HOLD.read_text(encoding="utf-8"))
    if hold.get("status") != V1_HOLD_STATUS:
        raise RepairContractError("durable V1 HOLD status changed")
    if sha256_file(V1R1_CONTRACT) != V1R1_CONTRACT_SHA256:
        raise RepairContractError("immutable V1R1 contract changed")
    if sha256_file(V1R1_HOLD) != V1R1_HOLD_SHA256:
        raise RepairContractError("durable V1R1 HOLD changed")
    hold_v1r1 = json.loads(V1R1_HOLD.read_text(encoding="utf-8"))
    if hold_v1r1.get("status") != V1R1_HOLD_STATUS:
        raise RepairContractError("durable V1R1 HOLD status changed")
    return {
        "v1_contract_sha256": V1_CONTRACT_SHA256,
        "v1_hold_sha256": V1_HOLD_SHA256,
        "v1r1_contract_sha256": V1R1_CONTRACT_SHA256,
        "v1r1_hold_sha256": V1R1_HOLD_SHA256,
    }


def v1_contract() -> ModuleType:
    global _V1_MODULE
    if _V1_MODULE is not None:
        return _V1_MODULE
    verify_immutable_predecessor()
    specification = importlib.util.spec_from_file_location(
        "q24_signed_pair_deployment_contract_v1_immutable", V1_CONTRACT
    )
    if specification is None or specification.loader is None:
        raise RepairContractError("could not load immutable V1 contract")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    _V1_MODULE = module
    return module


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def decode_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                RepairContractError(f"{label}: nonfinite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairContractError(f"{label}: invalid ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RepairContractError(f"{label}: not exact canonical object bytes")
    return value


def validate_scaffold(value: dict[str, Any]) -> None:
    exact_keys(value, SCAFFOLD_KEYS, "V1R2 H scaffold")
    exact_string(value["schema"], SCAFFOLD_SCHEMA, "V1R2 scaffold schema")
    exact_string(value["status"], SCAFFOLD_STATUS, "V1R2 scaffold status")
    strict_int_list(value["candidate_H"], HS, "V1R2 scaffold candidate H")
    strict_int_list(value["fold_order"], (0, 1, 2, 3), "V1R2 scaffold fold order")
    strict_int(value["observation_count_per_H"], 2064, "V1R2 scaffold count")
    strict_int(
        value["role_rank_exact_multiplicity_per_H"],
        3,
        "V1R2 scaffold rank multiplicity",
    )
    strict_tie_tolerance(value["tie_tolerance"], "V1R2 scaffold tie tolerance")
    exact_string(value["reducer"], REDUCER, "V1R2 scaffold reducer")
    if value["selected_H"] is not None:
        raise RepairContractError("V1R2 H scaffold selected H must be null")
    authority = exact_keys(value["authority"], AUTHORITY_KEYS, "scaffold authority")
    if any(item is not False for item in authority.values()):
        raise RepairContractError("V1R2 H scaffold grants authority")
    future = exact_keys(
        value["future_bindings"],
        BINDING_KEYS,
        "V1R2 H scaffold future bindings",
    )
    if future["oof_output_seal_sha256"] is not None:
        raise RepairContractError("scaffold OOF seal root must be null")
    if future["independent_oof_review_sha256"] is not None:
        raise RepairContractError("scaffold OOF review root must be null")
    validate_selection_sha_map(
        future["selection_receipt_sha256"],
        "scaffold selection receipt map",
        allow_null=True,
    )
    validate_sha_map(
        value["pair_on_mse_vector_float64_le_sha256"],
        "scaffold value-vector hash map",
        allow_null=True,
    )
    validate_sha_map(
        value["role_rank_int32_le_sha256"],
        "scaffold rank-vector hash map",
        allow_null=True,
    )
    pooled = exact_keys(value["pooled_pair_on_mse"], H_KEYS, "scaffold pooled MSE")
    if any(item is not None for item in pooled.values()):
        raise RepairContractError("scaffold pooled MSE values must be null")


def canonical_selection_hashes(
    selections: dict[int, dict[str, Any]]
) -> dict[str, str]:
    validate_selection_mapping(selections)
    return {
        str(fold): hashlib.sha256(
            canonical_json_bytes(selections[fold])
        ).hexdigest()
        for fold in range(4)
    }


def _computed_evidence(
    selections: dict[int, dict[str, Any]], *, require_pinned_runtime: bool
) -> dict[str, Any]:
    validate_selection_mapping(selections)
    v1 = v1_contract()
    try:
        pooled = v1.pooled_h_from_selections(
            selections, require_pinned_runtime=require_pinned_runtime
        )
    except Exception as error:
        raise RepairContractError(f"immutable V1 reducer rejected input: {error}") from error
    return validate_evidence_shape(pooled, "computed immutable-V1 evidence")


def _materialize_h_receipt(
    selections: dict[int, dict[str, Any]],
    *,
    oof_output_seal_sha256: str,
    independent_oof_review_sha256: str,
    require_pinned_runtime: bool,
) -> dict[str, Any]:
    if not is_sha256(oof_output_seal_sha256):
        raise RepairContractError("OOF output seal root is not canonical SHA-256")
    if not is_sha256(independent_oof_review_sha256):
        raise RepairContractError("independent OOF review root is not canonical SHA-256")
    pooled = _computed_evidence(
        selections, require_pinned_runtime=require_pinned_runtime
    )
    receipt = {
        "schema": MATERIALIZED_SCHEMA,
        "status": MATERIALIZED_STATUS,
        "authority": {"refit": False, "package": False, "submission": False},
        "bindings": {
            "oof_output_seal_sha256": oof_output_seal_sha256,
            "independent_oof_review_sha256": independent_oof_review_sha256,
            "selection_receipt_sha256": canonical_selection_hashes(selections),
        },
        "candidate_H": pooled["candidate_H"],
        "fold_order": pooled["fold_order"],
        "observation_count_per_H": pooled["observation_count_per_H"],
        "pair_on_mse_vector_float64_le_sha256": pooled[
            "pair_on_mse_vector_float64_le_sha256"
        ],
        "role_rank_int32_le_sha256": pooled["role_rank_int32_le_sha256"],
        "pooled_pair_on_mse": pooled["pooled_pair_on_mse"],
        "reducer": REDUCER,
        "role_rank_exact_multiplicity_per_H": pooled[
            "role_rank_exact_multiplicity_per_H"
        ],
        "selected_H": pooled["selected_H"],
        "tie_tolerance": pooled["tie_tolerance"],
    }
    validate_materialized_receipt(
        receipt,
        selections,
        require_pinned_runtime=require_pinned_runtime,
    )
    return receipt


def materialize_h_receipt(
    selections: dict[int, dict[str, Any]],
    *,
    oof_output_seal_sha256: str,
    independent_oof_review_sha256: str,
) -> dict[str, Any]:
    """Production materializer; exact CPython 3.11.5 is mandatory."""
    return _materialize_h_receipt(
        selections,
        oof_output_seal_sha256=oof_output_seal_sha256,
        independent_oof_review_sha256=independent_oof_review_sha256,
        require_pinned_runtime=True,
    )


def validate_materialized_receipt(
    receipt: dict[str, Any],
    selections: dict[int, dict[str, Any]],
    *,
    require_pinned_runtime: bool = True,
) -> None:
    exact_keys(receipt, MATERIALIZED_KEYS, "V1R2 materialized H receipt")
    exact_string(receipt["schema"], MATERIALIZED_SCHEMA, "materialized schema")
    exact_string(receipt["status"], MATERIALIZED_STATUS, "materialized status")
    authority = exact_keys(receipt["authority"], AUTHORITY_KEYS, "receipt authority")
    if any(item is not False for item in authority.values()):
        raise RepairContractError("materialized H receipt grants live authority")
    bindings = exact_keys(receipt["bindings"], BINDING_KEYS, "receipt bindings")
    if not is_sha256(bindings["oof_output_seal_sha256"]):
        raise RepairContractError("receipt OOF seal root is malformed")
    if not is_sha256(bindings["independent_oof_review_sha256"]):
        raise RepairContractError("receipt OOF review root is malformed")
    validate_selection_sha_map(
        bindings["selection_receipt_sha256"],
        "receipt selection source hash map",
        allow_null=False,
    )
    expected_selection_hashes = canonical_selection_hashes(selections)
    if bindings["selection_receipt_sha256"] != expected_selection_hashes:
        raise RepairContractError("receipt selection source hashes do not match inputs")

    strict_int_list(receipt["candidate_H"], HS, "materialized candidate H")
    strict_int_list(receipt["fold_order"], (0, 1, 2, 3), "materialized fold order")
    strict_int(receipt["observation_count_per_H"], 2064, "materialized count")
    strict_int(
        receipt["role_rank_exact_multiplicity_per_H"],
        3,
        "materialized rank multiplicity",
    )
    strict_selected_h(receipt["selected_H"], "materialized selected H")
    strict_tie_tolerance(receipt["tie_tolerance"], "materialized tie tolerance")
    exact_string(receipt["reducer"], REDUCER, "materialized reducer")
    value_hashes = validate_sha_map(
        receipt["pair_on_mse_vector_float64_le_sha256"],
        "receipt value-vector hash map",
        allow_null=False,
    )
    rank_hashes = validate_sha_map(
        receipt["role_rank_int32_le_sha256"],
        "receipt rank-vector hash map",
        allow_null=False,
    )
    computed = _computed_evidence(
        selections, require_pinned_runtime=require_pinned_runtime
    )
    if value_hashes != computed["pair_on_mse_vector_float64_le_sha256"]:
        raise RepairContractError("receipt value-vector hashes differ from reducer")
    if rank_hashes != computed["role_rank_int32_le_sha256"]:
        raise RepairContractError("receipt rank-vector hashes differ from reducer")
    pooled = validate_pooled_map(
        receipt["pooled_pair_on_mse"],
        "receipt pooled MSE",
    )
    validate_pooled_map(
        pooled,
        "receipt pooled MSE",
        expected=computed["pooled_pair_on_mse"],
    )
    if receipt["selected_H"] != computed["selected_H"]:
        raise RepairContractError("receipt selected H differs from reducer")


def validate_canonical_materialized_bytes(
    raw: bytes,
    selections: dict[int, dict[str, Any]],
    *,
    require_pinned_runtime: bool = True,
) -> dict[str, Any]:
    value = decode_canonical_json(raw, "V1R2 materialized H receipt")
    validate_materialized_receipt(
        value, selections, require_pinned_runtime=require_pinned_runtime
    )
    return value


def main() -> None:
    verify_immutable_predecessor()
    scaffold = json.loads(SCAFFOLD.read_text(encoding="utf-8"))
    validate_scaffold(scaffold)
    print(
        json.dumps(
            {
                "status": "PASS_V1R2_EXACT_TYPE_NULL_SCAFFOLD__NO_LIVE_AUTHORITY",
                "v1_contract_sha256": V1_CONTRACT_SHA256,
                "v1_hold_sha256": V1_HOLD_SHA256,
                "v1r1_contract_sha256": V1R1_CONTRACT_SHA256,
                "v1r1_hold_sha256": V1R1_HOLD_SHA256,
                "role_rank_int32_le_sha256": scaffold[
                    "role_rank_int32_le_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
