#!/Users/matt/anaconda3/bin/python3
"""One-shot nested signed-pair accuracy experiment.

The checked-in release and capability are HOLD scaffolds.  Preflight performs
no row, weight, bank, or target decode.  Execution requires a new independent
GO receipt, exact release, exact one-use capability, and three already sealed
target-blind banks.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREG = HERE / "PREREGISTRATION.json"
STATIC = HERE / "STATIC_VERIFICATION.json"
RELEASE_SCAFFOLD = HERE / "EXECUTION_RELEASE_SCAFFOLD.json"
CAPABILITY_SCAFFOLD = HERE / "TARGET_CAPABILITY_SCAFFOLD.json"
BANK_AUTHORITY_CONTRACT = HERE / "BANK_SOURCE_AUTHORITY_CONTRACT.json"
NOVELTY_ALLOWLIST_CONTRACT = HERE / "NOVELTY_REFERENCE_ALLOWLIST_CONTRACT.json"
BUNDLE = HERE / "SIGNED_PAIR_NESTED_ACCURACY_OUTPUT_V1"
CLAIM = HERE / "SIGNED_PAIR_NESTED_ACCURACY_RUN_CLAIM_V1.json"
FAILURE = HERE / "SIGNED_PAIR_NESTED_ACCURACY_TERMINAL_FAILURE_V1.json"

ACTION_V2 = P2 / "public_q24_signed_pair_action_state_v2"
NATIVE22 = P2 / "public_q24_signed_pair_qinnovation_distillation_v1"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
RAW_STATS = P2 / "public_full688_dynamic_multistate_residual_cv_v1/cache/raw_sufficient_stats.npz"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FEATURE_MANIFEST = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/manifest.json"
)

PARENT_HASHES = {
    ACTION_V2 / "PACKAGE_MANIFEST.json": "85c1f3073ef54fa822748c8c579bd1d68d0823bb1169014f92d6e05152c5f895",
    ACTION_V2 / "FREEZE_VERIFICATION.json": "e0846f84c452a1d99e3e296dc4755a28f2b985e18b7e5849414a4d64d4c47490",
    ACTION_V2 / "STATIC_VERIFICATION.json": "c726d844fab90ffca968e53ded72dd19d30a73c169ca51fb2015eb64dd1a3741",
    ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json": "2a9270595bb33c0fd0f17a347131f02358a2aee00f0f38c6ee9e8484b0c45a0b",
    NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json": "5c940c501c383fe65e9e4b5dda103ba287797d0cb18ed22bc6314588f37245fd",
    NATIVE22 / "NATIVE22_TERMINAL_AUDIT_V1.json": "5f3bcd094257a644c2a9ecb243d6bc357634b711be8d40ee3089a1d75cd8a62f",
    J_SOURCE: "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810",
    QREF_SOURCE: "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2",
    RAW_STATS: "1648a140768cdc254a7bb6957dcb8e82f4271118bc64b552bb08de294ec62208",
    FEATURE_MANIFEST: "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50",
}
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ROWS = 688
WIDTH = 1024
DEPTH = 16
OUTER_FOLDS = 4
OUTER_FIT = 516
OUTER_HELD = 172
INNER_FOLDS = 3
INNER_FIT = 344
INNER_HELD = 172
PAIR_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
ACTION_LAYERS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
HS = (16, 64, 256)
H_TAG = {16: 0, 64: 65_536, 256: 131_072}
J_SEED = 1_753_358_438
PAIR_SEED = 1_396_789_298
J_EPOCHS = 6
PAIR_EPOCHS = 10
RAW_HURDLE = 1.2425668176231516e-8
TIE_TOLERANCE = 0.0025
ACTIVE_CLAIM_NONCE: str | None = None
ACTIVE_TEMP_BUNDLE: Path | None = None


class AuthorityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def strict_int(value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthorityError("expected non-bool JSON integer")
    if minimum is not None and value < minimum:
        raise AuthorityError("integer below bound")
    if maximum is not None and value > maximum:
        raise AuthorityError("integer above bound")
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuthorityError(f"JSON object required: {path}")
    return value


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


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_exclusive_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def observation(label: str, started: float | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": label,
        "maximum_process_rss_bytes_observational_only": resident_bytes(),
    }
    if started is not None:
        record["elapsed_seconds_observational_only"] = time.perf_counter() - started
    return record


def verify_static_parents(*, include_target_bundle: bool) -> None:
    for path, expected in PARENT_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise AuthorityError(f"frozen predecessor changed: {path}")
    if include_target_bundle and (not TARGETS.is_file() or sha256(TARGETS) != TARGETS_SHA256):
        raise AuthorityError("observer-fit target bundle bytes changed")
    action_review = read_json(ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json")
    if action_review.get("status") != "GO_STATIC_REPAIR__EXECUTION_AND_DATA_ACCESS_HOLD":
        raise AuthorityError("signed-pair V2 independent predecessor is not the exact GO")
    native_result = read_json(NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json")
    native_audit = read_json(NATIVE22 / "NATIVE22_TERMINAL_AUDIT_V1.json")
    if (
        native_result.get("status") != "FAIL_NATIVE22_FROZEN_CAPACITY__NO_RETRY"
        or native_audit.get("status") != "GO_ACCEPT_CLOSE"
        or native_audit.get("result_sha256") != PARENT_HASHES[
            NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json"
        ]
    ):
        raise AuthorityError("native22 terminal predecessor changed")


def resolve_input(path_text: str, relative_to: Path) -> Path:
    if not isinstance(path_text, str) or not path_text:
        raise AuthorityError("nonempty input path required")
    path = Path(path_text)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def validate_release(path: Path) -> dict[str, Any]:
    release = read_json(path)
    if (
        release.get("schema") != "public-q24-signed-pair-nested-accuracy-execution-release-v1"
        or release.get("status") != "RELEASE_EXACTLY_ONE_NESTED_ACCURACY_RUN"
        or release.get("role") != "observer_fit"
        or release.get("target_columns") != ["mlp_id", "all_layer_means"]
    ):
        raise AuthorityError("no exact nested-accuracy release")
    for key in (
        "observer_selection_targets",
        "names",
        "scores",
        "board",
        "network",
        "package",
        "submission",
    ):
        if release.get(key) is not False:
            raise AuthorityError(f"release firewall changed: {key}")
    bindings = release.get("bindings")
    if not isinstance(bindings, dict):
        raise AuthorityError("release bindings missing")
    expected = {
        "preregistration_sha256": sha256(PREREG),
        "runner_sha256": sha256(Path(__file__)),
        "static_verification_sha256": sha256(STATIC),
        "target_bundle_sha256": TARGETS_SHA256,
        "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
        "outer_fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
        "signed_pair_v2_manifest_sha256": PARENT_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"],
        "signed_pair_v2_independent_review_sha256": PARENT_HASHES[
            ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"
        ],
        "native22_result_sha256": PARENT_HASHES[
            NATIVE22 / "NATIVE22_CAPACITY_OUTPUT_V1/RESULT.json"
        ],
        "native22_terminal_audit_sha256": PARENT_HASHES[
            NATIVE22 / "NATIVE22_TERMINAL_AUDIT_V1.json"
        ],
        "bank_source_authority_contract_sha256": sha256(BANK_AUTHORITY_CONTRACT),
        "novelty_reference_allowlist_contract_sha256": sha256(
            NOVELTY_ALLOWLIST_CONTRACT
        ),
    }
    for key, expected_value in expected.items():
        if bindings.get(key) != expected_value:
            raise AuthorityError(f"release binding mismatch: {key}")
    for key in (
        "independent_review_sha256",
        "bank_generation_receipt_sha256",
        "bank_source_authority_sha256",
        "bank_source_authority_review_sha256",
        "ordered_688_record_sha256",
        "single_use_target_capability_sha256",
    ):
        if not is_sha(bindings.get(key)):
            raise AuthorityError(f"release SHA binding absent: {key}")
    for map_key in ("complete_bank_sha256", "A15_zero_sha256"):
        mapping = bindings.get(map_key)
        if not isinstance(mapping, dict) or set(mapping) != {"16", "64", "256"}:
            raise AuthorityError(f"release H hash map changed: {map_key}")
        if any(not is_sha(value) for value in mapping.values()):
            raise AuthorityError(f"release H hash invalid: {map_key}")
    review_path = resolve_input(release.get("independent_review_path"), path.parent)
    review = read_json(review_path)
    if (
        sha256(review_path) != bindings["independent_review_sha256"]
        or review.get("status") != "GO_SIGNED_PAIR_NESTED_ACCURACY_OOF_ONE_SHOT"
        or review.get("preregistration_sha256") != sha256(PREREG)
        or review.get("runner_sha256") != sha256(Path(__file__))
        or review.get("static_verification_sha256") != sha256(STATIC)
    ):
        raise AuthorityError("independent review is not a GO for exact package bytes")
    receipt_path = resolve_input(release.get("bank_generation_receipt_path"), path.parent)
    bank_root = resolve_input(release.get("bank_root"), path.parent)
    bank_authority_path = resolve_input(
        release.get("bank_source_authority_path"), path.parent
    )
    bank_authority_review_path = resolve_input(
        release.get("bank_source_authority_review_path"), path.parent
    )
    capability_path = resolve_input(release.get("single_use_target_capability_path"), path.parent)
    release["_resolved"] = {
        "release_parent": str(path.parent.resolve()),
        "review_path": str(review_path),
        "receipt_path": str(receipt_path),
        "bank_root": str(bank_root),
        "bank_authority_path": str(bank_authority_path),
        "bank_authority_review_path": str(bank_authority_review_path),
        "capability_path": str(capability_path),
    }
    return release


def validate_capability(path: Path, release_path: Path, release: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    bindings = release["bindings"]
    if hashlib.sha256(raw).hexdigest() != bindings["single_use_target_capability_sha256"]:
        raise AuthorityError("target capability hash mismatch")
    capability = json.loads(raw.decode("ascii"))
    if not isinstance(capability, dict):
        raise AuthorityError("capability object required")
    if (
        capability.get("schema") != "public-q24-signed-pair-nested-accuracy-target-capability-v1"
        or capability.get("status") != "AUTHORIZE_EXACTLY_ONE_OBSERVER_FIT_NESTED_ACCURACY_RUN"
        or capability.get("release_sha256") != sha256(release_path)
        or capability.get("bank_generation_receipt_sha256")
        != bindings["bank_generation_receipt_sha256"]
        or capability.get("ordered_688_record_sha256")
        != bindings["ordered_688_record_sha256"]
        or capability.get("target_bundle_sha256") != TARGETS_SHA256
        or capability.get("outer_fold_assignment_uint8_sha256") != FOLD_ASSIGNMENT_SHA256
        or capability.get("role") != "observer_fit"
        or capability.get("target_columns") != ["mlp_id", "all_layer_means"]
        or capability.get("one_use") is not True
        or capability.get("retry") is not False
    ):
        raise AuthorityError("narrow target capability contract mismatch")
    for key in ("capability_id", "issued_by", "issued_utc"):
        if not isinstance(capability.get(key), str) or not capability[key]:
            raise AuthorityError(f"capability provenance absent: {key}")
    for key in (
        "observer_selection_targets",
        "names",
        "scores",
        "board",
        "network",
        "package",
        "submission",
    ):
        if capability.get(key) is not False:
            raise AuthorityError(f"capability firewall changed: {key}")
    return raw, capability


def consume_capability(path: Path, raw: bytes) -> Path:
    consumed = path.with_name(path.name + ".consumed")
    if consumed.exists():
        raise FileExistsError(consumed)
    os.link(path, consumed)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
        if consumed.read_bytes() != raw:
            raise AuthorityError("consumed capability bytes changed")
        path.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.exists() or consumed.read_bytes() != raw:
        raise AuthorityError("capability was not consumed exactly once")
    return consumed


@dataclass(frozen=True)
class BankAuthority:
    receipt_path: Path
    receipt_sha256: str
    ordered_record_sha256: str
    mlp_ids: np.ndarray
    bank_paths: dict[int, Path]
    zero_paths: dict[int, Path]
    resource_observations: Any
    novelty_diagnostics: Any


def contained_regular_file(root: Path, relative: Any, exact_bytes: int) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AuthorityError("bank relative path missing")
    if Path(relative).is_absolute():
        raise AuthorityError("bank artifact path must be relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise AuthorityError("bank artifact path escape")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != exact_bytes:
        raise AuthorityError("bank artifact type or byte count mismatch")
    return path


def validate_external_bank_source_authority(
    release: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    """Cross-bind receipt provenance to a reviewed pre-generation allowlist."""
    bindings = release["bindings"]
    authority_path = Path(release["_resolved"]["bank_authority_path"])
    review_path = Path(release["_resolved"]["bank_authority_review_path"])
    if authority_path.is_symlink() or review_path.is_symlink():
        raise AuthorityError("bank source authority may not be symlinked")
    if (
        not authority_path.is_file()
        or sha256(authority_path) != bindings["bank_source_authority_sha256"]
        or not review_path.is_file()
        or sha256(review_path) != bindings["bank_source_authority_review_sha256"]
    ):
        raise AuthorityError("external bank source authority bytes changed")
    authority = read_json(authority_path)
    review = read_json(review_path)
    required_sha_fields = (
        "signed_pair_v2_manifest_sha256",
        "signed_pair_v2_independent_review_sha256",
        "three_bank_source_manifest_sha256",
        "three_bank_source_freeze_verification_sha256",
        "three_bank_source_independent_review_sha256",
        "generation_release_authority_sha256",
        "generation_runner_sha256",
        "literal_source_sha256",
        "capture_adapter_sha256",
        "generation_single_use_capability_sha256",
        "ordered_688_row_scope_sha256",
        "ordered_688_record_sha256",
    )
    if (
        authority.get("schema") != "public-q24-signed-pair-bank-source-authority-v1"
        or authority.get("status")
        != "FROZEN_REVIEWED_ALLOWLIST_BEFORE_THREE_BANK_GENERATION"
        or authority.get("frozen_before_bank_generation") is not True
        or authority.get("issued_by") != "root"
        or not isinstance(authority.get("issued_utc"), str)
        or not authority.get("issued_utc")
        or any(not is_sha(authority.get(key)) for key in required_sha_fields)
    ):
        raise AuthorityError("bank source allowlist schema or provenance invalid")
    for permission in (
        "target_access_allowed",
        "training_allowed",
        "deployment_allowed",
        "package_allowed",
        "submission_allowed",
    ):
        if authority.get(permission) is not False:
            raise AuthorityError("bank source allowlist is not target-free")
    if (
        authority["signed_pair_v2_manifest_sha256"]
        != PARENT_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"]
        or authority["signed_pair_v2_independent_review_sha256"]
        != PARENT_HASHES[ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"]
        or authority["ordered_688_record_sha256"]
        != bindings["ordered_688_record_sha256"]
    ):
        raise AuthorityError("bank source allowlist predecessor or row scope changed")
    if (
        review.get("schema")
        != "public-q24-signed-pair-bank-source-authority-independent-review-v1"
        or review.get("status") != "GO_THREE_BANK_GENERATION_AUTHORITY_ALLOWLIST"
        or review.get("authority_sha256") != sha256(authority_path)
        or review.get("frozen_before_bank_generation") is not True
    ):
        raise AuthorityError("bank source allowlist lacks independent pre-generation GO")
    receipt_bindings = receipt.get("release_bindings")
    if not isinstance(receipt_bindings, dict):
        raise AuthorityError("generation receipt release bindings absent")
    if receipt.get("release_authority_sha256") != authority["generation_release_authority_sha256"]:
        raise AuthorityError("generation receipt authority is not externally allowlisted")
    crosswalk = {
        "runner_sha256": "generation_runner_sha256",
        "literal_source_sha256": "literal_source_sha256",
        "capture_adapter_sha256": "capture_adapter_sha256",
        "single_use_capability_sha256": "generation_single_use_capability_sha256",
        "ordered_688_row_scope_sha256": "ordered_688_row_scope_sha256",
        "ordered_688_record_sha256": "ordered_688_record_sha256",
        "three_bank_authority_manifest_sha256": "three_bank_source_manifest_sha256",
        "three_bank_freeze_verification_sha256": "three_bank_source_freeze_verification_sha256",
        "three_bank_independent_review_sha256": "three_bank_source_independent_review_sha256",
    }
    for receipt_key, authority_key in crosswalk.items():
        if receipt_bindings.get(receipt_key) != authority[authority_key]:
            raise AuthorityError(f"generation receipt is outside allowlist: {receipt_key}")
    consumption = receipt.get("complete_banks", {}).get("capability_consumption")
    if (
        not isinstance(consumption, dict)
        or consumption.get("consumed_sha256")
        != authority["generation_single_use_capability_sha256"]
        or consumption.get("original_absent") is not True
        or consumption.get("one_use") is not True
        or consumption.get("no_retry") is not True
    ):
        raise AuthorityError("generation capability consumption is not allowlisted")
    return {
        "authority_path": str(authority_path),
        "authority_sha256": sha256(authority_path),
        "independent_review_path": str(review_path),
        "independent_review_sha256": sha256(review_path),
        "generation_release_authority_sha256": authority[
            "generation_release_authority_sha256"
        ],
        "three_bank_source_independent_review_sha256": authority[
            "three_bank_source_independent_review_sha256"
        ],
    }


def validate_bank_authority(release: dict[str, Any]) -> BankAuthority:
    bindings = release["bindings"]
    receipt_path = Path(release["_resolved"]["receipt_path"])
    bank_root = Path(release["_resolved"]["bank_root"])
    if receipt_path.is_symlink() or not receipt_path.is_file() or not bank_root.is_dir():
        raise AuthorityError("bank authority path missing or symlinked")
    raw = receipt_path.read_bytes()
    receipt_hash = hashlib.sha256(raw).hexdigest()
    if receipt_hash != bindings["bank_generation_receipt_sha256"]:
        raise AuthorityError("bank generation receipt hash mismatch")
    receipt = json.loads(raw.decode("ascii"))
    if raw != canonical_json_bytes(receipt):
        raise AuthorityError("bank generation receipt is not exact canonical JSON")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "public-q24-signed-pair-three-bank-generation-receipt-v1"
        or receipt.get("status") != "SEALED_TARGET_BLIND_THREE_BANKS"
        or receipt.get("ordered_688_record_sha256") != bindings["ordered_688_record_sha256"]
        or receipt.get("ordered_ranks") != list(range(ROWS))
        or receipt.get("target_accessed") is not False
        or receipt.get("training_run") is not False
        or receipt.get("estimator_deployed") is not False
        or receipt.get("package_built") is not False
        or receipt.get("submission_made") is not False
    ):
        raise AuthorityError("bank generation receipt authority mismatch")
    external_provenance = validate_external_bank_source_authority(release, receipt)
    parent = receipt.get("parent_bindings", {})
    if (
        parent.get("signed_pair_v2_manifest_sha256")
        != PARENT_HASHES[ACTION_V2 / "PACKAGE_MANIFEST.json"]
        or parent.get("signed_pair_v2_freeze_verification_sha256")
        != PARENT_HASHES[ACTION_V2 / "FREEZE_VERIFICATION.json"]
        or parent.get("signed_pair_v2_independent_review_sha256")
        != PARENT_HASHES[ACTION_V2 / "INDEPENDENT_STATIC_REVIEW_V2.json"]
    ):
        raise AuthorityError("bank receipt predecessor binding changed")
    rows = receipt.get("row_receipts")
    if not isinstance(rows, list) or len(rows) != ROWS:
        raise AuthorityError("bank row receipt count mismatch")
    ids: list[int] = []
    for expected_rank, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AuthorityError("bank row receipt object required")
        rank = strict_int(row.get("rank"), minimum=0, maximum=ROWS - 1)
        mlp_id = strict_int(row.get("mlp_id"), minimum=0)
        if rank != expected_rank:
            raise AuthorityError("bank row rank order mismatch")
        ids.append(mlp_id)
        banks = row.get("banks")
        if not isinstance(banks, dict) or set(banks) != {"16", "64", "256"}:
            raise AuthorityError("row H bank set mismatch")
        for h in HS:
            item = banks[str(h)]
            if (
                not isinstance(item, dict)
                or item.get("shape") != [13, WIDTH, h]
                or item.get("dtype") != "float32"
                or item.get("finite") is not True
                or not is_sha(item.get("action_bytes_sha256"))
                or not is_sha(item.get("A15_zero_bytes_sha256"))
            ):
                raise AuthorityError("row bank schema mismatch")
            layer_hashes = item.get("layer_action_bytes_sha256")
            if (
                not isinstance(layer_hashes, list)
                or len(layer_hashes) != 13
                or any(not is_sha(value) for value in layer_hashes)
            ):
                raise AuthorityError("row layer action digest mismatch")
    if len(set(ids)) != ROWS:
        raise AuthorityError("bank mlp_id roster is not unique")
    final = receipt.get("complete_banks")
    if not isinstance(final, dict) or final.get("cross_rung_H16_prefix_byte_identical") is not True:
        raise AuthorityError("complete-bank prefix parity absent")
    complete = final.get("banks")
    if not isinstance(complete, dict) or set(complete) != {"16", "64", "256"}:
        raise AuthorityError("complete bank map mismatch")
    bank_paths: dict[int, Path] = {}
    zero_paths: dict[int, Path] = {}
    exact_bank_bytes = {16: 586_153_984, 64: 2_344_615_936, 256: 9_378_463_744}
    exact_zero_bytes = {16: 65_536, 64: 262_144, 256: 1_048_576}
    for h in HS:
        item = complete[str(h)]
        if (
            not isinstance(item, dict)
            or item.get("shape") != [ROWS, 13, WIDTH, h]
            or item.get("dtype") != "float32"
            or item.get("finite") is not True
            or item.get("sha256") != bindings["complete_bank_sha256"][str(h)]
            or item.get("A15_zero_sha256") != bindings["A15_zero_sha256"][str(h)]
        ):
            raise AuthorityError("complete bank receipt binding mismatch")
        bank_path = contained_regular_file(bank_root, item.get("path"), exact_bank_bytes[h])
        zero_path = contained_regular_file(bank_root, item.get("A15_zero_path"), exact_zero_bytes[h])
        if sha256(bank_path) != item["sha256"] or sha256(zero_path) != item["A15_zero_sha256"]:
            raise AuthorityError("complete bank artifact digest mismatch")
        with zero_path.open("rb", buffering=0) as handle:
            if any(byte != 0 for block in iter(lambda: handle.read(1 << 20), b"") for byte in block):
                raise AuthorityError("A15 buffer is not exact +0.0 bytes")
        bank_paths[h] = bank_path
        zero_paths[h] = zero_path
    return BankAuthority(
        receipt_path=receipt_path,
        receipt_sha256=receipt_hash,
        ordered_record_sha256=bindings["ordered_688_record_sha256"],
        mlp_ids=np.asarray(ids, dtype=np.int64),
        bank_paths=bank_paths,
        zero_paths=zero_paths,
        resource_observations=copy.deepcopy(final.get("exact_meters")),
        novelty_diagnostics={
            "bank_source_external_provenance": external_provenance,
            "novelty_reference_diagnostics": load_optional_novelty_diagnostics(release),
        },
    )


def load_optional_novelty_diagnostics(release: dict[str, Any]) -> dict[str, Any]:
    """Best-effort observations only; this function never raises or changes H."""
    records = release.get("novelty_diagnostic_receipts", [])
    output: dict[str, Any] = {"policy": "DIAGNOSTIC_ONLY_NO_VETO", "records": []}
    if not isinstance(records, list):
        output["unavailable"] = "release field is not a list"
        return output
    if not records:
        output["external_allowlist"] = "NOT_REQUIRED_NO_DIAGNOSTIC_RECEIPTS"
        return output
    bindings = release.get("bindings", {})
    try:
        allowlist_hash = bindings.get("novelty_reference_allowlist_sha256")
        review_hash = bindings.get("novelty_reference_allowlist_review_sha256")
        if not is_sha(allowlist_hash) or not is_sha(review_hash):
            raise ValueError("external novelty allowlist hashes missing")
        release_parent = Path(release["_resolved"]["release_parent"])
        allowlist_path = resolve_input(
            release.get("novelty_reference_allowlist_path"), release_parent
        )
        review_path = resolve_input(
            release.get("novelty_reference_allowlist_review_path"), release_parent
        )
        if sha256(allowlist_path) != allowlist_hash or sha256(review_path) != review_hash:
            raise ValueError("external novelty allowlist bytes changed")
        allowlist = read_json(allowlist_path)
        review = read_json(review_path)
        allowed = allowlist.get("allowed_receipt_sha256")
        if (
            allowlist.get("schema")
            != "public-q24-signed-pair-novelty-reference-allowlist-v1"
            or allowlist.get("status")
            != "FROZEN_REVIEWED_REFERENCE_RECEIPT_ALLOWLIST"
            or not isinstance(allowed, list)
            or any(not is_sha(value) for value in allowed)
            or review.get("status") != "GO_NOVELTY_REFERENCE_RECEIPT_ALLOWLIST"
            or review.get("allowlist_sha256") != allowlist_hash
        ):
            raise ValueError("external novelty allowlist schema/review invalid")
        allowed_hashes = set(allowed)
        output["external_allowlist"] = {
            "path": str(allowlist_path),
            "sha256": allowlist_hash,
            "review_path": str(review_path),
            "review_sha256": review_hash,
        }
    except Exception as error:
        output["external_allowlist"] = {
            "status": "UNAVAILABLE_DIAGNOSTIC_ONLY",
            "reason": f"{type(error).__name__}:{error}",
        }
        output["records"] = [
            {"status": "UNAVAILABLE_DIAGNOSTIC_ONLY__NO_EXTERNAL_ALLOWLIST"}
            for _ in records
        ]
        return output
    for record in records:
        try:
            if not isinstance(record, dict) or not is_sha(record.get("sha256")):
                raise ValueError("invalid diagnostic binding")
            if record["sha256"] not in allowed_hashes:
                raise ValueError("diagnostic receipt is not externally allowlisted")
            path = resolve_input(record.get("path"), release_parent)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("diagnostic hash mismatch")
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant: {value}")
                ),
            )
            output["records"].append({"path": str(path), "sha256": record["sha256"], "payload": payload})
        except Exception as error:
            output["records"].append({"status": "UNAVAILABLE_DIAGNOSTIC_ONLY", "reason": f"{type(error).__name__}:{error}"})
    return output


def open_bank_memmaps(authority: BankAuthority) -> dict[int, np.memmap]:
    return {
        h: np.memmap(
            authority.bank_paths[h],
            mode="r",
            dtype="<f4",
            shape=(ROWS, 13, WIDTH, h),
            order="C",
        )
        for h in HS
    }


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def int32_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values.astype("<i4", copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def tensor_state_hash(state: dict[str, torch.Tensor], order: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    names = tuple(state) if order is None else order
    for name in names:
        value = state[name].detach().cpu().numpy()
        if value.dtype != np.float32:
            raise AssertionError(f"state tensor is not float32: {name}")
        digest.update(np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def inner_folds(outer_fold: int, outer_fit: np.ndarray, mlp_ids: np.ndarray) -> np.ndarray:
    if outer_fit.dtype.kind not in "iu" or len(outer_fit) != OUTER_FIT:
        raise AssertionError("outer-fit schema changed")
    ranks = [strict_int(int(value), minimum=0, maximum=ROWS - 1) for value in outer_fit]
    if ranks != sorted(ranks) or len(set(ranks)) != OUTER_FIT:
        raise AssertionError("outer-fit ranks must be unique increasing")
    order = sorted(
        ranks,
        key=lambda rank: (
            hashlib.blake2s(
                f"signed-pair-capacity-v2:{outer_fold}:{int(mlp_ids[rank])}".encode("ascii"),
                digest_size=16,
            ).digest(),
            int(mlp_ids[rank]),
            rank,
        ),
    )
    assignment = np.empty(ROWS, dtype=np.int8)
    assignment.fill(-1)
    for position, rank in enumerate(order):
        assignment[rank] = position % INNER_FOLDS
    for inner in range(INNER_FOLDS):
        if int(np.sum(assignment[outer_fit] == inner)) != INNER_HELD:
            raise AssertionError("inner fold is not balanced")
    return assignment


def target_subset(qref: Any, records: list[dict[str, Any]], ranks: np.ndarray) -> dict[int, np.ndarray]:
    """Decode exactly one declared fold-role subset from the sealed Parquet bundle."""
    ordered = [strict_int(int(value), minimum=0, maximum=ROWS - 1) for value in ranks]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise AssertionError("target role ranks must be unique increasing")
    requested_ids = [strict_int(int(records[rank]["mlp_id"]), minimum=0) for rank in ordered]
    table = qref.pq.read_table(
        TARGETS,
        columns=["mlp_id", "all_layer_means"],
        filters=[("mlp_id", "in", requested_ids)],
    )
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not qref.pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(len(ids), DEPTH, WIDTH)
    if ids.dtype != np.int32 or targets.dtype != np.float32 or len(ids) != len(ordered):
        raise AssertionError("target subset schema or cardinality changed")
    by_id: dict[int, np.ndarray] = {}
    for index, mlp_id_value in enumerate(ids):
        mlp_id = int(mlp_id_value)
        if mlp_id in by_id:
            raise AssertionError("duplicate target mlp_id")
        by_id[mlp_id] = np.ascontiguousarray(targets[index], dtype=np.float32)
    if set(by_id) != set(requested_ids):
        raise AssertionError("target subset identity mismatch")
    return {rank: by_id[int(records[rank]["mlp_id"])] for rank in ordered}


def sparse_error(
    targets_by_rank: dict[int, np.ndarray], hosts: np.ndarray
) -> np.ndarray:
    error = np.zeros((ROWS, DEPTH, WIDTH), dtype=np.float32)
    for rank in sorted(targets_by_rank):
        error[rank] = np.asarray(targets_by_rank[rank] - hosts[rank], dtype=np.float32)
    return error


def normalizer_hash(centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (centers, scales, layer_rms):
        if value.dtype != np.float32:
            raise AssertionError("J normalizer is not float32")
        digest.update(np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def fit_j(
    j: Any,
    dataset: Any,
    error: np.ndarray,
    fit: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    layer_rms: np.ndarray,
    *,
    initialization_seed: int,
    shuffle_seed: int,
    stream_label: str,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(initialization_seed)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    initial_hash = tensor_state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=2.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )
    steps_per_epoch = math.ceil(len(fit) / 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=J_EPOCHS * steps_per_epoch, eta_min=1.0e-4
    )
    layer_weights = torch.linspace(0.5, 1.5, DEPTH - 2, dtype=torch.float32)
    rng = np.random.Generator(np.random.PCG64(shuffle_seed & 0xFFFF_FFFF))
    order_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch, exposure in enumerate((0.0, 0.25, 0.5, 0.75, 1.0, 1.0), start=1):
        order = rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), 3):
            ranks = order[offset : offset + 3]
            raw, sigma, weights, truth = j.load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states, _ = model(
                raw,
                sigma,
                weights,
                truth=truth,
                exposure=exposure,
                return_states=True,
            )
            normalized = (
                prediction[:, 2:] - truth[:, 2:]
            ) / model.layer_rms[2:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            rollout = torch.mean(layer_weights * layer_mse)
            final_focus = layer_mse[-1]
            auxiliary = states[:, 2:, :, 1:]
            aux_normalized = auxiliary / model.layer_rms[2:][None, :, None, None]
            aux_penalty = 2.0e-5 * aux_normalized.square().mean()
            identity_penalty = 5.0e-4 * (
                model.transport_delta.square().mean() + model.mix_delta.square().mean()
            )
            loss = rollout + final_focus + aux_penalty + identity_penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite J loss: {stream_label}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        trace.append(
            {
                "epoch": epoch,
                "exposure": exposure,
                "objective": objective_sum / examples,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **observation(f"{stream_label}:J:epoch{epoch}", started),
            }
        )
    model.eval()
    final_state = clone_state(model)
    return model, {
        "fit_rank_int32_sha256": int32_hash(fit),
        "normalizer_sha256": normalizer_hash(centers, scales, layer_rms),
        "initial_state_sha256": initial_hash,
        "batch_order_int32_sha256": order_digest.hexdigest(),
        "payload_sha256": tensor_state_hash(final_state),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": J_EPOCHS * steps_per_epoch,
        "trace": trace,
    }


def replay_j_row(
    j: Any,
    model: nn.Module,
    dataset: Any,
    rank: int,
    dummy_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, sigma, weights, _ = j.load_batch(
        dataset, np.asarray([rank], dtype=np.int64), dummy_error
    )
    with torch.no_grad():
        direct = model(raw, sigma, weights, exposure=1.0)
        if not isinstance(direct, torch.Tensor):
            raise TypeError("bound J returned a non-tensor")
        states = torch.zeros((1, WIDTH, 4), dtype=torch.float32)
        energies = torch.zeros_like(states)
        identity = torch.eye(4, dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        features_out: list[torch.Tensor] = []
        for layer in range(DEPTH):
            if layer < 2:
                states = torch.zeros_like(states)
                energies = torch.zeros_like(states)
                outputs.append(states[..., 0])
                continue
            if layer == 2:
                messages = torch.zeros_like(states)
                energies = torch.zeros_like(states)
            else:
                layer_weights = weights[:, layer]
                messages = torch.bmm(states.transpose(1, 2), layer_weights).transpose(1, 2)
                energies = torch.bmm(
                    states.square().transpose(1, 2), layer_weights.square()
                ).transpose(1, 2)
                phi = raw[:, layer, :, 5, None]
                messages = phi * messages
                energies = phi.square() * energies
            tail = torch.clamp(
                (raw[:, layer, :, 1:] - model.centers[layer]) / model.scales[layer],
                -8.0,
                8.0,
            )
            rms = model.layer_rms[layer]
            rms2 = rms.square()
            message_rms = torch.clamp(messages / rms, -8.0, 8.0)
            message_sigma = torch.clamp(
                messages
                / torch.clamp(sigma[:, layer, :, None] * rms, min=2.0**-30),
                -8.0,
                8.0,
            )
            log_energy = torch.clamp(
                torch.log1p(torch.clamp(energies / rms2, min=0.0)), 0.0, 8.0
            )
            coherence = torch.clamp(
                messages / torch.sqrt(energies + 1.0e-4 * rms2), -8.0, 8.0
            )
            features = torch.cat(
                (raw[:, layer, :, :1], tail, message_rms, message_sigma, log_energy, coherence),
                dim=-1,
            )
            if features.shape != (1, WIDTH, 36):
                raise AssertionError("live J feature width changed")
            hidden = model.trunk(features)
            source_standard = torch.einsum(
                "bwh,hc->bwc", hidden, model.source_weight[layer]
            ) + model.source_bias[layer]
            physical_scale = sigma[:, layer] * model.layer_rms[layer]
            source = physical_scale[..., None] * source_standard
            gate_standard = torch.einsum(
                "bwh,hc->bwc", hidden, model.gate_weight[layer]
            ) + model.gate_bias[layer]
            dynamic_gain = 1.0 + 0.25 * torch.tanh(
                gate_standard + model.transport_delta[layer]
            )
            mixing = identity + 0.12 * torch.tanh(model.mix_delta[layer])
            transported = torch.einsum(
                "bwc,cd->bwd", dynamic_gain * messages, mixing
            )
            states = transported + source
            outputs.append(states[..., 0])
            features_out.append(features)
        replay = torch.stack(outputs, dim=1)
        if not torch.equal(replay, direct):
            raise AssertionError("literal x36 replay changed bound J output bytes")
        x36 = torch.stack(features_out, dim=1)[0].cpu().numpy()
        output = replay[0].cpu().numpy()
        sigma_np = sigma[0, 2:].cpu().numpy()
    return (
        np.ascontiguousarray(x36, dtype=np.float32),
        np.ascontiguousarray(output, dtype=np.float32),
        np.ascontiguousarray(sigma_np, dtype=np.float32),
    )


@dataclass
class PairPrepared:
    ranks: np.ndarray
    rank_to_slot: dict[int, int]
    x36: np.ndarray
    sigma: np.ndarray
    base: np.ndarray
    residual: np.ndarray
    truth_final: np.ndarray
    weights_paths: tuple[Path, ...]
    replay_sha256: str


def prepare_pair(
    j: Any,
    model: nn.Module,
    dataset: Any,
    ranks: np.ndarray,
    targets_by_rank: dict[int, np.ndarray],
    hosts: np.ndarray,
    stream_label: str,
) -> PairPrepared:
    ordered = np.asarray(sorted(int(value) for value in ranks), dtype=np.int32)
    if set(int(value) for value in ordered) != set(targets_by_rank):
        raise AssertionError("prepared target role mismatch")
    count = len(ordered)
    x36 = np.empty((count, 14, WIDTH, 36), dtype=np.float32)
    sigma = np.empty((count, 14, WIDTH), dtype=np.float32)
    base = np.empty((count, 14, WIDTH), dtype=np.float32)
    residual = np.empty_like(base)
    truth_final = np.empty((count, WIDTH), dtype=np.float32)
    weights_paths: list[Path] = []
    replay_digest = hashlib.sha256()
    dummy_error = np.zeros((ROWS, DEPTH, WIDTH), dtype=np.float32)
    for slot, rank_value in enumerate(ordered):
        rank = int(rank_value)
        x_row, j_output, sigma_row = replay_j_row(
            j, model, dataset, rank, dummy_error
        )
        target = targets_by_rank[rank]
        base_row = np.asarray(hosts[rank, 2:] + j_output[2:], dtype=np.float32)
        x36[slot] = x_row
        sigma[slot] = sigma_row
        base[slot] = base_row
        residual[slot] = np.asarray(target[2:] - base_row, dtype=np.float32)
        truth_final[slot] = target[-1]
        weights_paths.append(dataset.load(rank).weights_path)
        replay_digest.update(np.ascontiguousarray(x_row.astype("<f4", copy=False)).tobytes())
        replay_digest.update(
            np.ascontiguousarray(j_output.astype("<f4", copy=False)).tobytes()
        )
        if (slot + 1) % 32 == 0:
            print(json.dumps({"stage": "prepare_pair", "stream": stream_label, "rows": slot + 1}), flush=True)
    return PairPrepared(
        ranks=ordered,
        rank_to_slot={int(rank): index for index, rank in enumerate(ordered)},
        x36=x36,
        sigma=sigma,
        base=base,
        residual=residual,
        truth_final=truth_final,
        weights_paths=tuple(weights_paths),
        replay_sha256=replay_digest.hexdigest(),
    )


def r_l_for(prepared: PairPrepared, fit: np.ndarray) -> np.ndarray:
    sums = np.zeros(14, dtype=np.float64)
    count = 0
    for rank_value in fit:
        slot = prepared.rank_to_slot[int(rank_value)]
        row = prepared.residual[slot].astype(np.float64)
        for layer_slot in range(14):
            sums[layer_slot] += np.add.reduce(
                np.square(row[layer_slot], dtype=np.float64), dtype=np.float64
            )
        count += WIDTH
    result = np.maximum(np.sqrt(sums / float(count)), 2.0**-20).astype(np.float32)
    if not np.all(np.isfinite(result)) or result.shape != (14,):
        raise FloatingPointError("nonfinite pair R_l")
    return result


def float32_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        if array.dtype != np.float32:
            raise AssertionError("float32 hash input changed")
        digest.update(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


class PairState(nn.Module):
    def __init__(self, initial: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.from_numpy(initial["U"].copy()))
        self.b = nn.Parameter(torch.from_numpy(initial["b"].copy()))
        self.V = nn.Parameter(torch.from_numpy(initial["V"].copy()))
        self.c = nn.Parameter(torch.from_numpy(initial["c"].copy()))
        self.d = nn.Parameter(torch.from_numpy(initial["d"].copy()))

    def deltas(
        self,
        x36: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        actions: torch.Tensor,
        r_l: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.zeros((WIDTH, 16), dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        for slot, layer in enumerate(PAIR_LAYERS):
            if layer == 2:
                message = torch.zeros((WIDTH, 16), dtype=torch.float32)
            else:
                physical = sigma[slot - 1, :, None] * r_l[slot - 1] * h
                transported = (physical.transpose(0, 1) @ weights[layer]).transpose(0, 1)
                denominator = torch.clamp(
                    sigma[slot, :, None] * r_l[slot], min=2.0**-30
                )
                raw_message = transported / denominator
                message = raw_message / (1.0 + torch.abs(raw_message))
            action = (
                actions[slot]
                if layer <= 14
                else torch.zeros((WIDTH, actions.shape[-1]), dtype=torch.float32)
            )
            z = torch.cat((x36[slot], message, action), dim=-1)
            hidden = torch.tanh(z @ self.U + self.b)
            h = torch.tanh(hidden @ self.V[slot] + self.c[slot])
            readout = 0.0625 * torch.tanh(self.d[slot])
            outputs.append(sigma[slot] * r_l[slot] * (h @ readout))
        return torch.stack(outputs, dim=0)


def initial_pair_state(h: int, seed: int) -> tuple[dict[str, np.ndarray], str]:
    if h not in HS:
        raise AssertionError("unfrozen H")
    rng = np.random.Generator(np.random.PCG64(seed & 0xFFFF_FFFF))
    U = np.ascontiguousarray(rng.standard_normal((52 + h, 64), dtype=np.float32))
    U *= np.float32(0.02)
    V = np.ascontiguousarray(rng.standard_normal((14, 64, 16), dtype=np.float32))
    V *= np.float32(0.02)
    state = {
        "U": U,
        "b": np.zeros(64, dtype=np.float32),
        "V": V,
        "c": np.zeros((14, 16), dtype=np.float32),
        "d": np.zeros((14, 16), dtype=np.float32),
    }
    expected_parameters = {16: 19_200, 64: 22_272, 256: 34_560}[h]
    if sum(value.size for value in state.values()) != expected_parameters:
        raise AssertionError("pair parameter count changed")
    return state, float32_hash(*(state[name] for name in ("U", "b", "V", "c", "d")))


def pair_loss_for_rank(
    model: PairState,
    prepared: PairPrepared,
    rank: int,
    actions: np.ndarray,
    r_l: np.ndarray,
) -> torch.Tensor:
    slot = prepared.rank_to_slot[rank]
    weights = np.load(prepared.weights_paths[slot], mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.float32 or weights.shape != (DEPTH, WIDTH, WIDTH):
        raise AssertionError("weight schema changed")
    delta = model.deltas(
        torch.from_numpy(np.ascontiguousarray(prepared.x36[slot])),
        torch.from_numpy(np.ascontiguousarray(prepared.sigma[slot])),
        torch.from_numpy(np.array(weights, dtype=np.float32, order="C", copy=True)),
        torch.from_numpy(np.array(actions, dtype=np.float32, order="C", copy=True)),
        torch.from_numpy(np.ascontiguousarray(r_l)),
    )
    target = torch.from_numpy(np.ascontiguousarray(prepared.residual[slot]))
    normalized = (target - delta) / torch.from_numpy(r_l)[:, None]
    layer_mse = normalized.square().mean(dim=1)
    layer_weights = torch.tensor(
        [np.float32(0.5 + slot_value / 13.0) for slot_value in range(14)],
        dtype=torch.float32,
    )
    return torch.sum(layer_weights * layer_mse) / torch.sum(layer_weights) + layer_mse[-1]


def train_pair_arm(
    prepared: PairPrepared,
    fit: np.ndarray,
    bank: np.memmap,
    h: int,
    r_l: np.ndarray,
    initial: dict[str, np.ndarray],
    initialization_sha256: str,
    shuffle_seed: int,
    *,
    pair_on: bool,
    stream_label: str,
) -> tuple[PairState, dict[str, Any]]:
    model = PairState(copy.deepcopy(initial))
    if tensor_state_hash(model.state_dict(), ("U", "b", "V", "c", "d")) != initialization_sha256:
        raise AssertionError("pair arm initialization bytes differ")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        weight_decay=2.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PAIR_EPOCHS * len(fit), eta_min=6.0e-5
    )
    rng = np.random.Generator(np.random.PCG64(shuffle_seed & 0xFFFF_FFFF))
    order_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    zero_actions = np.zeros((13, WIDTH, h), dtype=np.float32)
    started = time.perf_counter()
    for epoch in range(1, PAIR_EPOCHS + 1):
        order = rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        objective = 0.0
        model.train()
        for rank_value in order:
            rank = int(rank_value)
            actions = np.asarray(bank[rank], dtype=np.float32) if pair_on else zero_actions
            if actions.shape != (13, WIDTH, h) or not np.all(np.isfinite(actions)):
                raise FloatingPointError("nonfinite or malformed pair action")
            loss = pair_loss_for_rank(model, prepared, rank, actions, r_l)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite pair loss: {stream_label}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for parameter in model.parameters():
                    if not bool(torch.all(torch.isfinite(parameter))):
                        raise FloatingPointError("nonfinite pair parameter")
                    parameter.clamp_(-4.0, 4.0)
            scheduler.step()
            objective += float(loss.detach())
        trace.append(
            {
                "epoch": epoch,
                "mean_network_objective": objective / len(fit),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **observation(f"{stream_label}:pair:{'on' if pair_on else 'zero'}:epoch{epoch}", started),
            }
        )
    model.eval()
    return model, {
        "initialization_sha256": initialization_sha256,
        "batch_order_int32_sha256": order_digest.hexdigest(),
        "payload_sha256": tensor_state_hash(
            model.state_dict(), ("U", "b", "V", "c", "d")
        ),
        "trace": trace,
    }


def network_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.ascontiguousarray(prediction, dtype=np.float64).reshape(-1, order="C")
    target64 = np.ascontiguousarray(target, dtype=np.float64).reshape(-1, order="C")
    value = float(np.mean(np.square(prediction64 - target64), dtype=np.float64))
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError("nonfinite network MSE")
    return value


def score_pair_arms(
    on_model: PairState,
    zero_model: PairState,
    prepared: PairPrepared,
    held: np.ndarray,
    bank: np.memmap,
    h: int,
    r_l: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    zero_actions = np.zeros((13, WIDTH, h), dtype=np.float32)
    on_model.eval()
    zero_model.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            slot = prepared.rank_to_slot[rank]
            weights_np = np.load(
                prepared.weights_paths[slot], mmap_mode="r", allow_pickle=False
            )
            weights = torch.from_numpy(
                np.array(weights_np, dtype=np.float32, order="C", copy=True)
            )
            x36 = torch.from_numpy(np.ascontiguousarray(prepared.x36[slot]))
            sigma = torch.from_numpy(np.ascontiguousarray(prepared.sigma[slot]))
            scale = torch.from_numpy(np.ascontiguousarray(r_l))
            action_np = np.asarray(bank[rank], dtype=np.float32)
            on_delta = on_model.deltas(
                x36,
                sigma,
                weights,
                torch.from_numpy(
                    np.array(action_np, dtype=np.float32, order="C", copy=True)
                ),
                scale,
            ).cpu().numpy()
            zero_delta = zero_model.deltas(
                x36,
                sigma,
                weights,
                torch.from_numpy(zero_actions),
                scale,
            ).cpu().numpy()
            base_final = prepared.base[slot, -1].astype(np.float64)
            target_final = prepared.truth_final[slot].astype(np.float64)
            on_prediction = np.maximum(base_final + on_delta[-1].astype(np.float64), 0.0)
            zero_prediction = np.maximum(
                base_final + zero_delta[-1].astype(np.float64), 0.0
            )
            j_prediction = np.maximum(base_final, 0.0)
            rows.append(
                {
                    "role_rank": rank,
                    "pair_on_mse": network_mse(on_prediction, target_final),
                    "pair_zero_mse": network_mse(zero_prediction, target_final),
                    "fold_J_mse": network_mse(j_prediction, target_final),
                }
            )
    return rows


def aggregate_mse(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError("invalid aggregate MSE vector")
    return float(np.mean(values, dtype=np.float64))


def paired_stat(rows: list[dict[str, Any]], comparator_key: str) -> dict[str, Any]:
    gain = np.asarray(
        [float(row[comparator_key]) - float(row["pair_on_mse"]) for row in rows],
        dtype=np.float64,
    )
    if len(gain) < 2 or not np.all(np.isfinite(gain)):
        raise FloatingPointError("invalid paired gain vector")
    mean = float(np.mean(gain, dtype=np.float64))
    standard_deviation = float(np.std(gain, dtype=np.float64, ddof=1))
    standard_error = standard_deviation / math.sqrt(float(len(gain)))
    if standard_deviation == 0.0:
        sentinel = "POSITIVE_INFINITY" if mean > 0.0 else "ZERO" if mean == 0.0 else "NEGATIVE_INFINITY"
        z: float | str = sentinel
    else:
        z_value = mean / standard_error
        if not math.isfinite(z_value):
            raise FloatingPointError("nonfinite divided paired z")
        z = z_value
    return {
        "n": len(gain),
        "mean_gain": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "z": z,
    }


def z_at_least(statistic: dict[str, Any], threshold: float) -> bool:
    z = statistic["z"]
    if z == "POSITIVE_INFINITY":
        return True
    if isinstance(z, str):
        return False
    return float(z) >= threshold


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: strict_int(row["role_rank"], minimum=0, maximum=ROWS - 1))
    return {
        "networks": len(ordered),
        "pair_on_raw_mse": aggregate_mse(ordered, "pair_on_mse"),
        "pair_zero_raw_mse": aggregate_mse(ordered, "pair_zero_mse"),
        "fold_J_raw_mse": aggregate_mse(ordered, "fold_J_mse"),
        "strict_win_fraction_vs_pair_zero": float(
            np.mean(
                np.asarray(
                    [row["pair_on_mse"] < row["pair_zero_mse"] for row in ordered],
                    dtype=np.float64,
                )
            )
        ),
        "strict_win_fraction_vs_fold_J": float(
            np.mean(
                np.asarray(
                    [row["pair_on_mse"] < row["fold_J_mse"] for row in ordered],
                    dtype=np.float64,
                )
            )
        ),
        "paired_vs_pair_zero": paired_stat(ordered, "pair_zero_mse"),
        "paired_vs_fold_J": paired_stat(ordered, "fold_J_mse"),
    }


def select_h(pair_on_raw_by_h: dict[int, float]) -> int:
    """The sole H-selection law: no diagnostic or comparator input exists."""
    if set(pair_on_raw_by_h) != set(HS):
        raise AssertionError("all three H values must enter selection")
    for h in HS:
        value = pair_on_raw_by_h[h]
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise FloatingPointError("H selection requires finite nonnegative raw MSE")
    minimum = min(pair_on_raw_by_h.values())
    if minimum == 0.0:
        candidates = [h for h in HS if pair_on_raw_by_h[h] == 0.0]
    else:
        candidates = [
            h
            for h in HS
            if pair_on_raw_by_h[h] <= minimum * (1.0 + TIE_TOLERANCE)
        ]
    if not candidates:
        raise AssertionError("tie law produced no H")
    return min(candidates)


def pair_seeds(h: int, outer_fold: int, inner_fold: int | None) -> tuple[int, int]:
    tag = H_TAG[h]
    if inner_fold is None:
        initialization = PAIR_SEED + outer_fold + tag
        shuffle = (
            PAIR_SEED ^ ((outer_fold + 1) * 0x9E3779B1) ^ tag
        ) & 0xFFFF_FFFF
    else:
        initialization = (
            PAIR_SEED
            + 1_000_000
            + 100_000 * outer_fold
            + 10_000 * inner_fold
            + tag
        )
        shuffle = (
            PAIR_SEED
            ^ 0x51ED270B
            ^ ((outer_fold + 1) * 0x9E3779B1)
            ^ ((inner_fold + 1) * 0x85EBCA6B)
            ^ tag
        ) & 0xFFFF_FFFF
    return initialization, shuffle


def run_inner_selection(
    j: Any,
    dataset: Any,
    outer_fold: int,
    outer_fit: np.ndarray,
    targets_by_rank: dict[int, np.ndarray],
    hosts: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    mlp_ids: np.ndarray,
    banks: dict[int, np.memmap],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment = inner_folds(outer_fold, outer_fit, mlp_ids)
    rows_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in HS}
    inner_receipts: list[dict[str, Any]] = []
    saved: dict[str, Any] = {}
    for inner in range(INNER_FOLDS):
        inner_fit = np.asarray(
            [rank for rank in outer_fit if assignment[int(rank)] != inner],
            dtype=np.int32,
        )
        inner_held = np.asarray(
            [rank for rank in outer_fit if assignment[int(rank)] == inner],
            dtype=np.int32,
        )
        if len(inner_fit) != INNER_FIT or len(inner_held) != INNER_HELD:
            raise AssertionError("inner fold size changed")
        error = sparse_error(targets_by_rank, hosts)
        centers, scales, layer_rms = j.moments_for(
            inner_fit, error, sums, sums2
        )
        init_seed = J_SEED + 100_000 + 1_000 * outer_fold + inner
        shuffle_seed = (
            J_SEED
            ^ 0x51ED270B
            ^ ((outer_fold + 1) * 0x9E3779B1)
            ^ ((inner + 1) * 0x85EBCA6B)
        ) & 0xFFFF_FFFF
        j_model, j_receipt = fit_j(
            j,
            dataset,
            error,
            inner_fit,
            centers,
            scales,
            layer_rms,
            initialization_seed=init_seed,
            shuffle_seed=shuffle_seed,
            stream_label=f"outer{outer_fold}:inner{inner}",
        )
        prepared = prepare_pair(
            j,
            j_model,
            dataset,
            outer_fit,
            targets_by_rank,
            hosts,
            f"outer{outer_fold}:inner{inner}",
        )
        r_l = r_l_for(prepared, inner_fit)
        j_receipt["held_rank_int32_sha256"] = int32_hash(inner_held)
        j_receipt["x36_and_output_replay_sha256"] = prepared.replay_sha256
        h_receipts: dict[str, Any] = {}
        h_saved: dict[str, Any] = {}
        shared_j_mse: list[float] | None = None
        for h in HS:
            initialization_seed, pair_shuffle_seed = pair_seeds(h, outer_fold, inner)
            initial, initial_hash = initial_pair_state(h, initialization_seed)
            on_model, on_training = train_pair_arm(
                prepared,
                inner_fit,
                banks[h],
                h,
                r_l,
                initial,
                initial_hash,
                pair_shuffle_seed,
                pair_on=True,
                stream_label=f"outer{outer_fold}:inner{inner}:H{h}",
            )
            zero_model, zero_training = train_pair_arm(
                prepared,
                inner_fit,
                banks[h],
                h,
                r_l,
                initial,
                initial_hash,
                pair_shuffle_seed,
                pair_on=False,
                stream_label=f"outer{outer_fold}:inner{inner}:H{h}",
            )
            if on_training["batch_order_int32_sha256"] != zero_training["batch_order_int32_sha256"]:
                raise AssertionError("paired arms saw different rank streams")
            scored = score_pair_arms(
                on_model,
                zero_model,
                prepared,
                inner_held,
                banks[h],
                h,
                r_l,
            )
            if shared_j_mse is None:
                shared_j_mse = [float(row["fold_J_mse"]) for row in scored]
            elif shared_j_mse != [float(row["fold_J_mse"]) for row in scored]:
                raise AssertionError("inner J comparator differs across H")
            rows_by_h[h].extend(scored)
            h_receipts[str(h)] = {
                "r_l_sha256": float32_hash(r_l),
                "pair_on_training": on_training,
                "pair_zero_training": zero_training,
                "held_rows": scored,
            }
            h_saved[str(h)] = {
                "pair_on_state_dict": clone_state(on_model),
                "pair_zero_state_dict": clone_state(zero_model),
            }
            del on_model, zero_model
        inner_receipts.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner,
                "fit_rows": len(inner_fit),
                "held_rows": len(inner_held),
                "j": j_receipt,
                "H": h_receipts,
            }
        )
        saved[str(inner)] = {
            "j_state_dict": clone_state(j_model),
            "j_centers": centers,
            "j_scales": scales,
            "j_layer_rms": layer_rms,
            "H": h_saved,
        }
        del prepared, j_model, error
    metrics: dict[str, Any] = {}
    raw_by_h: dict[int, float] = {}
    for h in HS:
        rows_by_h[h].sort(key=lambda row: int(row["role_rank"]))
        expected = [int(value) for value in outer_fit]
        if [int(row["role_rank"]) for row in rows_by_h[h]] != expected:
            raise AssertionError("inner OOF does not cover outer fit exactly once")
        summary = summarize_rows(rows_by_h[h])
        raw = float(summary["pair_on_raw_mse"])
        raw_by_h[h] = raw
        metrics[str(h)] = {
            **summary,
            "diagnostic_alignment_only": {
                "pair_on_below_pair_zero": raw < float(summary["pair_zero_raw_mse"]),
                "pair_on_below_fold_J": raw < float(summary["fold_J_raw_mse"]),
                "paired_z_at_least_2_vs_pair_zero": z_at_least(
                    summary["paired_vs_pair_zero"], 2.0
                ),
                "paired_z_at_least_2_vs_fold_J": z_at_least(
                    summary["paired_vs_fold_J"], 2.0
                ),
                "strict_win_fraction_vs_fold_J_at_least_0_52": float(
                    summary["strict_win_fraction_vs_fold_J"]
                )
                >= 0.52,
                "can_veto_H_selection": False,
            },
            "per_network": rows_by_h[h],
        }
    selected = select_h(raw_by_h)
    selection = {
        "schema": "public-q24-signed-pair-inner-H-selection-v1",
        "status": "H_SELECTED_FROM_OUTER_FIT_ONLY__OUTER_HELD_ROLE_NOT_SCORED",
        "outer_fold": outer_fold,
        "candidate_H": list(HS),
        "all_H_eligible_regardless_of_novelty_rank_energy_or_resources": True,
        "outer_fit_rank_int32_sha256": int32_hash(outer_fit),
        "inner_assignment_int8_sha256": hashlib.sha256(
            np.ascontiguousarray(assignment[outer_fit].astype("|i1", copy=False)).tobytes()
        ).hexdigest(),
        "metrics": metrics,
        "pair_on_raw_mse_selection_input": {str(h): raw_by_h[h] for h in HS},
        "tie_tolerance": TIE_TOLERANCE,
        "selected_H": selected,
        "novelty_rank_energy_used_for_selection": False,
        "resource_or_residual_used_for_selection": False,
        "outer_held_target_values_passed_to_selection": 0,
        "inner_receipts": inner_receipts,
    }
    return selection, saved


def train_and_score_outer(
    j: Any,
    qref: Any,
    records: list[dict[str, Any]],
    dataset: Any,
    outer_fold: int,
    outer_fit: np.ndarray,
    outer_held: np.ndarray,
    fit_targets: dict[int, np.ndarray],
    hosts: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    selected_h: int,
    bank: np.memmap,
) -> tuple[dict[str, Any], dict[str, Any]]:
    error = sparse_error(fit_targets, hosts)
    centers, scales, layer_rms = j.moments_for(outer_fit, error, sums, sums2)
    j_model, j_receipt = fit_j(
        j,
        dataset,
        error,
        outer_fit,
        centers,
        scales,
        layer_rms,
        initialization_seed=J_SEED + outer_fold,
        shuffle_seed=(J_SEED ^ (outer_fold * 0x9E3779B1)) & 0xFFFF_FFFF,
        stream_label=f"outer{outer_fold}",
    )
    prepared_fit = prepare_pair(
        j,
        j_model,
        dataset,
        outer_fit,
        fit_targets,
        hosts,
        f"outer{outer_fold}:fit",
    )
    r_l = r_l_for(prepared_fit, outer_fit)
    initialization_seed, shuffle_seed = pair_seeds(selected_h, outer_fold, None)
    initial, initial_hash = initial_pair_state(selected_h, initialization_seed)
    on_model, on_training = train_pair_arm(
        prepared_fit,
        outer_fit,
        bank,
        selected_h,
        r_l,
        initial,
        initial_hash,
        shuffle_seed,
        pair_on=True,
        stream_label=f"outer{outer_fold}:selectedH{selected_h}",
    )
    zero_model, zero_training = train_pair_arm(
        prepared_fit,
        outer_fit,
        bank,
        selected_h,
        r_l,
        initial,
        initial_hash,
        shuffle_seed,
        pair_on=False,
        stream_label=f"outer{outer_fold}:selectedH{selected_h}",
    )
    if on_training["batch_order_int32_sha256"] != zero_training["batch_order_int32_sha256"]:
        raise AssertionError("outer paired rank streams differ")

    # This is deliberately the first fold-k outer-held target-role request.
    held_targets = target_subset(qref, records, outer_held)
    prepared_held = prepare_pair(
        j,
        j_model,
        dataset,
        outer_held,
        held_targets,
        hosts,
        f"outer{outer_fold}:held-after-selection",
    )
    rows = score_pair_arms(
        on_model,
        zero_model,
        prepared_held,
        outer_held,
        bank,
        selected_h,
        r_l,
    )
    summary = summarize_rows(rows)
    receipt = {
        "outer_fold": outer_fold,
        "selected_H": selected_h,
        "fit_rows": len(outer_fit),
        "held_rows": len(outer_held),
        "fit_rank_int32_sha256": int32_hash(outer_fit),
        "held_rank_int32_sha256": int32_hash(outer_held),
        "J": {
            **j_receipt,
            "fit_x36_and_output_replay_sha256": prepared_fit.replay_sha256,
            "held_x36_and_output_replay_sha256": prepared_held.replay_sha256,
        },
        "r_l": r_l.tolist(),
        "r_l_sha256": float32_hash(r_l),
        "pair_on_training": on_training,
        "pair_zero_training": zero_training,
        "summary": summary,
        "per_network": rows,
    }
    saved = {
        "j_state_dict": clone_state(j_model),
        "j_centers": centers,
        "j_scales": scales,
        "j_layer_rms": layer_rms,
        "r_l": r_l,
        "selected_H": selected_h,
        "pair_on_state_dict": clone_state(on_model),
        "pair_zero_state_dict": clone_state(zero_model),
    }
    return receipt, saved


def claim_owned() -> bool:
    if ACTIVE_CLAIM_NONCE is None or not CLAIM.is_file():
        return False
    try:
        value = read_json(CLAIM)
    except Exception:
        return False
    return value.get("owner_pid") == os.getpid() and value.get("owner_nonce") == ACTIVE_CLAIM_NONCE


def record_terminal_failure(error: BaseException) -> bool:
    if not claim_owned() or FAILURE.exists() or BUNDLE.exists():
        return False
    value = {
        "schema": "public-q24-signed-pair-nested-accuracy-terminal-failure-v1",
        "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
        "claim_sha256": sha256(CLAIM),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "private_partial_bundle": str(ACTIVE_TEMP_BUNDLE) if ACTIVE_TEMP_BUNDLE else None,
        "public_bundle_present": BUNDLE.exists(),
        "decision": "NO_RETRY__ATTRIBUTE_FAILURE_BEFORE_SUCCESSOR",
    }
    write_exclusive_durable(FAILURE, pretty_json_bytes(value))
    return True


def execute(release_path: Path, capability_path: Path) -> None:
    global ACTIVE_CLAIM_NONCE, ACTIVE_TEMP_BUNDLE
    if CLAIM.exists() or FAILURE.exists() or BUNDLE.exists():
        raise FileExistsError("nested accuracy transaction already claimed or completed")
    verify_static_parents(include_target_bundle=True)
    release = validate_release(release_path)
    expected_capability = Path(release["_resolved"]["capability_path"])
    if capability_path.resolve() != expected_capability:
        raise AuthorityError("CLI capability path differs from exact release")
    capability_raw, capability = validate_capability(
        capability_path, release_path, release
    )
    bank_authority = validate_bank_authority(release)

    owner_nonce = hashlib.sha256(
        os.getpid().to_bytes(8, "big", signed=False) + os.urandom(32)
    ).hexdigest()
    claim = {
        "schema": "public-q24-signed-pair-nested-accuracy-run-claim-v1",
        "status": "CLAIMED_AFTER_THREE_BANK_VERIFICATION_BEFORE_CAPABILITY_CONSUMPTION_OR_TARGET_DECODE__NO_RETRY",
        "bindings": {
            "release_sha256": sha256(release_path),
            "capability_sha256": hashlib.sha256(capability_raw).hexdigest(),
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "static_verification_sha256": sha256(STATIC),
            "bank_generation_receipt_sha256": bank_authority.receipt_sha256,
            "ordered_688_record_sha256": bank_authority.ordered_record_sha256,
            "target_bundle_sha256": TARGETS_SHA256,
            "outer_fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
        },
        "target_values_decoded_before_claim": 0,
        "row_or_weight_values_opened_before_claim": 0,
        "bank_bytes_verified_before_claim": True,
        "owner_pid": os.getpid(),
        "owner_nonce": owner_nonce,
    }
    write_exclusive_durable(CLAIM, pretty_json_bytes(claim))
    ACTIVE_CLAIM_NONCE = owner_nonce
    consumed_capability = consume_capability(capability_path, capability_raw)

    j = load_module("signed_pair_accuracy_bound_j", J_SOURCE)
    qref = load_module("signed_pair_accuracy_bound_qref", QREF_SOURCE)
    records, native_storage = qref.verify_native_rows()
    record_ids = np.asarray([int(record["mlp_id"]) for record in records], dtype=np.int64)
    if not np.array_equal(record_ids, bank_authority.mlp_ids):
        raise AuthorityError("three-bank and native observer-fit identities differ")
    folds = qref.balanced_folds(records)
    if hashlib.sha256(np.ascontiguousarray(folds.astype("|u1", copy=False)).tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AuthorityError("outer fold bytes changed")
    dataset = qref.All688Dataset(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False)
                for rank in range(ROWS)
            ]
        ),
        dtype=np.float32,
    )
    if hosts.shape != (ROWS, DEPTH, WIDTH):
        raise AssertionError("host array schema changed")
    stats = np.load(RAW_STATS, allow_pickle=False)
    sums = stats["sum"]
    sums2 = stats["sum2"]
    if sums.dtype != np.float64 or sums.shape != (ROWS, DEPTH, 19) or sums2.shape != sums.shape:
        raise AssertionError("raw sufficient-stat schema changed")
    banks = open_bank_memmaps(bank_authority)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    temporary = HERE / f".{BUNDLE.name}.{os.getpid()}.{owner_nonce[:16]}.tmp"
    temporary.mkdir(mode=0o700)
    ACTIVE_TEMP_BUNDLE = temporary
    saved: dict[str, Any] = {
        "schema": "public-q24-signed-pair-nested-accuracy-models-v1",
        "inner": {},
        "outer": {},
    }
    outer_receipts: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selection_hashes: dict[str, str] = {}
    for outer_fold in range(OUTER_FOLDS):
        outer_fit = np.flatnonzero(folds != outer_fold).astype(np.int32)
        outer_held = np.flatnonzero(folds == outer_fold).astype(np.int32)
        if len(outer_fit) != OUTER_FIT or len(outer_held) != OUTER_HELD:
            raise AssertionError("outer fold size changed")
        fit_targets = target_subset(qref, records, outer_fit)
        selection, inner_saved = run_inner_selection(
            j,
            dataset,
            outer_fold,
            outer_fit,
            fit_targets,
            hosts,
            sums,
            sums2,
            record_ids,
            banks,
        )
        selection_path = temporary / f"SELECTION_OUTER_FOLD_{outer_fold}.json"
        write_exclusive_durable(selection_path, pretty_json_bytes(selection))
        selection_hash = sha256(selection_path)
        selection_hashes[str(outer_fold)] = selection_hash
        frozen_selection = read_json(selection_path)
        selected_h = strict_int(frozen_selection.get("selected_H"))
        if selected_h not in HS or sha256(selection_path) != selection_hash:
            raise AuthorityError("durable H selection changed before held scoring")
        outer_receipt, outer_saved = train_and_score_outer(
            j,
            qref,
            records,
            dataset,
            outer_fold,
            outer_fit,
            outer_held,
            fit_targets,
            hosts,
            sums,
            sums2,
            selected_h,
            banks[selected_h],
        )
        outer_receipt["selection_receipt_sha256"] = selection_hash
        outer_receipts.append(outer_receipt)
        selected_rows.extend(outer_receipt["per_network"])
        saved["inner"][str(outer_fold)] = inner_saved
        saved["outer"][str(outer_fold)] = outer_saved

    selected_rows.sort(key=lambda row: int(row["role_rank"]))
    if [int(row["role_rank"]) for row in selected_rows] != list(range(ROWS)):
        raise AssertionError("final selected-H OOF coverage changed")
    final_summary = summarize_rows(selected_rows)
    fold_gates = []
    for receipt in outer_receipts:
        summary = receipt["summary"]
        fold_gates.append(
            {
                "outer_fold": receipt["outer_fold"],
                "selected_H": receipt["selected_H"],
                "pair_on_below_pair_zero": summary["pair_on_raw_mse"]
                < summary["pair_zero_raw_mse"],
                "pair_on_below_fold_J": summary["pair_on_raw_mse"]
                < summary["fold_J_raw_mse"],
            }
        )
    gates = {
        "finite_complete_688": True,
        "selected_pair_on_raw_strictly_below_1_2425668176231516e_8": float(
            final_summary["pair_on_raw_mse"]
        )
        < RAW_HURDLE,
        "pair_on_below_pair_zero_all_four_folds": all(
            row["pair_on_below_pair_zero"] for row in fold_gates
        ),
        "pair_on_below_fold_J_all_four_folds": all(
            row["pair_on_below_fold_J"] for row in fold_gates
        ),
        "strict_network_win_fraction_vs_fold_J_at_least_0_55": float(
            final_summary["strict_win_fraction_vs_fold_J"]
        )
        >= 0.55,
        "paired_z_vs_pair_zero_at_least_3": z_at_least(
            final_summary["paired_vs_pair_zero"], 3.0
        ),
        "aggregate_pair_on_below_fold_J": float(final_summary["pair_on_raw_mse"])
        < float(final_summary["fold_J_raw_mse"]),
    }
    passed = all(gates.values())

    models_path = temporary / "MODELS.pt"
    torch.save(saved, models_path)
    with models_path.open("rb") as handle:
        os.fsync(handle.fileno())
    result = {
        "schema": "public-q24-signed-pair-nested-accuracy-oof-result-v1",
        "status": "PASS_SIGNED_PAIR_RAW_HURDLE__NO_SUBMISSION_AUTHORITY"
        if passed
        else "FAIL_SIGNED_PAIR_FROZEN_NESTED_OOF__NO_RETRY",
        "access_declarations": {
            "role": "observer_fit",
            "target_columns_opened": ["mlp_id", "all_layer_means"],
            "observer_selection_targets_opened": [],
            "names_opened": [],
            "scores_opened": [],
            "board_opened": False,
            "network": False,
            "package": False,
            "submission": False,
        },
        "bindings": {
            "claim_sha256": sha256(CLAIM),
            "release_sha256": sha256(release_path),
            "consumed_capability_sha256": sha256(consumed_capability),
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "static_verification_sha256": sha256(STATIC),
            "bank_generation_receipt_sha256": bank_authority.receipt_sha256,
            "ordered_688_record_sha256": bank_authority.ordered_record_sha256,
            "target_bundle_sha256": TARGETS_SHA256,
            "outer_fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
            "selection_receipt_sha256": selection_hashes,
            "models_sha256": sha256(models_path),
        },
        "accuracy_first_amendment": {
            "all_H_entered_each_inner_selection": list(HS),
            "novelty_rank_energy_used_for_selection": False,
            "resource_residual_used_for_selection_or_accuracy_verdict": False,
        },
        "aggregate": final_summary,
        "fold_results": outer_receipts,
        "fold_gates": fold_gates,
        "gates": gates,
        "resource_observations_deployment_only": bank_authority.resource_observations,
        "novelty_observations_diagnostic_only": bank_authority.novelty_diagnostics,
        "native_storage_bytes_verified_after_claim": native_storage,
        "per_network_selected_H": selected_rows,
        "decision": (
            "GO_SIGNED_PAIR_ACCURACY_INTEGRATION_AND_RESOURCE_OPTIMIZATION_CHILD__NO_SUBMISSION"
            if passed
            else "CLOSE_THIS_SIGNED_PAIR_THREE_BANK_SPELLING__OPEN_SUPPORT_GE3_K6_CHILD__NO_RETRY"
        ),
    }
    result_path = temporary / "RESULT.json"
    write_exclusive_durable(result_path, pretty_json_bytes(result))
    directory = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    temporary.replace(BUNDLE)
    parent = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    ACTIVE_TEMP_BUNDLE = None
    print(
        json.dumps(
            {
                "status": result["status"],
                "pair_on_raw_mse": final_summary["pair_on_raw_mse"],
                "pair_zero_raw_mse": final_summary["pair_zero_raw_mse"],
                "fold_J_raw_mse": final_summary["fold_J_raw_mse"],
                "selected_H_by_fold": [row["selected_H"] for row in outer_receipts],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def preflight(release_path: Path | None) -> None:
    verify_static_parents(include_target_bundle=False)
    checks: dict[str, Any] = {
        "runtime_python": ".".join(map(str, sys.version_info[:3])),
        "runtime_numpy": np.__version__,
        "runtime_torch": torch.__version__,
        "runtime_exact": (
            sys.version_info[:3] == (3, 11, 5)
            and np.__version__ == "1.26.4"
            and torch.__version__ == "2.7.0"
        ),
        "preregistration_present": PREREG.is_file(),
        "static_verification_present": STATIC.is_file(),
        "claim_absent": not CLAIM.exists(),
        "terminal_failure_absent": not FAILURE.exists(),
        "output_bundle_absent": not BUNDLE.exists(),
        "release_supplied": release_path is not None,
        "pyarrow_imported": "pyarrow" in sys.modules,
        "bank_bytes_opened": 0,
        "row_or_weight_values_opened": 0,
        "target_values_decoded": 0,
    }
    if release_path is not None:
        try:
            validate_release(release_path.resolve())
            checks["release_validation"] = "PASS_METADATA_ONLY_NO_BANK_OPEN"
        except Exception as error:
            checks["release_validation"] = f"HOLD:{type(error).__name__}:{error}"
    ready = (
        checks["runtime_exact"]
        and checks["preregistration_present"]
        and checks["static_verification_present"]
        and checks["claim_absent"]
        and checks["terminal_failure_absent"]
        and checks["output_bundle_absent"]
        and checks["release_supplied"]
        and checks.get("release_validation") == "PASS_METADATA_ONLY_NO_BANK_OPEN"
        and checks["pyarrow_imported"] is False
    )
    print(
        json.dumps(
            {
                "schema": "public-q24-signed-pair-nested-accuracy-preflight-v1",
                "status": "READY_FOR_BANK_VERIFICATION_AND_ONE_SHOT_CLAIM"
                if ready
                else "HOLD_NO_EXACT_RELEASE",
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--capability", type=Path)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.release)
        return
    if args.release is None or args.capability is None:
        raise SystemExit("--execute requires --release and --capability")
    try:
        execute(args.release.resolve(), args.capability.resolve())
    except BaseException as error:
        record_terminal_failure(error)
        raise


if __name__ == "__main__":
    main()
