#!/usr/bin/env python3
"""Stdlib-only lean-runtime-V5 seal and independent-review issuer skeleton.

This program is inert unless an explicit issue mode, exact confirmation phrase,
and absent output path are supplied.  It never imports numerical libraries and
never opens target, score, name, board, training, or estimator artifacts.  Seal
issuance authenticates the exact lean runtime-V5 receipt/control roots, hashes all
six published bank files, checks every A15 byte, and replays the eight-sector
byte-prefix law.  Review issuance repeats those byte checks and emits a distinct
review schema.

The evidence is deliberately narrow: frozen runtime control flow plus the
authenticated runtime receipt declare that generation was target-free.  This
utility observes only its own zero target access; it does not claim external
process tracing or hostile-interpreter isolation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


HERE = Path(__file__).absolute().parent
RUNNER = HERE / "nested_accuracy_runner.py"
RUNTIME_V5_PROFILE_STATUS = "EXACT_LEAN_RUNTIME_V5"
SEAL_CONFIRMATION = "ISSUE_EXACT_LEAN_RUNTIME_V5_POST_GENERATION_SEAL"
REVIEW_CONFIRMATION = "ISSUE_EXACT_INDEPENDENT_POST_GENERATION_REVIEW_V5"
EXPECTED_H = (16, 64, 256)
EXPECTED_FILES = {
    16: ("actions_H16.f32le", 586_153_984, "A15_zero_H16.f32le", 65_536),
    64: ("actions_H64.f32le", 2_344_615_936, "A15_zero_H64.f32le", 262_144),
    256: ("actions_H256.f32le", 9_378_463_744, "A15_zero_H256.f32le", 1_048_576),
}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
RUNTIME_ANCHOR_KEYS = {
    "schema",
    "status",
    "issued_by",
    "issued_utc",
    "bindings",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
}
RUNTIME_CAPABILITY_KEYS = {
    "schema",
    "status",
    "capability_id",
    "issued_by",
    "issued_utc",
    "authority_anchor_sha256",
    "runner_sha256",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "one_use",
    "no_retry",
    "source_import_allowed",
    "row_access_allowed",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
}
RUNTIME_RELEASE_KEYS = {
    "schema",
    "status",
    "issued_by",
    "issued_utc",
    "authority_anchor_sha256",
    "single_use_capability_sha256",
    "bindings",
    "one_use",
    "no_retry",
    "source_import_allowed",
    "row_access_allowed",
    "target_access_allowed",
    "training_allowed",
    "estimator_deployment_allowed",
    "package_allowed",
    "submission_allowed",
}
RUNTIME_CLAIM_KEYS = {
    "schema",
    "status",
    "callable_bundle_manifest_sha256",
    "execution_stack_manifest_sha256",
    "independent_review_sha256",
    "authority_anchor_sha256",
    "single_use_capability_sha256",
    "final_release_sha256",
    "runner_sha256",
    "callable_component_sha256",
    "ordered_688_row_scope_sha256",
    "ordered_688_record_sha256",
    "capability_id",
    "target_access_allowed",
    "training_allowed",
    "package_allowed",
    "submission_allowed",
}


class SealError(RuntimeError):
    """Fail-closed post-generation sealing/review rejection."""


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


def _strict_json(raw: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise SealError(f"{label} contains nonfinite JSON constant {value}")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SealError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"{label} is not canonical ASCII JSON") from error
    if raw != canonical_json_bytes(value):
        raise SealError(f"{label} bytes are not canonical JSON")
    return value


def _canonical_id(value: Any, label: str, *, exact: str | None = None) -> str:
    if type(value) is not str or ID_RE.fullmatch(value) is None:
        raise SealError(f"{label} is not a canonical identifier")
    if exact is not None and value != exact:
        raise SealError(f"{label} issuer mismatch")
    return value


def _canonical_utc(value: Any, label: str) -> str:
    if type(value) is not str:
        raise SealError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise SealError(f"{label} is not canonical UTC") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SealError(f"{label} is not canonical UTC")
    return value


def _lstat_regular_one_link(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    metadata = None
    for part in absolute.parts[1:]:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SealError(f"{label} contains a symlink component")
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise SealError(f"{label} is not a regular file")
    if metadata.st_nlink != 1:
        raise SealError(f"{label} must have exactly one link")
    return absolute


def _lstat_nonsymlink_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    metadata = None
    for part in absolute.parts[1:]:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise SealError(f"{label} contains a symlink component")
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise SealError(f"{label} is not a directory")
    return absolute


def _open_checked(
    path: Path, label: str, expected_size: int | None = None
) -> tuple[int, int]:
    checked = _lstat_regular_one_link(path, label)
    before = os.lstat(checked)
    if expected_size is not None and before.st_size != expected_size:
        raise SealError(f"{label} byte count changed")
    descriptor = os.open(checked, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_dev != before.st_dev
        or observed.st_ino != before.st_ino
        or observed.st_size != before.st_size
    ):
        os.close(descriptor)
        raise SealError(f"{label} identity changed while opening")
    return descriptor, int(observed.st_size)


def _read_exact_file(path: Path, label: str) -> bytes:
    descriptor, size = _open_checked(path, label)
    try:
        parts: list[bytes] = []
        remaining = size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                raise SealError(f"{label} ended before its authenticated size")
            parts.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise SealError(f"{label} grew while reading")
        return b"".join(parts)
    finally:
        os.close(descriptor)


def _read_canonical(path: Path, label: str) -> tuple[bytes, Any]:
    raw = _read_exact_file(path, label)
    return raw, _strict_json(raw, label)


def _sha256(
    path: Path,
    label: str,
    expected_size: int | None = None,
    *,
    require_all_zero: bool = False,
) -> str:
    descriptor, size = _open_checked(path, label, expected_size)
    digest = hashlib.sha256()
    try:
        remaining = size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                raise SealError(f"{label} ended before its authenticated size")
            if require_all_zero and any(block):
                raise SealError(f"{label} contains a nonzero byte")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise SealError(f"{label} grew while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_exact_descriptor(descriptor: int, count: int, label: str) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        block = os.read(descriptor, remaining)
        if not block:
            raise SealError(f"{label} ended before its authenticated size")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "signed_pair_nested_accuracy_oof_v4_v5_sealer_contract", RUNNER
    )
    if specification is None or specification.loader is None:
        raise SealError("cannot load frozen V5 receipt contract")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _validate_runtime_control_documents(
    runner: Any,
    anchor_path: Path,
    release_path: Path,
    consumed_capability_path: Path,
) -> dict[str, Any]:
    anchor_raw, anchor = _read_canonical(anchor_path, "generation runtime anchor")
    release_raw, release = _read_canonical(release_path, "generation final release")
    capability_raw, capability = _read_canonical(
        consumed_capability_path, "consumed generation capability"
    )
    consumed_path = Path(os.path.abspath(os.fspath(consumed_capability_path)))
    if not consumed_path.name.endswith(".consumed"):
        raise SealError("consumed generation capability name lacks .consumed suffix")
    original_path = consumed_path.with_name(consumed_path.name.removesuffix(".consumed"))
    if original_path.exists() or original_path.is_symlink():
        raise SealError("original generation capability path still exists")
    consumed_stat = os.lstat(consumed_path)
    anchor_sha = hashlib.sha256(anchor_raw).hexdigest()
    release_sha = hashlib.sha256(release_raw).hexdigest()
    capability_sha = hashlib.sha256(capability_raw).hexdigest()
    expected_bindings = {
        "runner_sha256": runner.RUNTIME_V5_HASHES["runner_sha256"],
        "callable_bundle_manifest_sha256": runner.RUNTIME_V5_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": runner.RUNTIME_V5_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": runner.RUNTIME_V5_HASHES[
            "independent_review_sha256"
        ],
        "ordered_688_row_scope_sha256": runner.ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": runner.ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": runner.RUNTIME_V5_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ],
    }
    if (
        type(anchor) is not dict
        or set(anchor) != RUNTIME_ANCHOR_KEYS
        or anchor["schema"] != "public-q24-three-bank-runtime-authority-anchor-v5"
        or anchor["status"] != "FROZEN_RUNTIME_AND_ORDERED_SCOPE_ALLOWLIST"
        or anchor["bindings"] != expected_bindings
    ):
        raise SealError("generation runtime anchor changed")
    if (
        type(capability) is not dict
        or set(capability) != RUNTIME_CAPABILITY_KEYS
        or capability["schema"] != "public-q24-signed-pair-bank-capability-v5"
        or capability["status"] != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS"
        or capability["authority_anchor_sha256"] != anchor_sha
        or capability["runner_sha256"] != expected_bindings["runner_sha256"]
        or capability["callable_bundle_manifest_sha256"]
        != expected_bindings["callable_bundle_manifest_sha256"]
        or capability["execution_stack_manifest_sha256"]
        != expected_bindings["execution_stack_manifest_sha256"]
        or capability["ordered_688_row_scope_sha256"]
        != expected_bindings["ordered_688_row_scope_sha256"]
        or capability["ordered_688_record_sha256"]
        != expected_bindings["ordered_688_record_sha256"]
        or capability["manifest_compatibility_scan_receipt_sha256"]
        != expected_bindings["manifest_compatibility_scan_receipt_sha256"]
    ):
        raise SealError("consumed generation capability changed")
    if (
        type(release) is not dict
        or set(release) != RUNTIME_RELEASE_KEYS
        or release["schema"] != "public-q24-three-bank-runtime-final-release-v5"
        or release["status"] != "AUTHORIZE_ONE_TARGET_BLIND_688_BANK_PASS"
        or release["authority_anchor_sha256"] != anchor_sha
        or release["single_use_capability_sha256"] != capability_sha
        or release["bindings"] != expected_bindings
    ):
        raise SealError("generation final release changed")
    anchor_utc = _canonical_utc(anchor["issued_utc"], "generation anchor issued_utc")
    capability_utc = _canonical_utc(
        capability["issued_utc"], "generation capability issued_utc"
    )
    release_utc = _canonical_utc(
        release["issued_utc"], "generation release issued_utc"
    )
    if not anchor_utc <= capability_utc <= release_utc:
        raise SealError("generation runtime control chronology changed")
    for document, label in (
        (anchor, "generation anchor"),
        (capability, "generation capability"),
        (release, "generation release"),
    ):
        _canonical_id(document["issued_by"], f"{label} issued_by")
    capability_id = _canonical_id(
        capability["capability_id"], "generation capability_id"
    )
    for document in (anchor, capability, release):
        for key in (
            "target_access_allowed",
            "training_allowed",
            "estimator_deployment_allowed",
            "package_allowed",
            "submission_allowed",
        ):
            if document.get(key) is not False:
                raise SealError(f"generation authority firewall changed: {key}")
    if (
        capability["one_use"] is not True
        or capability["no_retry"] is not True
        or capability["source_import_allowed"] is not True
        or capability["row_access_allowed"] is not True
        or release["one_use"] is not True
        or release["no_retry"] is not True
        or release["source_import_allowed"] is not True
        or release["row_access_allowed"] is not True
    ):
        raise SealError("generation one-use or source/row permission law changed")
    return {
        "anchor_sha256": anchor_sha,
        "release_sha256": release_sha,
        "capability_sha256": capability_sha,
        "capability_id": capability_id,
        "anchor_issued_utc": anchor_utc,
        "capability_issued_utc": capability_utc,
        "release_issued_utc": release_utc,
        "capability_inode": {
            "device": int(consumed_stat.st_dev),
            "inode": int(consumed_stat.st_ino),
            "size": int(consumed_stat.st_size),
            "initial_link_count": 1,
            "consumed_link_count": int(consumed_stat.st_nlink),
        },
    }


def _validate_runtime_claim(
    runner: Any,
    claim: Any,
    *,
    controls: dict[str, Any],
) -> dict[str, Any]:
    if type(claim) is not dict or set(claim) != RUNTIME_CLAIM_KEYS:
        raise SealError("generation claim fields are not exact lean runtime V5")
    expected = {
        "callable_bundle_manifest_sha256": runner.RUNTIME_V5_HASHES[
            "callable_bundle_manifest_sha256"
        ],
        "execution_stack_manifest_sha256": runner.RUNTIME_V5_HASHES[
            "execution_stack_manifest_sha256"
        ],
        "independent_review_sha256": runner.RUNTIME_V5_HASHES[
            "independent_review_sha256"
        ],
        "authority_anchor_sha256": controls["anchor_sha256"],
        "single_use_capability_sha256": controls["capability_sha256"],
        "final_release_sha256": controls["release_sha256"],
        "runner_sha256": runner.RUNTIME_V5_HASHES["runner_sha256"],
        "callable_component_sha256": runner.RUNTIME_V5_COMPONENT_SHA256,
        "ordered_688_row_scope_sha256": runner.ORDERED_SCOPE_BINDINGS[
            "ordered_688_row_scope_sha256"
        ],
        "ordered_688_record_sha256": runner.ORDERED_SCOPE_BINDINGS[
            "ordered_688_record_sha256"
        ],
        "capability_id": controls["capability_id"],
    }
    if (
        claim["schema"] != "public-q24-three-bank-runtime-exclusive-claim-v5"
        or claim["status"] != "CLAIMED_ONE_TARGET_BLIND_688_BANK_PASS"
        or any(claim[key] != value for key, value in expected.items())
    ):
        raise SealError("generation claim differs from exact lean runtime-V5 transaction")
    for key in (
        "target_access_allowed",
        "training_allowed",
        "package_allowed",
        "submission_allowed",
    ):
        if claim[key] is not False:
            raise SealError(f"generation claim firewall changed: {key}")
    return claim


def _hash_banks_and_replay_sector_law(
    runner: Any, bundle: Path, receipt: dict[str, Any]
) -> dict[str, Any]:
    root = _lstat_nonsymlink_directory(bundle, "generation bundle")
    banks = receipt["complete_banks"]["banks"]
    paths: dict[int, Path] = {}
    zeros: dict[int, Path] = {}
    complete_sha: dict[str, str] = {}
    zero_sha: dict[str, str] = {}
    expected_inventory = {"THREE_BANK_GENERATION_RECEIPT.json"}
    for h in EXPECTED_H:
        tape_name, tape_bytes, zero_name, zero_bytes = EXPECTED_FILES[h]
        item = banks[str(h)]
        if item["path"] != tape_name or item["A15_zero_path"] != zero_name:
            raise SealError(f"H{h} canonical artifact names changed")
        tape = root / tape_name
        zero = root / zero_name
        paths[h] = tape
        zeros[h] = zero
        complete_sha[str(h)] = _sha256(tape, f"H{h} bank", tape_bytes)
        zero_sha[str(h)] = _sha256(
            zero, f"H{h} A15", zero_bytes, require_all_zero=True
        )
        if complete_sha[str(h)] != item["sha256"]:
            raise SealError(f"H{h} complete bank hash changed")
        if zero_sha[str(h)] != item["A15_zero_sha256"]:
            raise SealError(f"H{h} A15 hash changed")
        expected_inventory.update((tape_name, zero_name))
    if {item.name for item in root.iterdir()} != expected_inventory:
        raise SealError("generation bundle inventory is not exactly seven files")

    layer_bytes = {h: 1024 * h * 4 for h in EXPECTED_H}
    comparisons = 0
    descriptors: dict[int, int] = {}
    try:
        for h in EXPECTED_H:
            descriptor, _ = _open_checked(
                paths[h], f"H{h} bank sector replay", EXPECTED_FILES[h][1]
            )
            descriptors[h] = descriptor
        for rank in range(688):
            for layer_slot in range(13):
                payload = {
                    h: _read_exact_descriptor(
                        descriptors[h],
                        layer_bytes[h],
                        f"H{h} sector replay rank={rank} layer={layer_slot}",
                    )
                    for h in EXPECTED_H
                }
                try:
                    comparisons += validate_sector_layer_payloads(payload, width=1024)
                except SealError as error:
                    raise SealError(
                        "eight-sector prefix mismatch at "
                        f"rank={rank},layer_slot={layer_slot}: {error}"
                    ) from error
        for h in EXPECTED_H:
            if os.read(descriptors[h], 1):
                raise SealError(f"H{h} bank grew during sector-prefix replay")
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    return {
        "complete_bank_sha256": complete_sha,
        "A15_zero_sha256": zero_sha,
        "sector_prefix_comparisons": comparisons,
    }


def validate_sector_layer_payloads(payload: dict[int, bytes], *, width: int) -> int:
    """Pure-byte replay of one layer's exact 8-sector 2/8/32 prefix law."""
    if type(payload) is not dict or set(payload) != {16, 64, 256}:
        raise SealError("sector payload H inventory changed")
    if type(width) is not int or isinstance(width, bool) or width <= 0:
        raise SealError("sector payload width is invalid")
    if any(len(payload[h]) != width * h * 4 for h in EXPECTED_H):
        raise SealError("sector payload byte count changed")
    comparisons = 0
    for unit in range(width):
        for sector in range(8):
            p16 = (unit * 16 + sector * 2) * 4
            p64 = (unit * 64 + sector * 8) * 4
            p256 = (unit * 256 + sector * 32) * 4
            if (
                payload[16][p16 : p16 + 8]
                != payload[64][p64 : p64 + 8]
                or payload[16][p16 : p16 + 8]
                != payload[256][p256 : p256 + 8]
                or payload[64][p64 : p64 + 32]
                != payload[256][p256 : p256 + 32]
            ):
                raise SealError(f"unit={unit},sector={sector}")
            comparisons += 3
    return comparisons


def _validated_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    runner = _load_runner()
    source_raw, source = _read_canonical(
        args.source_authority, "pre-generation source authority"
    )
    review_raw, source_review = _read_canonical(
        args.source_authority_review, "pre-generation source-authority review"
    )
    source_sha = hashlib.sha256(source_raw).hexdigest()
    source_review_sha = hashlib.sha256(review_raw).hexdigest()
    checked_source = runner.validate_lean_v5_source_authority_object(source)
    checked_source_review = runner.validate_lean_v5_source_authority_review_object(
        source_review, authority_sha256=source_sha, authority=checked_source
    )
    if (
        source_sha != runner.V5_SOURCE_AUTHORITY_SHA256
        or source_review_sha != runner.V5_SOURCE_AUTHORITY_REVIEW_SHA256
        or checked_source["source_bindings"][
            "v4_terminal_successor_record_sha256"
        ]
        != runner.V5_SUCCESSOR_RECORD_SHA256
        or checked_source["source_bindings"][
            "manifest_compatibility_scan_receipt_sha256"
        ]
        != runner.RUNTIME_V5_HASHES[
            "manifest_compatibility_scan_receipt_sha256"
        ]
        or checked_source["source_bindings"]["v4_exclusive_claim_sha256"]
        != runner.V4_TERMINAL_FAILURE_BINDINGS["exclusive_claim_sha256"]
        or checked_source["source_bindings"]["v4_terminal_failure_sha256"]
        != runner.V4_TERMINAL_FAILURE_BINDINGS["terminal_failure_sha256"]
        or checked_source["source_bindings"]["v4_consumed_capability_sha256"]
        != runner.V4_TERMINAL_FAILURE_BINDINGS["consumed_capability_sha256"]
        or checked_source["source_bindings"][
            "independent_incident_root_cause_sha256"
        ]
        != runner.RUNTIME_V5_HASHES["independent_incident_root_cause_sha256"]
        or checked_source["source_bindings"][
            "flopscope_reset_verification_sha256"
        ]
        != runner.RUNTIME_V5_HASHES["flopscope_reset_verification_sha256"]
    ):
        raise SealError("source authority/review is not the frozen lean V5 chain")
    controls = _validate_runtime_control_documents(
        runner,
        args.generation_anchor,
        args.generation_release,
        args.consumed_generation_capability,
    )
    receipt_path = args.generation_bundle / "THREE_BANK_GENERATION_RECEIPT.json"
    receipt_raw, receipt = _read_canonical(
        receipt_path, "lean runtime-V5 generation receipt"
    )
    checked_receipt, _ = runner.validate_generation_receipt_object(receipt)
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    claim_raw, claim = _read_canonical(args.generation_claim, "generation claim")
    _validate_runtime_claim(runner, claim, controls=controls)
    claim_sha = hashlib.sha256(claim_raw).hexdigest()
    chain = checked_receipt["authority_chain"]
    runtime_consumption = checked_receipt["complete_banks"][
        "capability_consumption"
    ]
    if (
        controls["anchor_sha256"] != chain["authority_anchor_sha256"]
        or controls["release_sha256"] != chain["final_release_sha256"]
        or controls["capability_sha256"] != chain["single_use_capability_sha256"]
        or claim_sha != checked_receipt["claim_sha256"]
        or checked_source["source_bindings"]["generation_execution_anchor_sha256"]
        != controls["anchor_sha256"]
        or checked_source["source_bindings"][
            "generation_single_use_capability_sha256"
        ]
        != controls["capability_sha256"]
        or checked_source["source_bindings"]["generation_final_release_sha256"]
        != controls["release_sha256"]
        or runtime_consumption["consumed_sha256"] != controls["capability_sha256"]
        or runtime_consumption["one_use_inode"] != controls["capability_inode"]
        or runtime_consumption["original_absent"] is not True
    ):
        raise SealError("generation receipt/control/source-authority chain mismatch")
    source_utc = checked_source["issued_utc"]
    source_review_utc = checked_source_review["issued_utc"]
    if not controls["release_issued_utc"] <= source_utc <= source_review_utc:
        raise SealError("runtime release/source-authority review chronology changed")
    bank_evidence = _hash_banks_and_replay_sector_law(
        runner, args.generation_bundle, checked_receipt
    )
    return runner, {
        "source_authority_sha256": source_sha,
        "source_authority_review_sha256": source_review_sha,
        "v4_terminal_successor_record_sha256": checked_source["source_bindings"][
            "v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_bindings": dict(
            runner.V4_TERMINAL_FAILURE_BINDINGS
        ),
        "independent_incident_root_cause_sha256": checked_source[
            "source_bindings"
        ]["independent_incident_root_cause_sha256"],
        "flopscope_reset_verification_sha256": checked_source[
            "source_bindings"
        ]["flopscope_reset_verification_sha256"],
        "manifest_compatibility_scan_receipt_sha256": checked_source[
            "source_bindings"
        ]["manifest_compatibility_scan_receipt_sha256"],
        "lean_generation_law": checked_source["generation_law"],
        "source_authority_issued_utc": source_utc,
        "source_authority_review_issued_utc": source_review_utc,
        "controls": controls,
        "receipt_sha256": receipt_sha,
        "claim_sha256": claim_sha,
        "bank_evidence": bank_evidence,
    }


def build_seal(
    runner: Any, evidence: dict[str, Any], *, seal_id: str, issued_utc: str
) -> dict[str, Any]:
    _canonical_id(seal_id, "seal_id")
    seal_utc = _canonical_utc(issued_utc, "seal issued_utc")
    if seal_utc < evidence["source_authority_review_issued_utc"]:
        raise SealError("post-generation seal predates reviewed source authority")
    controls = evidence["controls"]
    banks = evidence["bank_evidence"]
    document = {
        "schema": "public-q24-signed-pair-lean-three-bank-post-generation-seal-v5",
        "status": "SEALED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANK_GENERATION",
        "seal_id": seal_id,
        "issued_by": "post_generation_sealer",
        "issued_utc": issued_utc,
        "pre_generation_source_authority_sha256": evidence[
            "source_authority_sha256"
        ],
        "pre_generation_source_authority_review_sha256": evidence[
            "source_authority_review_sha256"
        ],
        "pre_generation_v4_terminal_successor_record_sha256": evidence[
            "v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_bindings": evidence[
            "v4_terminal_failure_bindings"
        ],
        "independent_incident_root_cause_sha256": evidence[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": evidence[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": evidence[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "runtime_v5_static_bindings": dict(runner.RUNTIME_V5_HASHES),
        "generation_transaction_bindings": {
            "generation_claim_sha256": evidence["claim_sha256"],
            "generation_execution_anchor_sha256": controls["anchor_sha256"],
            "generation_final_release_sha256": controls["release_sha256"],
            "generation_single_use_capability_sha256": controls[
                "capability_sha256"
            ],
            "consumed_capability_sha256": controls["capability_sha256"],
            "generation_receipt_sha256": evidence["receipt_sha256"],
        },
        "ordered_scope_bindings": dict(runner.LEAN_ORDERED_SCOPE_BINDINGS),
        "lean_generation_law": evidence["lean_generation_law"],
        "bank_bindings": {
            "complete_bank_sha256": banks["complete_bank_sha256"],
            "A15_zero_sha256": banks["A15_zero_sha256"],
        },
        "transaction": {
            "claim_unique": True,
            "capability_input_had_one_link": True,
            "capability_consumed_by_same_inode_hard_link_then_unlink": True,
            "capability_copy_created_with_O_EXCL": False,
            "capability_inode_preserved": True,
            "capability_original_absent": True,
            "one_use": True,
            "retry": False,
        },
        "verification_evidence": {
            "generation_receipt_schema_exact_v4": True,
            "generation_receipt_and_bank_keysets_exact_v4": True,
            "generation_receipt_canonical_bytes": True,
            "all_six_bank_files_hashed": True,
            "all_A15_bytes_zero": True,
            "sector_block_prefix_law_verified": True,
            "sentinel_ranks_exact_0_343_687": True,
            "sentinel_parity_scope_authenticated_not_reexecuted": True,
            "all_row_dual_parity_claimed": False,
            "manifest_compatibility_scan_authenticated": True,
            "v4_terminal_failure_and_no_retry_authenticated": True,
            "independent_incident_root_cause_authenticated": True,
            "flopscope_post_commit_reset_law_authenticated": True,
            "runtime_receipt_declares_target_accessed_false": True,
        },
        "access_accounting": {
            "sealer_target_artifacts_opened": 0,
            "sealer_target_values_decoded": 0,
            "sealer_observer_selection_artifacts_opened": 0,
            "network": False,
            "package": False,
            "submission": False,
        },
        "limitations": {
            "generation_target_freedom_basis": (
                "FROZEN_RUNTIME_CONTROL_FLOW_PLUS_AUTHENTICATED_RUNTIME_RECEIPT_"
                "DECLARATION__NOT_EXTERNAL_PROCESS_OBSERVATION"
            ),
            "sentinel_parity_basis": (
                "AUTHENTICATED_RUNTIME_RECEIPT_DECLARATION__NOT_SEALER_REEXECUTION"
            ),
            "flopscope_reset_basis": (
                "FROZEN_RUNTIME_V5_CONTROL_FLOW_PLUS_RESET_VERIFICATION_AND_REVIEW__"
                "NOT_EXTERNAL_PROCESS_OBSERVATION"
            ),
            "flopscope_reset_execution_independently_observed": False,
            "all_row_dual_parity_claimed": False,
            "hostile_interpreter_security_claim": False,
            "review_independence_cryptographically_proven": False,
        },
    }
    runner.validate_lean_v5_post_generation_seal_object(document)
    return document


def build_review(
    runner: Any,
    evidence: dict[str, Any],
    seal_raw: bytes,
    seal: dict[str, Any],
    *,
    review_id: str,
    issued_utc: str,
) -> dict[str, Any]:
    _canonical_id(review_id, "review_id")
    _canonical_utc(issued_utc, "review issued_utc")
    seal_sha = hashlib.sha256(seal_raw).hexdigest()
    runner.validate_lean_v5_post_generation_seal_object(seal)
    expected_seal = build_seal(
        runner,
        evidence,
        seal_id=seal["seal_id"],
        issued_utc=seal["issued_utc"],
    )
    if seal != expected_seal:
        raise SealError("seal differs from independently reconstructed evidence")
    document = {
        "schema": (
            "public-q24-signed-pair-lean-three-bank-post-generation-seal-"
            "independent-review-v5"
        ),
        "status": "GO_AUTHENTICATED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANKS",
        "review_id": review_id,
        "issued_by": "independent_reviewer",
        "issued_utc": issued_utc,
        "post_generation_seal_sha256": seal_sha,
        "pre_generation_source_authority_sha256": seal[
            "pre_generation_source_authority_sha256"
        ],
        "pre_generation_source_authority_review_sha256": seal[
            "pre_generation_source_authority_review_sha256"
        ],
        "pre_generation_v4_terminal_successor_record_sha256": seal[
            "pre_generation_v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_bindings": seal["v4_terminal_failure_bindings"],
        "independent_incident_root_cause_sha256": seal[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": seal[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": seal[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "runtime_v5_static_bindings": seal["runtime_v5_static_bindings"],
        "generation_transaction_bindings": seal[
            "generation_transaction_bindings"
        ],
        "ordered_scope_bindings": seal["ordered_scope_bindings"],
        "lean_generation_law": seal["lean_generation_law"],
        "bank_bindings": seal["bank_bindings"],
        "reviewed_after_generation": True,
        "receipt_and_all_six_bank_files_independently_rehashed": True,
        "sector_block_prefix_law_independently_replayed": True,
        "sentinel_scope_and_no_all_row_overclaim_verified": True,
        "generation_receipt_and_bank_keysets_exact_v4_verified": True,
        "manifest_compatibility_scan_authenticated": True,
        "v4_terminal_failure_and_no_retry_authenticated": True,
        "independent_incident_root_cause_authenticated": True,
        "flopscope_post_commit_reset_law_authenticated_not_reexecuted": True,
        "sealer_access_accounting_verified": True,
        "runtime_target_free_declaration_authenticated_not_observed": True,
        "sentinel_parity_authenticated_not_reexecuted": True,
        "generation_provenance_authenticated": True,
        "reviewer_different_from_seal_issuer": True,
        "hostile_interpreter_security_claim": False,
    }
    runner.validate_lean_v5_post_generation_seal_review_object(
        document, seal_sha256=seal_sha, seal=seal
    )
    return document


def _write_exclusive(path: Path, document: dict[str, Any]) -> None:
    payload = canonical_json_bytes(document)
    descriptor = os.open(
        os.fspath(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise SealError("short exclusive output write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.read_bytes() != payload:
        raise SealError("durable output replay mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--issue-seal", action="store_true")
    mode.add_argument("--issue-review", action="store_true")
    parser.add_argument("--source-authority", type=Path)
    parser.add_argument("--source-authority-review", type=Path)
    parser.add_argument("--generation-anchor", type=Path)
    parser.add_argument("--generation-release", type=Path)
    parser.add_argument("--consumed-generation-capability", type=Path)
    parser.add_argument("--generation-claim", type=Path)
    parser.add_argument("--generation-bundle", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seal-id")
    parser.add_argument("--review-id")
    parser.add_argument("--issued-utc")
    parser.add_argument("--confirmation")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "public-q24-post-generation-sealer-preflight-v5",
                    "status": "STATIC_SKELETON_READY__NO_ARTIFACT_OPEN",
                    "seal_confirmation": SEAL_CONFIRMATION,
                    "review_confirmation": REVIEW_CONFIRMATION,
                    "target_artifacts_opened": 0,
                    "bank_artifacts_opened": 0,
                },
                sort_keys=True,
            )
        )
        return
    if RUNTIME_V5_PROFILE_STATUS != "EXACT_LEAN_RUNTIME_V5":
        raise SystemExit(
            "post-generation issuance held before artifact inspection: "
            "fresh Runtime V5 live roots are not bound"
        )
    required = (
        "source_authority",
        "source_authority_review",
        "generation_anchor",
        "generation_release",
        "consumed_generation_capability",
        "generation_claim",
        "generation_bundle",
        "output",
        "issued_utc",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required issue arguments: " + ",".join(missing))
    if args.issue_seal:
        if args.confirmation != SEAL_CONFIRMATION or args.seal_id is None:
            raise SystemExit("seal issuance requires exact confirmation and --seal-id")
    elif (
        args.confirmation != REVIEW_CONFIRMATION
        or args.review_id is None
        or args.seal is None
    ):
        raise SystemExit(
            "review issuance requires exact confirmation, --review-id, and --seal"
        )
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("issue output must be absent before protected evidence is read")
    _lstat_nonsymlink_directory(args.output.parent, "issue output directory")
    runner, evidence = _validated_inputs(args)
    if args.issue_seal:
        document = build_seal(
            runner, evidence, seal_id=args.seal_id, issued_utc=args.issued_utc
        )
    else:
        seal_raw, seal = _read_canonical(args.seal, "post-generation seal")
        document = build_review(
            runner,
            evidence,
            seal_raw,
            seal,
            review_id=args.review_id,
            issued_utc=args.issued_utc,
        )
    _write_exclusive(args.output, document)
    print(
        json.dumps(
            {
                "status": document["status"],
                "output": str(args.output),
                "sha256": hashlib.sha256(canonical_json_bytes(document)).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
