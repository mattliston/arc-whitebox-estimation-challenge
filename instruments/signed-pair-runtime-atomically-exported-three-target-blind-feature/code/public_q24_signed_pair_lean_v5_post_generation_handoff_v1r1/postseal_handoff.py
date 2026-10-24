#!/usr/bin/env python3
"""Fail-closed V5 atomic-bank verification and OOF V4 handoff helper V1R1.

The status mode is metadata-only and never opens the in-progress staging tree.
Protected verification is impossible until the runtime's final atomic directory
rename has published ``THREE_BANK_BUNDLE`` and removed the staging name.

After publication, verification authenticates the frozen V5/OOF roots, validates
the exact V5-over-V4 generation receipt, independently hashes all seven bundle files,
recomputes every layer and row digest, checks every float32 for finiteness, checks
every A15 byte for zero, replays the 2/8/32 eight-sector prefix law, and checks
the complete meter-vector/row-scalar alignment. It does not open targets,
weights, row manifests, scores, estimators, or training artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any


# Loading the frozen sibling modules must not mutate their directories.
sys.dont_write_bytecode = True


HERE = Path(__file__).absolute().parent
P2 = HERE.parent

TRANSACTION_ROOT = (
    P2 / "public_q24_signed_pair_lean_v5_three_bank_generation_transaction_v1"
)
BUNDLE = TRANSACTION_ROOT / "THREE_BANK_BUNDLE"
STAGING = TRANSACTION_ROOT / ".THREE_BANK_BUNDLE.staging"
FAILURE = TRANSACTION_ROOT / "THREE_BANK_RUN.FAILURE.json"
CLAIM = TRANSACTION_ROOT / "THREE_BANK_RUN.CLAIM.json"

OOF = P2 / "public_q24_signed_pair_nested_accuracy_oof_v4r1"
OOF_RUNNER = OOF / "nested_accuracy_runner.py"
OOF_SEALER = OOF / "post_generation_sealer.py"

SOURCE_AUTHORITY = (
    P2
    / "public_q24_signed_pair_lean_v5_bank_source_authority_instance_v1"
    / "BANK_SOURCE_AUTHORITY_V5.json"
)
SOURCE_AUTHORITY_REVIEW = (
    P2
    / "public_q24_signed_pair_lean_v5_bank_source_authority_review_v1"
    / "INDEPENDENT_LIVE_REVIEW_V5.json"
)
GENERATION_ANCHOR = (
    P2
    / "public_q24_signed_pair_lean_v5_three_bank_authority_instance_v1"
    / "RUNTIME_AUTHORITY_ANCHOR_V5.json"
)
GENERATION_RELEASE = (
    P2
    / "public_q24_signed_pair_lean_v5_three_bank_authority_instance_v1"
    / "FINAL_EXECUTION_RELEASE_V5.json"
)
CONSUMED_CAPABILITY = (
    P2
    / "public_q24_signed_pair_lean_v5_three_bank_authority_instance_v1"
    / "SINGLE_USE_CAPABILITY_V5.json.consumed"
)
GENERATION_COMMAND = (
    P2
    / "public_q24_signed_pair_lean_v5_three_bank_authority_instance_v1"
    / "EXACTLY_ONE_LEAN_V5_LAUNCH_COMMAND_V1.json"
)
V4_TERMINAL_LINEAGE = (
    P2
    / "public_q24_signed_pair_three_bank_runtime_v5"
    / "V4_TERMINAL_FAILURE_LINEAGE_V5.json"
)
V4_TERMINAL_SUCCESSOR = (
    P2
    / "public_q24_signed_pair_lean_v5_bank_source_authority_instance_v1"
    / "V4_TERMINAL_SUCCESSOR_RECORD_V1.json"
)
PREDECESSOR_POST_HANDOFF = (
    P2 / "public_q24_signed_pair_lean_v4_post_generation_handoff_v1"
)
PREDECESSOR_POST_HANDOFF_REVIEW = (
    P2
    / "public_q24_signed_pair_lean_v4_post_generation_handoff_review_v1"
    / "INDEPENDENT_STATIC_REVIEW_V1.json"
)
HELD_V1_HANDOFF = P2 / "public_q24_signed_pair_lean_v5_post_generation_handoff_v1"
HELD_V1_HANDOFF_REVIEW = (
    P2
    / "public_q24_signed_pair_lean_v5_post_generation_handoff_review_v1"
    / "INDEPENDENT_STATIC_REVIEW_V1.json"
)

EXPECTED_H = (16, 64, 256)
ROWS = 688
LAYERS = 13
WIDTH = 1024
SECTORS = 8
MAX_CANONICAL_JSON_BYTES = 128 << 20
AUTHORIZED_INTERPRETER = P2 / "data_gate" / ".venv" / "bin" / "python"
AUTHORIZED_INTERPRETER_VERSION = (3, 11, 5)
AUTHORIZED_INTERPRETER_SHA256 = (
    "6116d7bcb852d61f4384daa1c43488771e6794636d959f2a7065b54c92a070a1"
)
EXPECTED_FILES = {
    16: ("actions_H16.f32le", 586_153_984, "A15_zero_H16.f32le", 65_536),
    64: ("actions_H64.f32le", 2_344_615_936, "A15_zero_H64.f32le", 262_144),
    256: ("actions_H256.f32le", 9_378_463_744, "A15_zero_H256.f32le", 1_048_576),
}
RECEIPT_NAME = "THREE_BANK_GENERATION_RECEIPT.json"
EXPECTED_INVENTORY = {
    RECEIPT_NAME,
    *(name for name, _, _, _ in EXPECTED_FILES.values()),
    *(name for _, _, name, _ in EXPECTED_FILES.values()),
}

VERIFY_CONFIRMATION = "VERIFY_EXACT_ATOMIC_LEAN_RUNTIME_V5_BUNDLE"
SEAL_CONFIRMATION = "ISSUE_EXACT_LEAN_RUNTIME_V5_POST_GENERATION_SEAL"
REQUEST_CONFIRMATION = "ISSUE_EXACT_V5_POST_GENERATION_REVIEW_REQUEST"
REVIEW_CONFIRMATION = "ISSUE_EXACT_INDEPENDENT_POST_GENERATION_REVIEW_V5"
HANDOFF_CONFIRMATION = "ISSUE_EXACT_REVIEWED_V5_OOF_V4_CONSUMER_HANDOFF"
SEAL_ID = "q24-sp3b-lean-v5-postseal-v1"
REVIEW_ID = "q24-sp3b-lean-v5-postseal-review-v1"
OOF_BANK_BINDING_KEYS = {
    "source_authority_sha256",
    "source_authority_review_sha256",
    "pre_generation_v4_terminal_successor_record_sha256",
    "v4_terminal_failure_lineage_sha256",
    "independent_incident_root_cause_sha256",
    "flopscope_reset_verification_sha256",
    "manifest_compatibility_scan_receipt_sha256",
    "v4_exclusive_claim_sha256",
    "v4_terminal_failure_sha256",
    "v4_consumed_capability_sha256",
    "lean_runtime_v5_package_manifest_sha256",
    "lean_runtime_v5_independent_review_sha256",
    "generation_execution_anchor_sha256",
    "generation_final_release_sha256",
    "generation_single_use_capability_sha256",
    "generation_claim_sha256",
    "generation_receipt_sha256",
    "post_generation_seal_sha256",
    "post_generation_seal_review_sha256",
    "ordered_688_record_sha256",
    "complete_bank_sha256",
    "A15_zero_sha256",
}
OOF_STATIC_PACKAGE_BINDING_KEYS = {
    "package_manifest_sha256",
    "freeze_receipt_sha256",
    "freeze_verification_sha256",
    "static_verification_sha256",
    "runner_sha256",
    "post_generation_sealer_sha256",
    "post_generation_seal_contract_sha256",
    "v5_provenance_fixtures_sha256",
    "v4r1_authority_bindings_sha256",
    "static_verifier_sha256",
    "independent_static_review_sha256",
}

# Every byte root below existed before this helper was frozen. Dynamic bundle,
# seal, seal-review, and consumer-authority roots are intentionally absent here.
STATIC_ROOTS = {
    "runtime_v5_package_manifest": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "PACKAGE_MANIFEST.json",
        "6307642b0ebe6397e9ffb7c5d163fe29970b5644fb39f76a240fe59afbd4935d",
    ),
    "runtime_v5_freeze_receipt": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "FREEZE_RECEIPT.json",
        "0e86f6c85e29e9d87da7538694041d65cce63cd7690fd7bba48dd7d8d06ce384",
    ),
    "runtime_v5_freeze_verification": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "FREEZE_VERIFICATION.json",
        "355368e1ed6a4ec0dea54f59bbb9e3273e72e9e674eae825fbfed18f69bceff9",
    ),
    "runtime_v5_static_verification": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "STATIC_VERIFICATION.json",
        "fb771014f153980f23e4bbe1fe60d1f229bce1b11849dc09464fa747c021f4a7",
    ),
    "runtime_v5_runner": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "runtime_runner.py",
        "db30d0743940b654116d94542278b970caa3eece8d9da0d641f46e0291e0c3c4",
    ),
    "runtime_v5_independent_review": (
        P2
        / "public_q24_signed_pair_three_bank_runtime_review_v5"
        / "INDEPENDENT_STATIC_REVIEW_V5.json",
        "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e",
    ),
    "flopscope_reset_verification": (
        P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "FLOPSCOPE_RESET_VERIFICATION_V5.json",
        "9253d354914da94d193dadd43e118ef7b6afaa9a2b11769c8d599cbfd62b26af",
    ),
    "independent_incident_root_cause": (
        P2
        / "public_q24_signed_pair_lean_v4_flopscope_accumulator_incident_audit_v1"
        / "INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json",
        "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c",
    ),
    "v4_terminal_failure_lineage": (
        V4_TERMINAL_LINEAGE,
        "0dfdfff19826c163958bfc0b6866b31a9f334a7a2f62262a447b5baa3f5d7ded",
    ),
    "v4_terminal_successor_record": (
        V4_TERMINAL_SUCCESSOR,
        "4dcaa9865195caab75d8c6217ea51b94a27cc3c008c65212bed47d3b4293f42a",
    ),
    "generation_anchor": (
        GENERATION_ANCHOR,
        "4c105ab659bd5e749ae15b8d2f82603319a951aa618c80bb709c9ee0ab42fd55",
    ),
    "generation_release": (
        GENERATION_RELEASE,
        "b6fcb5c57a9890552aa9f277d81cb935d59412264fcb4001927470421520af25",
    ),
    "consumed_generation_capability": (
        CONSUMED_CAPABILITY,
        "301edd86eb5d46fdbcd7238b934565902f6d1129cb907ba6ebbb275503df1004",
    ),
    "generation_claim": (
        CLAIM,
        "4473b27bdaf3f5357c2e3e2830a820e848b5bbff236352d6c8ad0fe571bd9a57",
    ),
    "generation_launch_command": (
        GENERATION_COMMAND,
        "8110b966be52e865e34b9470e16423444a513ef428927dfd97ee24fd29e90bd2",
    ),
    "source_authority": (
        SOURCE_AUTHORITY,
        "fe74c7264767fda21c566e2cd993e96c5a61b2e5c5332734a8dbfe884b06d785",
    ),
    "source_authority_review": (
        SOURCE_AUTHORITY_REVIEW,
        "cff0066295f7b432868719e9ad274ef4a47dc0356889376df461ce3993551fef",
    ),
    "prelaunch_authority_builder_independent_review": (
        P2
        / "public_q24_signed_pair_lean_v5_prelaunch_authority_builder_review_v1"
        / "INDEPENDENT_STATIC_REVIEW_V1.json",
        "89b29a7838903d88957f0eb2397f0ef7c88821b50f3c5995cdbccfe57ef79565",
    ),
    "manifest_compatibility_scan_receipt": (
        P2
        / "public_q24_signed_pair_three_bank_runtime_v5"
        / "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json",
        "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a",
    ),
    "predecessor_v4_postseal_package_manifest": (
        PREDECESSOR_POST_HANDOFF / "PACKAGE_MANIFEST.json",
        "375d6a34c448607792b7043ef4cea4f00f0b9e34a277fda63a8664ffe4c5437c",
    ),
    "predecessor_v4_postseal_freeze_receipt": (
        PREDECESSOR_POST_HANDOFF / "FREEZE_RECEIPT.json",
        "38f6f2a9f0887ebc9ccf34632173f58382bda52ada2240dac1bfffb6efce3413",
    ),
    "predecessor_v4_postseal_freeze_verification": (
        PREDECESSOR_POST_HANDOFF / "FREEZE_VERIFICATION.json",
        "8552b4e8a5f67c7d182f4ccc810ebc3e5846f47a58c9f1bb0ddae1d74b67e24a",
    ),
    "predecessor_v4_postseal_static_verification": (
        PREDECESSOR_POST_HANDOFF / "STATIC_VERIFICATION.json",
        "09cc53c11163b591977c56565d9388ffa47f759d47124ec32f1f4aeb86ef9f43",
    ),
    "predecessor_v4_postseal_helper": (
        PREDECESSOR_POST_HANDOFF / "postseal_handoff.py",
        "5e690e42f8f3ad01510528dded88592c9f685240890b22aba20f701bdf9c2481",
    ),
    "predecessor_v4_postseal_independent_review": (
        PREDECESSOR_POST_HANDOFF_REVIEW,
        "05d74c21d19a4b15a210d9c9b530eca109579223d019b951f216284f734432f7",
    ),
    "held_v1_postseal_package_manifest": (
        HELD_V1_HANDOFF / "PACKAGE_MANIFEST.json",
        "ceda2efca3599cbd017795013ce83994a5bebf5d3a6547e9f175356fac4fdc20",
    ),
    "held_v1_postseal_freeze_receipt": (
        HELD_V1_HANDOFF / "FREEZE_RECEIPT.json",
        "52305fb1da405de88ed96d1b62553fe56e406df837b08264283a08f30ba3327e",
    ),
    "held_v1_postseal_freeze_verification": (
        HELD_V1_HANDOFF / "FREEZE_VERIFICATION.json",
        "6f6633de0f73c8d13df8b485a1ea68cc174e5451249833ce1f2c2be48ef7dc4d",
    ),
    "held_v1_postseal_static_verification": (
        HELD_V1_HANDOFF / "STATIC_VERIFICATION.json",
        "b6c89c45b6bf49a0938c8025582075f0f7ba857da87e50e00b402e991f889970",
    ),
    "held_v1_postseal_helper": (
        HELD_V1_HANDOFF / "postseal_handoff.py",
        "5d06d2de99591f0a79239fb397a99526d2246ac0a9a3cd50dbaa725370113421",
    ),
    "held_v1_postseal_independent_review": (
        HELD_V1_HANDOFF_REVIEW,
        "e0f6aa3f87f549c14272be886746a95ee4235f4dd1de2eb453aebb0e55bce201",
    ),
    "oof_v4r1_package_manifest": (
        OOF / "PACKAGE_MANIFEST.json",
        "eaa79cc6da52140ab13d452ff95c46f4beee3d5261f207b4b708fd9ec0daecca",
    ),
    "oof_v4r1_freeze_receipt": (
        OOF / "FREEZE_RECEIPT.json",
        "129a13985335b10c7f467b3461d126efdbbe12e7939630dc957c64438c6872ff",
    ),
    "oof_v4r1_freeze_verification": (
        OOF / "FREEZE_VERIFICATION.json",
        "ae55f467e0d87d888d9c4125f7d0525a74250038c8fcd96f179452c01371c04e",
    ),
    "oof_v4r1_static_verification": (
        OOF / "STATIC_VERIFICATION.json",
        "0c3931a34b62a68f152a08a8290f9d8deabe6d21941b59e6c8b7e6052d7023f3",
    ),
    "oof_v4r1_runner": (
        OOF_RUNNER,
        "e963246f4980d1e90e7192cdc9892cc3cd0ebd8a7dc181580d375c2b875e7198",
    ),
    "oof_v4r1_post_generation_sealer": (
        OOF_SEALER,
        "e9b70c1777fbc54c474621c7107acc4ef15d691a63b6ad0085f545e572086290",
    ),
    "oof_v4r1_post_generation_seal_contract": (
        OOF / "POST_GENERATION_SEAL_CONTRACT.json",
        "72f2fbc3d6ab44ac25427aa725fe1a164f8611a829e4c8be61efa8dc51913930",
    ),
    "oof_v4r1_v5_provenance_fixtures": (
        OOF / "v5_provenance_fixtures.py",
        "eb52c54080c14356c571b87bd859e1a25d4c73fcd199369e21af29e2ef82d6b4",
    ),
    "oof_v4r1_authority_bindings": (
        OOF / "V4R1_AUTHORITY_BINDINGS.json",
        "6b213850137f6610a359850144d0b5857eab31c27b68cee41ff674f5ee129fac",
    ),
    "oof_v4r1_static_verifier": (
        OOF / "verify_static.py",
        "aefb0ae4360750a085adeee1b311cdc7ac807acf8646afee09f14173b51f490d",
    ),
    "oof_v4r1_independent_static_review": (
        P2
        / "public_q24_signed_pair_nested_accuracy_oof_review_v4r1"
        / "INDEPENDENT_STATIC_REVIEW_V4R1.json",
        "509eb18e66ab1590a756791028a787de7a07a5a37c9aa1c14e8dd628cac21b61",
    ),
}


class HandoffError(RuntimeError):
    """Fail-closed handoff rejection."""


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
        raise HandoffError(f"{label} contains nonfinite JSON constant {value}")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HandoffError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"{label} is not canonical ASCII JSON") from error
    if raw != canonical_json_bytes(value):
        raise HandoffError(f"{label} bytes are not canonical JSON")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat_path(path: Path, label: str) -> tuple[Path, os.stat_result]:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    metadata: os.stat_result | None = None
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise HandoffError(f"{label} is absent") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HandoffError(f"{label} contains a symlink component")
    if metadata is None:
        raise HandoffError(f"{label} has no terminal component")
    return absolute, metadata


def _regular_one_link(path: Path, label: str, size: int | None = None) -> Path:
    absolute, metadata = _lstat_path(path, label)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise HandoffError(f"{label} is not a one-link regular file")
    if size is not None and metadata.st_size != size:
        raise HandoffError(f"{label} byte count changed")
    return absolute


def _directory(path: Path, label: str) -> Path:
    absolute, metadata = _lstat_path(path, label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise HandoffError(f"{label} is not a directory")
    return absolute


def _open_checked(
    path: Path, label: str, size: int | None = None
) -> tuple[int, int]:
    absolute = _regular_one_link(path, label, size)
    before = os.lstat(absolute)
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_dev != before.st_dev
        or observed.st_ino != before.st_ino
        or observed.st_size != before.st_size
    ):
        os.close(descriptor)
        raise HandoffError(f"{label} identity changed while opening")
    return descriptor, int(observed.st_size)


def _read_descriptor(descriptor: int, count: int, label: str) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        block = os.read(descriptor, remaining)
        if not block:
            raise HandoffError(f"{label} ended before its authenticated size")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def _read_file(path: Path, label: str) -> bytes:
    descriptor, size = _open_checked(path, label)
    try:
        raw = _read_descriptor(descriptor, size, label)
        if os.read(descriptor, 1):
            raise HandoffError(f"{label} grew while reading")
        return raw
    finally:
        os.close(descriptor)


def _read_canonical(path: Path, label: str) -> tuple[bytes, Any]:
    descriptor, size = _open_checked(path, label)
    if size > MAX_CANONICAL_JSON_BYTES:
        os.close(descriptor)
        raise HandoffError(f"{label} exceeds frozen JSON byte bound")
    try:
        raw = _read_descriptor(descriptor, size, label)
        if os.read(descriptor, 1):
            raise HandoffError(f"{label} grew while reading")
    finally:
        os.close(descriptor)
    return raw, _strict_json(raw, label)


def _sha256(path: Path, label: str, size: int | None = None) -> str:
    descriptor, observed_size = _open_checked(path, label, size)
    digest = hashlib.sha256()
    try:
        remaining = observed_size
        while remaining:
            block = os.read(descriptor, min(8 << 20, remaining))
            if not block:
                raise HandoffError(f"{label} ended before its authenticated size")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise HandoffError(f"{label} grew while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def verify_authorized_interpreter() -> dict[str, Any]:
    if tuple(sys.version_info[:3]) != AUTHORIZED_INTERPRETER_VERSION:
        raise HandoffError("protected V5 postseal mode requires pinned CPython 3.11.5")
    if _absolute(Path(sys.executable)) != _absolute(AUTHORIZED_INTERPRETER):
        raise HandoffError("protected V5 postseal mode used the wrong interpreter path")
    resolved = Path(os.path.realpath(os.fspath(AUTHORIZED_INTERPRETER)))
    observed = _sha256(resolved, "resolved authorized interpreter")
    if observed != AUTHORIZED_INTERPRETER_SHA256:
        raise HandoffError("protected V5 postseal interpreter bytes changed")
    return {
        "implementation": "CPython",
        "version": list(AUTHORIZED_INTERPRETER_VERSION),
        "invocation_path": str(AUTHORIZED_INTERPRETER),
        "resolved_binary_sha256": observed,
    }


def _write_exclusive(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise HandoffError("output must be absent")
    parent = _directory(path.parent, "output parent")
    payload = canonical_json_bytes(value)
    descriptor = os.open(
        parent / path.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise HandoffError("short exclusive output write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if _read_file(path, "issued output") != payload:
        raise HandoffError("durable output replay mismatch")
    return hashlib.sha256(payload).hexdigest()


def _load_frozen(path: Path, expected_sha: str, name: str) -> Any:
    if _sha256(path, name) != expected_sha:
        raise HandoffError(f"{name} frozen hash changed")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise HandoffError(f"cannot load {name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def verify_static_roots() -> dict[str, str]:
    verify_authorized_interpreter()
    observed: dict[str, str] = {}
    for name, (path, expected) in STATIC_ROOTS.items():
        actual = _sha256(path, name)
        if actual != expected:
            raise HandoffError(f"frozen root changed: {name}")
        observed[name] = actual
    original = CONSUMED_CAPABILITY.with_name(
        CONSUMED_CAPABILITY.name.removesuffix(".consumed")
    )
    if original.exists() or original.is_symlink():
        raise HandoffError("original V5 capability name reappeared")
    return observed


def _validate_generation_command() -> str:
    raw, command = _read_canonical(GENERATION_COMMAND, "V5 launch command")
    exact_keys = {
        "schema",
        "status",
        "command_count",
        "argv",
        "execute_via_argv_without_shell",
        "working_directory",
        "transaction_root_must_be_existing_empty_nonsymlink_directory",
        "authority_anchor_sha256",
        "final_release_sha256",
        "single_use_capability_sha256",
        "manifest_compatibility_scan_receipt_sha256",
        "source_authority_and_independent_live_review_required_before_launch",
        "no_retry",
        "v4_terminal_attempt_reused",
        "v4_staging_used_as_input",
        "shell_rendering_for_display_only",
    }
    if type(command) is not dict or set(command) != exact_keys:
        raise HandoffError("V5 launch-command field inventory changed")
    argv = command["argv"]
    if (
        command["schema"]
        != "public-q24-lean-v5-three-bank-exactly-one-launch-command-v1"
        or command["status"]
        != "PREPARED_NOT_AUTHORIZED_UNTIL_V5_SOURCE_REVIEW_GO"
        or command["command_count"] != 1
        or command["execute_via_argv_without_shell"] is not True
        or type(argv) is not list
        or len(argv) != 28
        or argv[0]
        != str(P2 / "data_gate" / ".venv" / "bin" / "python")
        or argv[1]
        != str(P2 / "public_q24_signed_pair_three_bank_runtime_v5" / "runtime_runner.py")
        or argv[-2:] != ["--transaction-root", str(TRANSACTION_ROOT)]
        or command["working_directory"] != str(P2.parents[3])
        or command["authority_anchor_sha256"]
        != STATIC_ROOTS["generation_anchor"][1]
        or command["final_release_sha256"]
        != STATIC_ROOTS["generation_release"][1]
        or command["single_use_capability_sha256"]
        != STATIC_ROOTS["consumed_generation_capability"][1]
        or command["manifest_compatibility_scan_receipt_sha256"]
        != STATIC_ROOTS["manifest_compatibility_scan_receipt"][1]
        or command[
            "transaction_root_must_be_existing_empty_nonsymlink_directory"
        ]
        is not True
        or command[
            "source_authority_and_independent_live_review_required_before_launch"
        ]
        is not True
        or command["no_retry"] is not True
        or command["v4_terminal_attempt_reused"] is not False
        or command["v4_staging_used_as_input"] is not False
    ):
        raise HandoffError("V5 launch-command control binding changed")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != STATIC_ROOTS["generation_launch_command"][1]:
        raise HandoffError("V5 launch-command hash changed")
    return observed


def transaction_status() -> dict[str, Any]:
    """Inspect names and lstat metadata only; never descend into staging."""
    root = _directory(TRANSACTION_ROOT, "V5 transaction root")
    with os.scandir(root) as iterator:
        entries = {entry.name: entry for entry in iterator}
    names = set(entries)
    if CLAIM.name not in names:
        raise HandoffError("V5 transaction claim is absent")
    claim_stat = entries[CLAIM.name].stat(follow_symlinks=False)
    if not stat.S_ISREG(claim_stat.st_mode) or claim_stat.st_nlink != 1:
        raise HandoffError("V5 transaction claim metadata changed")

    staging_present = STAGING.name in names
    bundle_present = BUNDLE.name in names
    failure_present = FAILURE.name in names
    if staging_present:
        staging_stat = entries[STAGING.name].stat(follow_symlinks=False)
        if not stat.S_ISDIR(staging_stat.st_mode):
            raise HandoffError("V5 staging name is not a directory")
    if bundle_present:
        bundle_stat = entries[BUNDLE.name].stat(follow_symlinks=False)
        if not stat.S_ISDIR(bundle_stat.st_mode):
            raise HandoffError("published V5 bundle name is not a directory")
    if failure_present:
        failure_stat = entries[FAILURE.name].stat(follow_symlinks=False)
        if not stat.S_ISREG(failure_stat.st_mode) or failure_stat.st_nlink != 1:
            raise HandoffError("V5 failure receipt metadata changed")

    if failure_present:
        if bundle_present:
            raise HandoffError("terminal failure and published bundle coexist")
        allowed = {CLAIM.name, FAILURE.name}
        if staging_present:
            allowed.add(STAGING.name)
        if names != allowed:
            raise HandoffError("terminal V5 transaction inventory drift")
        state = "TERMINAL_FAILURE_NO_RETRY"
    elif bundle_present:
        if staging_present or names != {CLAIM.name, BUNDLE.name}:
            raise HandoffError("atomic V5 publication inventory drift")
        bundle_root = _directory(BUNDLE, "published V5 bundle")
        with os.scandir(bundle_root) as iterator:
            bundle_names = {entry.name for entry in iterator}
        if bundle_names != EXPECTED_INVENTORY:
            raise HandoffError("published V5 bundle inventory is not exactly seven files")
        state = "READY_ATOMIC_BUNDLE_PUBLISHED"
    elif staging_present:
        if names != {CLAIM.name, STAGING.name}:
            raise HandoffError("running V5 transaction inventory drift")
        state = "RUNNING_STAGING_UNOPENED"
    elif names == {CLAIM.name}:
        state = "CLAIMED_TRANSITION_STAGING_NOT_OBSERVED"
    else:
        raise HandoffError("unrecognized V5 transaction inventory")
    return {
        "schema": "public-q24-lean-v5-atomic-bundle-status-v1",
        "status": state,
        "transaction_root": str(root),
        "claim_present": True,
        "staging_present": staging_present,
        "bundle_present": bundle_present,
        "failure_present": failure_present,
        "staging_entries_opened": 0,
        "bank_files_opened": 0,
        "target_artifacts_opened": 0,
    }


def _require_ready_bundle() -> Path:
    status = transaction_status()
    if status["status"] != "READY_ATOMIC_BUNDLE_PUBLISHED":
        raise HandoffError(f"atomic V5 bundle is not ready: {status['status']}")
    return _directory(BUNDLE, "published V5 bundle")


def _require_current_atomic_ready(purpose: str) -> dict[str, Any]:
    """Reassert the complete atomic-publication gate immediately before issuance."""
    status = transaction_status()
    if (
        status["status"] != "READY_ATOMIC_BUNDLE_PUBLISHED"
        or status["claim_present"] is not True
        or status["staging_present"] is not False
        or status["bundle_present"] is not True
        or status["failure_present"] is not False
        or status["staging_entries_opened"] != 0
        or status["bank_files_opened"] != 0
        or status["target_artifacts_opened"] != 0
    ):
        raise HandoffError(f"{purpose} requires the current exact atomic V5 bundle")
    return status


def _validate_receipt_and_chain(
    runner: Any, sealer: Any, receipt: Any, receipt_raw: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checked_receipt, mlp_ids = runner.validate_generation_receipt_object(receipt)
    if mlp_ids != [row["mlp_id"] for row in checked_receipt["row_receipts"]]:
        raise HandoffError("receipt mlp_id crosswalk changed")
    source_raw, source = _read_canonical(SOURCE_AUTHORITY, "V5 source authority")
    source_review_raw, source_review = _read_canonical(
        SOURCE_AUTHORITY_REVIEW, "V5 source authority review"
    )
    source_sha = hashlib.sha256(source_raw).hexdigest()
    source_review_sha = hashlib.sha256(source_review_raw).hexdigest()
    checked_source = runner.validate_lean_v5_source_authority_object(source)
    runner.validate_lean_v5_source_authority_review_object(
        source_review,
        authority_sha256=source_sha,
        authority=checked_source,
    )
    controls = sealer._validate_runtime_control_documents(
        runner,
        GENERATION_ANCHOR,
        GENERATION_RELEASE,
        CONSUMED_CAPABILITY,
    )
    claim_raw, claim = _read_canonical(CLAIM, "V5 generation claim")
    sealer._validate_runtime_claim(runner, claim, controls=controls)
    claim_sha = hashlib.sha256(claim_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    chain = checked_receipt["authority_chain"]
    consumption = checked_receipt["complete_banks"]["capability_consumption"]
    bindings = checked_source["source_bindings"]
    command_sha = _validate_generation_command()
    if (
        source_sha != STATIC_ROOTS["source_authority"][1]
        or source_review_sha != STATIC_ROOTS["source_authority_review"][1]
        or claim_sha != STATIC_ROOTS["generation_claim"][1]
        or controls["anchor_sha256"] != chain["authority_anchor_sha256"]
        or controls["release_sha256"] != chain["final_release_sha256"]
        or controls["capability_sha256"] != chain["single_use_capability_sha256"]
        or claim_sha != checked_receipt["claim_sha256"]
        or consumption["consumed_sha256"] != controls["capability_sha256"]
        or consumption["one_use_inode"] != controls["capability_inode"]
        or consumption["original_absent"] is not True
        or bindings["generation_execution_anchor_sha256"]
        != controls["anchor_sha256"]
        or bindings["generation_final_release_sha256"]
        != controls["release_sha256"]
        or bindings["generation_single_use_capability_sha256"]
        != controls["capability_sha256"]
        or bindings["v5_runtime_package_manifest_sha256"]
        != STATIC_ROOTS["runtime_v5_package_manifest"][1]
        or bindings["v5_runtime_independent_review_sha256"]
        != STATIC_ROOTS["runtime_v5_independent_review"][1]
        or bindings["generation_runner_sha256"]
        != STATIC_ROOTS["runtime_v5_runner"][1]
        or bindings["flopscope_reset_verification_sha256"]
        != STATIC_ROOTS["flopscope_reset_verification"][1]
        or bindings["independent_incident_root_cause_sha256"]
        != STATIC_ROOTS["independent_incident_root_cause"][1]
        or bindings["v4_terminal_failure_lineage_sha256"]
        != STATIC_ROOTS["v4_terminal_failure_lineage"][1]
        or bindings["v4_terminal_successor_record_sha256"]
        != STATIC_ROOTS["v4_terminal_successor_record"][1]
        or bindings["authority_builder_independent_review_sha256"]
        != STATIC_ROOTS["prelaunch_authority_builder_independent_review"][1]
        or bindings["manifest_compatibility_scan_receipt_sha256"]
        != STATIC_ROOTS["manifest_compatibility_scan_receipt"][1]
    ):
        raise HandoffError("V5 receipt/control/source/reset chain mismatch")
    compatibility = {
        "receipt_schema_exact_v4": (
            checked_receipt["schema"]
            == "public-q24-signed-pair-three-bank-generation-receipt-v4"
        ),
        "receipt_top_level_keyset_exact_v4": (
            set(checked_receipt) == runner.LEAN_RECEIPT_KEYS
        ),
        "runtime_profile_exact_lean_v5": (
            runner.GENERATION_RECEIPT_PROFILE == "EXACT_LEAN_RUNTIME_V5"
        ),
        "nested_reset_field_absent": "flopscope_reset" not in checked_receipt,
        "post_commit_reset_law_authenticated": (
            checked_source["generation_law"]
            == runner.expected_lean_v5_generation_law()
        ),
        "flopscope_reset_verification_sha256": bindings[
            "flopscope_reset_verification_sha256"
        ],
        "reset_execution_independently_observed": False,
        "authorized_interpreter_version": list(AUTHORIZED_INTERPRETER_VERSION),
        "authorized_interpreter_sha256": AUTHORIZED_INTERPRETER_SHA256,
        "oof_v4r1_science_identity_uses_exact_source_bytes": True,
        "interpreter_dependent_ast_dump_used_as_authority": False,
    }
    if not all(
        compatibility[key] is True
        for key in (
            "receipt_schema_exact_v4",
            "receipt_top_level_keyset_exact_v4",
            "runtime_profile_exact_lean_v5",
            "nested_reset_field_absent",
            "post_commit_reset_law_authenticated",
        )
    ):
        raise HandoffError("V5 receipt is not reset-compatible exact V4 schema")
    return checked_receipt, {
        "source_authority_sha256": source_sha,
        "source_authority_review_sha256": source_review_sha,
        "generation_claim_sha256": claim_sha,
        "generation_receipt_sha256": receipt_sha,
        "generation_execution_anchor_sha256": controls["anchor_sha256"],
        "generation_final_release_sha256": controls["release_sha256"],
        "generation_single_use_capability_sha256": controls["capability_sha256"],
        "pre_generation_v4_terminal_successor_record_sha256": bindings[
            "v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_lineage_sha256": bindings[
            "v4_terminal_failure_lineage_sha256"
        ],
        "independent_incident_root_cause_sha256": bindings[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": bindings[
            "flopscope_reset_verification_sha256"
        ],
        "v4_exclusive_claim_sha256": bindings["v4_exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": bindings["v4_terminal_failure_sha256"],
        "v4_consumed_capability_sha256": bindings[
            "v4_consumed_capability_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": bindings[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "generation_launch_command_sha256": command_sha,
    }, compatibility


def verify_layer_payloads(
    payloads: dict[int, bytes], *, np: Any, width: int = WIDTH
) -> int:
    if type(payloads) is not dict or set(payloads) != set(EXPECTED_H):
        raise HandoffError("layer payload H inventory changed")
    arrays: dict[int, Any] = {}
    sectors: dict[int, Any] = {}
    for h in EXPECTED_H:
        if len(payloads[h]) != width * h * 4:
            raise HandoffError(f"H{h} layer byte count changed")
        floats = np.frombuffer(payloads[h], dtype=np.dtype("<f4"))
        if not bool(np.isfinite(floats).all()):
            raise HandoffError(f"H{h} layer contains a nonfinite float32")
        arrays[h] = floats
        sectors[h] = np.frombuffer(payloads[h], dtype=np.uint8).reshape(
            width, SECTORS, (h // SECTORS) * 4
        )
    comparisons = (
        (16, 64, sectors[16], sectors[64][:, :, : 2 * 4]),
        (16, 256, sectors[16], sectors[256][:, :, : 2 * 4]),
        (64, 256, sectors[64], sectors[256][:, :, : 8 * 4]),
    )
    for narrow_h, wide_h, narrow, prefix in comparisons:
        if not bool(np.array_equal(narrow, prefix)):
            raise HandoffError(f"sector-prefix mismatch H{narrow_h}->H{wide_h}")
    return len(comparisons)


def _scan_all_seven_files(
    bundle: Path, receipt_raw: bytes, receipt: dict[str, Any]
) -> dict[str, Any]:
    # Imported only after atomic publication and receipt/control authentication.
    import numpy as np

    rows = receipt["row_receipts"]
    complete = receipt["complete_banks"]["banks"]
    descriptors: dict[int, int] = {}
    complete_digests = {h: hashlib.sha256() for h in EXPECTED_H}
    complete_bytes_read = {h: 0 for h in EXPECTED_H}
    file_sha256 = {RECEIPT_NAME: hashlib.sha256(receipt_raw).hexdigest()}
    layer_hashes = 0
    row_hashes = 0
    prefix_checks = 0
    finite_values = 0
    try:
        for h in EXPECTED_H:
            tape_name, tape_size, _, _ = EXPECTED_FILES[h]
            item = complete[str(h)]
            if item["path"] != tape_name:
                raise HandoffError(f"H{h} receipt path changed")
            descriptor, _ = _open_checked(bundle / tape_name, f"H{h} bank", tape_size)
            descriptors[h] = descriptor
        for rank in range(ROWS):
            row_digests = {h: hashlib.sha256() for h in EXPECTED_H}
            for layer_slot in range(LAYERS):
                payloads: dict[int, bytes] = {}
                for h in EXPECTED_H:
                    count = WIDTH * h * 4
                    payload = _read_descriptor(
                        descriptors[h], count, f"H{h}/rank={rank}/layer={layer_slot}"
                    )
                    payloads[h] = payload
                    complete_digests[h].update(payload)
                    row_digests[h].update(payload)
                    complete_bytes_read[h] += len(payload)
                    observed_layer = hashlib.sha256(payload).hexdigest()
                    claimed_layer = rows[rank]["banks"][str(h)][
                        "layer_action_bytes_sha256"
                    ][layer_slot]
                    if observed_layer != claimed_layer:
                        raise HandoffError(
                            f"layer hash mismatch H{h}/rank={rank}/layer={layer_slot}"
                        )
                    layer_hashes += 1
                    finite_values += WIDTH * h
                prefix_checks += verify_layer_payloads(payloads, np=np)
            for h in EXPECTED_H:
                if (
                    row_digests[h].hexdigest()
                    != rows[rank]["banks"][str(h)]["action_bytes_sha256"]
                ):
                    raise HandoffError(f"row hash mismatch H{h}/rank={rank}")
                row_hashes += 1
        for h in EXPECTED_H:
            if os.read(descriptors[h], 1):
                raise HandoffError(f"H{h} bank has trailing bytes")
            observed = complete_digests[h].hexdigest()
            if observed != complete[str(h)]["sha256"]:
                raise HandoffError(f"complete hash mismatch H{h}")
            file_sha256[EXPECTED_FILES[h][0]] = observed
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    zero_bytes_read: dict[int, int] = {}
    for h in EXPECTED_H:
        _, _, zero_name, zero_size = EXPECTED_FILES[h]
        if complete[str(h)]["A15_zero_path"] != zero_name:
            raise HandoffError(f"H{h} A15 receipt path changed")
        descriptor, _ = _open_checked(bundle / zero_name, f"H{h} A15", zero_size)
        digest = hashlib.sha256()
        count = 0
        try:
            remaining = zero_size
            while remaining:
                block = os.read(descriptor, min(8 << 20, remaining))
                if not block:
                    raise HandoffError(f"H{h} A15 ended early")
                if any(block):
                    raise HandoffError(f"H{h} A15 contains a nonzero byte")
                digest.update(block)
                count += len(block)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise HandoffError(f"H{h} A15 has trailing bytes")
        finally:
            os.close(descriptor)
        observed = digest.hexdigest()
        if observed != complete[str(h)]["A15_zero_sha256"]:
            raise HandoffError(f"H{h} A15 complete hash mismatch")
        for rank in range(ROWS):
            if (
                rows[rank]["banks"][str(h)]["A15_zero_bytes_sha256"]
                != observed
            ):
                raise HandoffError(f"H{h} A15 row hash mismatch rank={rank}")
        file_sha256[zero_name] = observed
        zero_bytes_read[h] = count

    if set(file_sha256) != EXPECTED_INVENTORY:
        raise HandoffError("not all seven V5 bundle files were independently hashed")
    return {
        "file_sha256": file_sha256,
        "complete_bank_bytes_read": {
            str(h): complete_bytes_read[h] for h in EXPECTED_H
        },
        "A15_bytes_read": {str(h): zero_bytes_read[h] for h in EXPECTED_H},
        "layer_hashes_recomputed": layer_hashes,
        "row_hashes_recomputed": row_hashes,
        "finite_float32_values_checked": finite_values,
        "vectorized_sector_prefix_checks": prefix_checks,
        "logical_unit_sector_prefix_comparisons": ROWS
        * LAYERS
        * WIDTH
        * SECTORS
        * 3,
        "all_float32_finite": True,
        "all_A15_bytes_zero": True,
        "all_row_hashes_match_receipt": True,
        "all_layer_hashes_match_receipt": True,
        "all_sector_prefixes_byte_identical": True,
        "maximum_bank_layer_payload_bytes_buffered": WIDTH
        * sum(EXPECTED_H)
        * 4,
        "bank_hash_block_bytes": 8 << 20,
        "maximum_canonical_json_bytes": MAX_CANONICAL_JSON_BYTES,
        "resource_bound": (
            "BANK_PAYLOAD_STREAMING__ONE_THREE_H_LAYER__"
            "CANONICAL_JSON_CAPPED_128MIB"
        ),
    }


def _sentinel_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = receipt["row_receipts"]
    sentinels = [rank for rank, row in enumerate(rows) if row["sentinel_rank"]]
    if sentinels != [0, 343, 687]:
        raise HandoffError("sentinel rank inventory changed")
    non_sentinel_nulls = 0
    for rank, row in enumerate(rows):
        if rank in sentinels:
            comparator = row["source_action_comparator"]
            parent = row["five_component_parent_comparator"]
            if (
                row["comparator_mode"] != "SENTINEL_DUAL_PARITY"
                or comparator["all_action_and_A15_bytes_equal"] is not True
                or parent["all_five_component_bytes_equal"] is not True
                or parent["base_component_sha256"]
                != parent["capture_component_sha256"]
                or row["h256_prefix_derivation"][
                    "sentinel_derived_matches_literal_and_reference"
                ]
                is not True
            ):
                raise HandoffError(f"sentinel receipt claim failed rank={rank}")
            for h in EXPECTED_H:
                bank = row["banks"][str(h)]
                for side in ("literal", "reference"):
                    claimed = comparator[side][str(h)]
                    if (
                        claimed["action_sha256"] != bank["action_bytes_sha256"]
                        or claimed["A15_zero_sha256"]
                        != bank["A15_zero_bytes_sha256"]
                    ):
                        raise HandoffError(
                            f"sentinel comparator/bank crosswalk failed rank={rank}/H{h}"
                        )
        else:
            if (
                row["comparator_mode"] != "LEAN_CAPTURE_LITERAL_ONLY"
                or row["source_action_comparator"] is not None
                or row["five_component_parent_comparator"] is not None
                or row["h256_prefix_derivation"][
                    "sentinel_derived_matches_literal_and_reference"
                ]
                is not None
            ):
                raise HandoffError(f"non-sentinel parity overclaim rank={rank}")
            non_sentinel_nulls += 1
    final = receipt["complete_banks"]
    if (
        final["sentinel_ranks"] != sentinels
        or final["literal_reference_bytes_equal_sentinels"] is not True
        or final["base_capture_five_components_equal_sentinels"] is not True
        or final["h256_prefix_derivation_complete_all_rows"] is not True
    ):
        raise HandoffError("complete receipt sentinel declarations changed")
    return {
        "sentinel_ranks": sentinels,
        "sentinel_receipt_claims_authenticated": True,
        "sentinel_bank_digest_crosswalk_verified": True,
        "non_sentinel_rows_with_null_dual_parity_evidence": non_sentinel_nulls,
        "all_row_dual_parity_claimed": False,
        "sentinel_comparators_reexecuted_by_handoff_verifier": False,
    }


def verify_meter_vector_alignment(
    receipt: dict[str, Any], *, expected_rows: int = ROWS
) -> dict[str, Any]:
    rows = receipt["row_receipts"]
    vectors = receipt["complete_banks"]["exact_meters"]
    expected_names = {
        "complete_flops_by_rank",
        "complete_calls_by_rank",
        "base_parent_flops_by_rank",
        "base_parent_calls_by_rank",
        "capture_parent_flops_by_rank",
        "capture_parent_calls_by_rank",
        "literal_source_hook_flops_by_rank",
        "literal_source_hook_calls_by_rank",
        "reference_source_hook_flops_by_rank",
        "reference_source_hook_calls_by_rank",
        "literal_workspace_flops_by_rank",
        "literal_workspace_calls_by_rank",
        "reference_workspace_flops_by_rank",
        "reference_workspace_calls_by_rank",
        "peak_rss_bytes_by_rank",
        "wall_seconds_by_rank",
        "residual_seconds_by_rank",
        "reference_source_hook_capture_count_by_rank",
    }
    if (
        type(rows) is not list
        or len(rows) != expected_rows
        or type(vectors) is not dict
        or set(vectors) != expected_names
    ):
        raise HandoffError("exact meter-vector inventory or row count changed")
    comparisons = 0
    float_values = 0
    integer_values = 0
    for name in sorted(expected_names):
        vector = vectors[name]
        scalar_name = name.removesuffix("_by_rank")
        if type(vector) is not list or len(vector) != expected_rows:
            raise HandoffError(f"meter vector length changed: {name}")
        is_float = name in {"wall_seconds_by_rank", "residual_seconds_by_rank"}
        for rank, value in enumerate(vector):
            if is_float:
                if type(value) is not float or not math.isfinite(value) or value < 0.0:
                    raise HandoffError(f"meter vector float changed: {name}/{rank}")
                float_values += 1
            else:
                if type(value) is not int or value < 0:
                    raise HandoffError(f"meter vector integer changed: {name}/{rank}")
                integer_values += 1
            if value != rows[rank]["exact_meters"][scalar_name]:
                raise HandoffError(f"meter vector/row mismatch: {name}/{rank}")
            comparisons += 1
    return {
        "meter_vector_names": sorted(expected_names),
        "meter_vector_count": len(expected_names),
        "meter_vector_length": expected_rows,
        "meter_vector_row_scalar_comparisons": comparisons,
        "finite_nonnegative_float_meter_values": float_values,
        "nonnegative_integer_meter_values": integer_values,
        "all_meter_vectors_align_with_row_scalars": True,
    }


def verify_bundle() -> dict[str, Any]:
    bundle = _require_ready_bundle()
    static_roots = verify_static_roots()
    runner = _load_frozen(
        OOF_RUNNER, STATIC_ROOTS["oof_v4r1_runner"][1], "frozen_oof_v4r1_runner"
    )
    sealer = _load_frozen(
        OOF_SEALER,
        STATIC_ROOTS["oof_v4r1_post_generation_sealer"][1],
        "frozen_oof_v4r1_sealer",
    )
    receipt_raw, receipt = _read_canonical(
        bundle / RECEIPT_NAME, "V5 generation receipt"
    )
    checked_receipt, chain, compatibility = _validate_receipt_and_chain(
        runner, sealer, receipt, receipt_raw
    )
    scan = _scan_all_seven_files(bundle, receipt_raw, checked_receipt)
    sentinels = _sentinel_evidence(checked_receipt)
    meters = verify_meter_vector_alignment(checked_receipt)
    if transaction_status()["status"] != "READY_ATOMIC_BUNDLE_PUBLISHED":
        raise HandoffError("atomic V5 publication changed during verification")
    return {
        "schema": "public-q24-lean-v5-post-generation-handoff-verification-v1",
        "status": "PASS_EXACT_ATOMIC_V5_BUNDLE_READY_FOR_CANONICAL_SEAL",
        "issued_by": "postseal_handoff_verifier",
        "issued_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "atomic_publication": {
            "transaction_root": str(TRANSACTION_ROOT),
            "published_bundle": str(bundle),
            "claim_present": True,
            "staging_absent": True,
            "failure_absent": True,
            "exact_seven_file_inventory": True,
            "publish_law": "ONE_ATOMIC_DIRECTORY_RENAME",
        },
        "static_roots": static_roots,
        "generation_chain": chain,
        "receipt_compatibility": compatibility,
        "bundle_evidence": scan,
        "sentinel_evidence": sentinels,
        "meter_evidence": meters,
        "access_accounting": {
            "bundle_files_opened": 7,
            "staging_entries_opened": 0,
            "target_artifacts_opened": 0,
            "target_values_decoded": 0,
            "weight_arrays_opened": 0,
            "row_manifests_opened": 0,
            "scores_opened": 0,
            "training_runs": 0,
            "estimator_runs": 0,
            "submissions": 0,
        },
        "limitations": {
            "runtime_target_freedom": "AUTHENTICATED_RECEIPT_AND_FROZEN_CONTROL_FLOW_NOT_EXTERNAL_PROCESS_TRACING",
            "sentinel_parity": "AUTHENTICATED_RECEIPT_DIGEST_CROSSWALK_NOT_COMPARATOR_REEXECUTION",
            "hostile_interpreter_security_claim": False,
        },
    }


def _validate_verification(value: Any) -> dict[str, Any]:
    exact_keys = {
        "schema",
        "status",
        "issued_by",
        "issued_utc",
        "atomic_publication",
        "static_roots",
        "generation_chain",
        "receipt_compatibility",
        "bundle_evidence",
        "sentinel_evidence",
        "meter_evidence",
        "access_accounting",
        "limitations",
    }
    if type(value) is not dict or set(value) != exact_keys:
        raise HandoffError("handoff verification field inventory changed")
    try:
        parsed_utc = datetime.strptime(
            value["issued_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as error:
        raise HandoffError("handoff verification issued_utc changed") from error
    if (
        parsed_utc.strftime("%Y-%m-%dT%H:%M:%SZ") != value["issued_utc"]
        or value["schema"]
        != "public-q24-lean-v5-post-generation-handoff-verification-v1"
        or value["status"]
        != "PASS_EXACT_ATOMIC_V5_BUNDLE_READY_FOR_CANONICAL_SEAL"
        or value["issued_by"] != "postseal_handoff_verifier"
    ):
        raise HandoffError("handoff verification is not a passing V1 receipt")
    if value["static_roots"] != {
        name: expected for name, (_, expected) in STATIC_ROOTS.items()
    }:
        raise HandoffError("handoff verification static roots changed")
    atomic = value["atomic_publication"]
    if atomic != {
        "transaction_root": str(TRANSACTION_ROOT),
        "published_bundle": str(BUNDLE),
        "claim_present": True,
        "staging_absent": True,
        "failure_absent": True,
        "exact_seven_file_inventory": True,
        "publish_law": "ONE_ATOMIC_DIRECTORY_RENAME",
    }:
        raise HandoffError("handoff verification atomic-publication evidence changed")
    chain = value["generation_chain"]
    chain_keys = {
        "source_authority_sha256",
        "source_authority_review_sha256",
        "generation_claim_sha256",
        "generation_receipt_sha256",
        "generation_execution_anchor_sha256",
        "generation_final_release_sha256",
        "generation_single_use_capability_sha256",
        "pre_generation_v4_terminal_successor_record_sha256",
        "v4_terminal_failure_lineage_sha256",
        "independent_incident_root_cause_sha256",
        "flopscope_reset_verification_sha256",
        "v4_exclusive_claim_sha256",
        "v4_terminal_failure_sha256",
        "v4_consumed_capability_sha256",
        "manifest_compatibility_scan_receipt_sha256",
        "generation_launch_command_sha256",
    }
    if set(chain) != chain_keys or any(
        type(item) is not str
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in chain.values()
    ):
        raise HandoffError("handoff verification generation-chain hashes changed")
    expected_static_chain = {
        "source_authority_sha256": STATIC_ROOTS["source_authority"][1],
        "source_authority_review_sha256": STATIC_ROOTS[
            "source_authority_review"
        ][1],
        "generation_claim_sha256": STATIC_ROOTS["generation_claim"][1],
        "generation_execution_anchor_sha256": STATIC_ROOTS["generation_anchor"][1],
        "generation_final_release_sha256": STATIC_ROOTS["generation_release"][1],
        "generation_single_use_capability_sha256": STATIC_ROOTS[
            "consumed_generation_capability"
        ][1],
        "pre_generation_v4_terminal_successor_record_sha256": STATIC_ROOTS[
            "v4_terminal_successor_record"
        ][1],
        "v4_terminal_failure_lineage_sha256": STATIC_ROOTS[
            "v4_terminal_failure_lineage"
        ][1],
        "independent_incident_root_cause_sha256": STATIC_ROOTS[
            "independent_incident_root_cause"
        ][1],
        "flopscope_reset_verification_sha256": STATIC_ROOTS[
            "flopscope_reset_verification"
        ][1],
        "v4_exclusive_claim_sha256": (
            "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"
        ),
        "v4_terminal_failure_sha256": (
            "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"
        ),
        "v4_consumed_capability_sha256": (
            "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"
        ),
        "manifest_compatibility_scan_receipt_sha256": STATIC_ROOTS[
            "manifest_compatibility_scan_receipt"
        ][1],
        "generation_launch_command_sha256": STATIC_ROOTS[
            "generation_launch_command"
        ][1],
    }
    if any(chain[key] != expected for key, expected in expected_static_chain.items()):
        raise HandoffError("handoff verification static generation chain changed")
    if value["receipt_compatibility"] != {
        "receipt_schema_exact_v4": True,
        "receipt_top_level_keyset_exact_v4": True,
        "runtime_profile_exact_lean_v5": True,
        "nested_reset_field_absent": True,
        "post_commit_reset_law_authenticated": True,
        "flopscope_reset_verification_sha256": STATIC_ROOTS[
            "flopscope_reset_verification"
        ][1],
        "reset_execution_independently_observed": False,
        "authorized_interpreter_version": list(AUTHORIZED_INTERPRETER_VERSION),
        "authorized_interpreter_sha256": AUTHORIZED_INTERPRETER_SHA256,
        "oof_v4r1_science_identity_uses_exact_source_bytes": True,
        "interpreter_dependent_ast_dump_used_as_authority": False,
    }:
        raise HandoffError("handoff verification reset compatibility changed")
    evidence = value["bundle_evidence"]
    evidence_keys = {
        "file_sha256",
        "complete_bank_bytes_read",
        "A15_bytes_read",
        "layer_hashes_recomputed",
        "row_hashes_recomputed",
        "finite_float32_values_checked",
        "vectorized_sector_prefix_checks",
        "logical_unit_sector_prefix_comparisons",
        "all_float32_finite",
        "all_A15_bytes_zero",
        "all_row_hashes_match_receipt",
        "all_layer_hashes_match_receipt",
        "all_sector_prefixes_byte_identical",
        "maximum_bank_layer_payload_bytes_buffered",
        "bank_hash_block_bytes",
        "maximum_canonical_json_bytes",
        "resource_bound",
    }
    if set(evidence) != evidence_keys or set(evidence["file_sha256"]) != EXPECTED_INVENTORY:
        raise HandoffError("handoff verification bundle-evidence inventory changed")
    if any(
        type(item) is not str
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in evidence["file_sha256"].values()
    ):
        raise HandoffError("handoff verification file hash is invalid")
    if (
        evidence["file_sha256"][RECEIPT_NAME]
        != chain["generation_receipt_sha256"]
        or evidence["complete_bank_bytes_read"]
        != {str(h): EXPECTED_FILES[h][1] for h in EXPECTED_H}
        or evidence["A15_bytes_read"]
        != {str(h): EXPECTED_FILES[h][3] for h in EXPECTED_H}
        or evidence["layer_hashes_recomputed"] != ROWS * LAYERS * len(EXPECTED_H)
        or evidence["row_hashes_recomputed"] != ROWS * len(EXPECTED_H)
        or evidence["finite_float32_values_checked"]
        != ROWS * LAYERS * WIDTH * sum(EXPECTED_H)
        or evidence["vectorized_sector_prefix_checks"] != ROWS * LAYERS * 3
        or evidence["logical_unit_sector_prefix_comparisons"]
        != ROWS * LAYERS * WIDTH * SECTORS * 3
        or evidence["maximum_bank_layer_payload_bytes_buffered"]
        != WIDTH * sum(EXPECTED_H) * 4
        or evidence["bank_hash_block_bytes"] != 8 << 20
        or evidence["maximum_canonical_json_bytes"]
        != MAX_CANONICAL_JSON_BYTES
        or evidence["resource_bound"]
        != (
            "BANK_PAYLOAD_STREAMING__ONE_THREE_H_LAYER__"
            "CANONICAL_JSON_CAPPED_128MIB"
        )
        or any(
            evidence[key] is not True
            for key in (
                "all_float32_finite",
                "all_A15_bytes_zero",
                "all_row_hashes_match_receipt",
                "all_layer_hashes_match_receipt",
                "all_sector_prefixes_byte_identical",
            )
        )
    ):
        raise HandoffError("handoff verification complete-byte evidence changed")
    if value["sentinel_evidence"] != {
        "sentinel_ranks": [0, 343, 687],
        "sentinel_receipt_claims_authenticated": True,
        "sentinel_bank_digest_crosswalk_verified": True,
        "non_sentinel_rows_with_null_dual_parity_evidence": 685,
        "all_row_dual_parity_claimed": False,
        "sentinel_comparators_reexecuted_by_handoff_verifier": False,
    }:
        raise HandoffError("handoff verification sentinel evidence changed")
    expected_meter_names = sorted(
        {
            "complete_flops_by_rank",
            "complete_calls_by_rank",
            "base_parent_flops_by_rank",
            "base_parent_calls_by_rank",
            "capture_parent_flops_by_rank",
            "capture_parent_calls_by_rank",
            "literal_source_hook_flops_by_rank",
            "literal_source_hook_calls_by_rank",
            "reference_source_hook_flops_by_rank",
            "reference_source_hook_calls_by_rank",
            "literal_workspace_flops_by_rank",
            "literal_workspace_calls_by_rank",
            "reference_workspace_flops_by_rank",
            "reference_workspace_calls_by_rank",
            "peak_rss_bytes_by_rank",
            "wall_seconds_by_rank",
            "residual_seconds_by_rank",
            "reference_source_hook_capture_count_by_rank",
        }
    )
    if value["meter_evidence"] != {
        "meter_vector_names": expected_meter_names,
        "meter_vector_count": len(expected_meter_names),
        "meter_vector_length": ROWS,
        "meter_vector_row_scalar_comparisons": len(expected_meter_names) * ROWS,
        "finite_nonnegative_float_meter_values": 2 * ROWS,
        "nonnegative_integer_meter_values": (len(expected_meter_names) - 2) * ROWS,
        "all_meter_vectors_align_with_row_scalars": True,
    }:
        raise HandoffError("handoff verification meter-vector evidence changed")
    if value["access_accounting"] != {
        "bundle_files_opened": 7,
        "staging_entries_opened": 0,
        "target_artifacts_opened": 0,
        "target_values_decoded": 0,
        "weight_arrays_opened": 0,
        "row_manifests_opened": 0,
        "scores_opened": 0,
        "training_runs": 0,
        "estimator_runs": 0,
        "submissions": 0,
    }:
        raise HandoffError("handoff verification access accounting changed")
    if value["limitations"] != {
        "runtime_target_freedom": "AUTHENTICATED_RECEIPT_AND_FROZEN_CONTROL_FLOW_NOT_EXTERNAL_PROCESS_TRACING",
        "sentinel_parity": "AUTHENTICATED_RECEIPT_DIGEST_CROSSWALK_NOT_COMPARATOR_REEXECUTION",
        "hostile_interpreter_security_claim": False,
    }:
        raise HandoffError("handoff verification limitations changed")
    return value


def _canonical_sealer_evidence(runner: Any, sealer: Any) -> dict[str, Any]:
    if transaction_status()["status"] != "READY_ATOMIC_BUNDLE_PUBLISHED":
        raise HandoffError("atomic V5 bundle is not ready for canonical sealing")
    arguments = argparse.Namespace(
        source_authority=SOURCE_AUTHORITY,
        source_authority_review=SOURCE_AUTHORITY_REVIEW,
        generation_anchor=GENERATION_ANCHOR,
        generation_release=GENERATION_RELEASE,
        consumed_generation_capability=CONSUMED_CAPABILITY,
        generation_claim=CLAIM,
        generation_bundle=BUNDLE,
    )
    _, evidence = sealer._validated_inputs(arguments)
    receipt_raw, receipt = _read_canonical(
        BUNDLE / RECEIPT_NAME, "V5 receipt for issuance replay"
    )
    checked_receipt, chain, compatibility = _validate_receipt_and_chain(
        runner, sealer, receipt, receipt_raw
    )
    if (
        evidence["receipt_sha256"] != chain["generation_receipt_sha256"]
        or evidence["claim_sha256"] != chain["generation_claim_sha256"]
    ):
        raise HandoffError("canonical sealer and V5 receipt-chain replay differ")
    replay = dict(evidence)
    replay["handoff_meter_evidence"] = verify_meter_vector_alignment(
        checked_receipt
    )
    replay["handoff_receipt_compatibility"] = compatibility
    return replay


def _crosscheck_canonical_sealer_evidence(
    verification: dict[str, Any], evidence: dict[str, Any]
) -> None:
    _validate_verification(verification)
    bundle_evidence = verification["bundle_evidence"]
    chain = verification["generation_chain"]
    if (
        evidence["receipt_sha256"] != chain["generation_receipt_sha256"]
        or evidence["claim_sha256"] != chain["generation_claim_sha256"]
        or evidence["source_authority_sha256"]
        != chain["source_authority_sha256"]
        or evidence["source_authority_review_sha256"]
        != chain["source_authority_review_sha256"]
        or evidence["bank_evidence"]["complete_bank_sha256"]
        != {
            str(h): bundle_evidence["file_sha256"][EXPECTED_FILES[h][0]]
            for h in EXPECTED_H
        }
        or evidence["bank_evidence"]["A15_zero_sha256"]
        != {
            str(h): bundle_evidence["file_sha256"][EXPECTED_FILES[h][2]]
            for h in EXPECTED_H
        }
        or evidence["bank_evidence"]["sector_prefix_comparisons"]
        != bundle_evidence["logical_unit_sector_prefix_comparisons"]
        or evidence["handoff_meter_evidence"] != verification["meter_evidence"]
        or evidence["handoff_receipt_compatibility"]
        != verification["receipt_compatibility"]
    ):
        raise HandoffError(
            "canonical sealer receipt/bank/prefix/meter/reset replay differs from full verification"
        )


def build_review_request(
    verification_raw: bytes,
    verification: dict[str, Any],
    seal_raw: bytes,
    seal: dict[str, Any],
    runner: Any,
) -> dict[str, Any]:
    _validate_verification(verification)
    verification_sha = hashlib.sha256(verification_raw).hexdigest()
    seal_sha = hashlib.sha256(seal_raw).hexdigest()
    runner.validate_lean_v5_post_generation_seal_object(seal)
    evidence = verification["bundle_evidence"]
    if (
        seal["generation_transaction_bindings"]["generation_receipt_sha256"]
        != evidence["file_sha256"][RECEIPT_NAME]
        or seal["bank_bindings"]["complete_bank_sha256"]
        != {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][0]]
            for h in EXPECTED_H
        }
        or seal["bank_bindings"]["A15_zero_sha256"]
        != {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][2]]
            for h in EXPECTED_H
        }
    ):
        raise HandoffError("canonical seal differs from full handoff verification")
    return {
        "schema": "public-q24-lean-v5-post-generation-independent-review-request-v1",
        "status": "REQUEST_INDEPENDENT_CANONICAL_V5_REVIEW__NO_OOF_HANDOFF_YET",
        "issued_by": "postseal_handoff_verifier",
        "issued_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "handoff_verification_sha256": verification_sha,
        "post_generation_seal_sha256": seal_sha,
        "frozen_oof_v4r1_package_manifest_sha256": STATIC_ROOTS[
            "oof_v4r1_package_manifest"
        ][1],
        "frozen_oof_v4r1_runner_sha256": STATIC_ROOTS["oof_v4r1_runner"][1],
        "frozen_oof_v4r1_post_generation_sealer_sha256": STATIC_ROOTS[
            "oof_v4r1_post_generation_sealer"
        ][1],
        "required_review_schema": (
            "public-q24-signed-pair-lean-three-bank-post-generation-seal-"
            "independent-review-v5"
        ),
        "required_review_status": (
            "GO_AUTHENTICATED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANKS"
        ),
        "review_must_repeat_receipt_and_six_bank_rehash": True,
        "review_must_repeat_sector_prefix_replay": True,
        "review_must_repeat_meter_vector_alignment": True,
        "review_must_authenticate_reset_compatibility": True,
        "reviewer_must_differ_from_seal_issuer": True,
        "review_output_must_be_absent": True,
        "targets_may_be_opened": False,
        "training_or_scoring_may_run": False,
        "oof_consumer_handoff_allowed_before_GO_review": False,
    }


def _validate_review_request(
    value: Any, *, verification_sha256: str, seal_sha256: str
) -> dict[str, Any]:
    exact_keys = {
        "schema",
        "status",
        "issued_by",
        "issued_utc",
        "handoff_verification_sha256",
        "post_generation_seal_sha256",
        "frozen_oof_v4r1_package_manifest_sha256",
        "frozen_oof_v4r1_runner_sha256",
        "frozen_oof_v4r1_post_generation_sealer_sha256",
        "required_review_schema",
        "required_review_status",
        "review_must_repeat_receipt_and_six_bank_rehash",
        "review_must_repeat_sector_prefix_replay",
        "review_must_repeat_meter_vector_alignment",
        "review_must_authenticate_reset_compatibility",
        "reviewer_must_differ_from_seal_issuer",
        "review_output_must_be_absent",
        "targets_may_be_opened",
        "training_or_scoring_may_run",
        "oof_consumer_handoff_allowed_before_GO_review",
    }
    if type(value) is not dict or set(value) != exact_keys:
        raise HandoffError("canonical V5 review-request field inventory changed")
    try:
        parsed_utc = datetime.strptime(
            value["issued_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as error:
        raise HandoffError("canonical V5 review-request time changed") from error
    expected = {
        "schema": "public-q24-lean-v5-post-generation-independent-review-request-v1",
        "status": "REQUEST_INDEPENDENT_CANONICAL_V5_REVIEW__NO_OOF_HANDOFF_YET",
        "issued_by": "postseal_handoff_verifier",
        "handoff_verification_sha256": verification_sha256,
        "post_generation_seal_sha256": seal_sha256,
        "frozen_oof_v4r1_package_manifest_sha256": STATIC_ROOTS[
            "oof_v4r1_package_manifest"
        ][1],
        "frozen_oof_v4r1_runner_sha256": STATIC_ROOTS["oof_v4r1_runner"][1],
        "frozen_oof_v4r1_post_generation_sealer_sha256": STATIC_ROOTS[
            "oof_v4r1_post_generation_sealer"
        ][1],
        "required_review_schema": (
            "public-q24-signed-pair-lean-three-bank-post-generation-seal-"
            "independent-review-v5"
        ),
        "required_review_status": (
            "GO_AUTHENTICATED_LEAN_RUNTIME_V5_TARGET_BLIND_THREE_BANKS"
        ),
        "review_must_repeat_receipt_and_six_bank_rehash": True,
        "review_must_repeat_sector_prefix_replay": True,
        "review_must_repeat_meter_vector_alignment": True,
        "review_must_authenticate_reset_compatibility": True,
        "reviewer_must_differ_from_seal_issuer": True,
        "review_output_must_be_absent": True,
        "targets_may_be_opened": False,
        "training_or_scoring_may_run": False,
        "oof_consumer_handoff_allowed_before_GO_review": False,
    }
    if (
        parsed_utc.strftime("%Y-%m-%dT%H:%M:%SZ") != value["issued_utc"]
        or any(value[key] != expected_value for key, expected_value in expected.items())
    ):
        raise HandoffError("canonical V5 review request changed")
    return value


def build_consumer_handoff(
    verification_raw: bytes,
    verification: dict[str, Any],
    seal_raw: bytes,
    seal: dict[str, Any],
    review_raw: bytes,
    review: dict[str, Any],
    runner: Any,
    *,
    seal_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    _validate_verification(verification)
    verification_sha = hashlib.sha256(verification_raw).hexdigest()
    seal_sha = hashlib.sha256(seal_raw).hexdigest()
    review_sha = hashlib.sha256(review_raw).hexdigest()
    checked_seal = runner.validate_lean_v5_post_generation_seal_object(seal)
    runner.validate_lean_v5_post_generation_seal_review_object(
        review, seal_sha256=seal_sha, seal=checked_seal
    )
    evidence = verification["bundle_evidence"]
    chain = verification["generation_chain"]
    if (
        review["post_generation_seal_sha256"] != seal_sha
        or seal["generation_transaction_bindings"]["generation_receipt_sha256"]
        != evidence["file_sha256"][RECEIPT_NAME]
        or seal["bank_bindings"]["complete_bank_sha256"]
        != {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][0]]
            for h in EXPECTED_H
        }
        or seal["bank_bindings"]["A15_zero_sha256"]
        != {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][2]]
            for h in EXPECTED_H
        }
    ):
        raise HandoffError("reviewed canonical seal differs from full verification")
    bank_bindings = {
        "source_authority_sha256": chain["source_authority_sha256"],
        "source_authority_review_sha256": chain[
            "source_authority_review_sha256"
        ],
        "pre_generation_v4_terminal_successor_record_sha256": chain[
            "pre_generation_v4_terminal_successor_record_sha256"
        ],
        "v4_terminal_failure_lineage_sha256": chain[
            "v4_terminal_failure_lineage_sha256"
        ],
        "independent_incident_root_cause_sha256": chain[
            "independent_incident_root_cause_sha256"
        ],
        "flopscope_reset_verification_sha256": chain[
            "flopscope_reset_verification_sha256"
        ],
        "manifest_compatibility_scan_receipt_sha256": chain[
            "manifest_compatibility_scan_receipt_sha256"
        ],
        "v4_exclusive_claim_sha256": chain["v4_exclusive_claim_sha256"],
        "v4_terminal_failure_sha256": chain["v4_terminal_failure_sha256"],
        "v4_consumed_capability_sha256": chain[
            "v4_consumed_capability_sha256"
        ],
        "lean_runtime_v5_package_manifest_sha256": STATIC_ROOTS[
            "runtime_v5_package_manifest"
        ][1],
        "lean_runtime_v5_independent_review_sha256": STATIC_ROOTS[
            "runtime_v5_independent_review"
        ][1],
        "generation_execution_anchor_sha256": chain[
            "generation_execution_anchor_sha256"
        ],
        "generation_final_release_sha256": chain[
            "generation_final_release_sha256"
        ],
        "generation_single_use_capability_sha256": chain[
            "generation_single_use_capability_sha256"
        ],
        "generation_claim_sha256": chain["generation_claim_sha256"],
        "generation_receipt_sha256": chain["generation_receipt_sha256"],
        "post_generation_seal_sha256": seal_sha,
        "post_generation_seal_review_sha256": review_sha,
        "ordered_688_record_sha256": (
            "902466f86013e2bf7e597518d124e4d98a5654fa63949ca8068522383d95bab3"
        ),
        "complete_bank_sha256": {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][0]]
            for h in EXPECTED_H
        },
        "A15_zero_sha256": {
            str(h): evidence["file_sha256"][EXPECTED_FILES[h][2]]
            for h in EXPECTED_H
        },
    }
    if set(bank_bindings) != OOF_BANK_BINDING_KEYS:
        raise HandoffError("OOF V4 bank-binding inventory changed")
    oof_static_package_bindings = {
        "package_manifest_sha256": STATIC_ROOTS["oof_v4r1_package_manifest"][1],
        "freeze_receipt_sha256": STATIC_ROOTS["oof_v4r1_freeze_receipt"][1],
        "freeze_verification_sha256": STATIC_ROOTS[
            "oof_v4r1_freeze_verification"
        ][1],
        "static_verification_sha256": STATIC_ROOTS[
            "oof_v4r1_static_verification"
        ][1],
        "runner_sha256": STATIC_ROOTS["oof_v4r1_runner"][1],
        "post_generation_sealer_sha256": STATIC_ROOTS[
            "oof_v4r1_post_generation_sealer"
        ][1],
        "post_generation_seal_contract_sha256": STATIC_ROOTS[
            "oof_v4r1_post_generation_seal_contract"
        ][1],
        "v5_provenance_fixtures_sha256": STATIC_ROOTS[
            "oof_v4r1_v5_provenance_fixtures"
        ][1],
        "v4r1_authority_bindings_sha256": STATIC_ROOTS[
            "oof_v4r1_authority_bindings"
        ][1],
        "static_verifier_sha256": STATIC_ROOTS[
            "oof_v4r1_static_verifier"
        ][1],
        "independent_static_review_sha256": STATIC_ROOTS[
            "oof_v4r1_independent_static_review"
        ][1],
    }
    if set(oof_static_package_bindings) != OOF_STATIC_PACKAGE_BINDING_KEYS:
        raise HandoffError("OOF V4R1 static-package binding inventory changed")
    return {
        "schema": "public-q24-lean-v5-reviewed-bank-oof-v4-consumer-handoff-v1",
        "status": "GO_BIND_EXACT_REVIEWED_V5_BANKS_IN_FRESH_OOF_V4_AUTHORITY",
        "issued_by": "postseal_handoff_verifier",
        "issued_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "handoff_verification_sha256": verification_sha,
        "oof_v4_static_package_bindings": oof_static_package_bindings,
        "generation_control_bindings": {
            "exact_launch_command_sha256": chain[
                "generation_launch_command_sha256"
            ],
            "runtime_v5_runner_sha256": STATIC_ROOTS[
                "runtime_v5_runner"
            ][1],
            "prelaunch_authority_builder_independent_review_sha256": STATIC_ROOTS[
                "prelaunch_authority_builder_independent_review"
            ][1],
            "predecessor_v4_postseal_package_manifest_sha256": STATIC_ROOTS[
                "predecessor_v4_postseal_package_manifest"
            ][1],
            "predecessor_v4_postseal_independent_review_sha256": STATIC_ROOTS[
                "predecessor_v4_postseal_independent_review"
            ][1],
        },
        "bank_bindings": bank_bindings,
        "external_paths": {
            "bank_root": str(BUNDLE),
            "bank_generation_receipt": str(BUNDLE / RECEIPT_NAME),
            "bank_source_authority": str(SOURCE_AUTHORITY),
            "bank_source_authority_review": str(SOURCE_AUTHORITY_REVIEW),
            "bank_post_generation_seal": str(_absolute(seal_path)),
            "bank_post_generation_seal_review": str(_absolute(review_path)),
        },
        "full_bundle_reverification_passed_before_handoff": True,
        "meter_vector_alignment_passed_before_handoff": True,
        "reset_compatible_exact_v4_receipt_passed_before_handoff": True,
        "independent_canonical_post_generation_review_passed": True,
        "target_access_allowed": False,
        "training_or_scoring_allowed_by_this_handoff": False,
        "execution_authority_issued_by_this_handoff": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--verify-bundle", action="store_true")
    mode.add_argument("--issue-canonical-seal", action="store_true")
    mode.add_argument("--issue-review-request", action="store_true")
    mode.add_argument("--issue-canonical-review", action="store_true")
    mode.add_argument("--issue-consumer-handoff", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-request", type=Path)
    parser.add_argument("--confirmation")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.preflight:
        print(
            json.dumps(
                {
                    "schema": "public-q24-lean-v5-postseal-handoff-preflight-v1r1",
                    "status": "STATIC_READY__NO_TRANSACTION_OR_BANK_OPEN",
                    "verify_confirmation": VERIFY_CONFIRMATION,
                    "seal_confirmation": SEAL_CONFIRMATION,
                    "request_confirmation": REQUEST_CONFIRMATION,
                    "review_confirmation": REVIEW_CONFIRMATION,
                    "handoff_confirmation": HANDOFF_CONFIRMATION,
                    "transaction_entries_opened": 0,
                    "staging_entries_opened": 0,
                    "bank_files_opened": 0,
                    "target_artifacts_opened": 0,
                },
                sort_keys=True,
            )
        )
        return
    if args.status:
        print(json.dumps(transaction_status(), sort_keys=True))
        return
    if args.output is None:
        raise SystemExit("protected issuance requires --output")
    if args.verify_bundle:
        if args.confirmation != VERIFY_CONFIRMATION:
            raise SystemExit("bundle verification confirmation mismatch")
        document = verify_bundle()
    elif args.issue_canonical_seal:
        if args.confirmation != SEAL_CONFIRMATION or args.verification is None:
            raise SystemExit(
                "canonical seal requires exact confirmation and --verification"
            )
        verification_raw, verification = _read_canonical(
            args.verification, "handoff verification"
        )
        del verification_raw
        verify_static_roots()
        runner = _load_frozen(
            OOF_RUNNER,
            STATIC_ROOTS["oof_v4r1_runner"][1],
            "frozen_oof_v4r1_runner_for_seal",
        )
        sealer = _load_frozen(
            OOF_SEALER,
            STATIC_ROOTS["oof_v4r1_post_generation_sealer"][1],
            "frozen_oof_v4r1_sealer_for_seal",
        )
        evidence = _canonical_sealer_evidence(runner, sealer)
        _crosscheck_canonical_sealer_evidence(verification, evidence)
        document = sealer.build_seal(
            runner,
            evidence,
            seal_id=SEAL_ID,
            issued_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    elif args.issue_canonical_review:
        if (
            args.confirmation != REVIEW_CONFIRMATION
            or args.verification is None
            or args.seal is None
            or args.review_request is None
        ):
            raise SystemExit(
                "canonical review requires exact confirmation, --verification, --seal, and --review-request"
            )
        verification_raw, verification = _read_canonical(
            args.verification, "handoff verification"
        )
        seal_raw, seal = _read_canonical(args.seal, "canonical V5 seal")
        review_request_raw, review_request = _read_canonical(
            args.review_request, "canonical V5 review request"
        )
        del review_request_raw
        _validate_review_request(
            review_request,
            verification_sha256=hashlib.sha256(verification_raw).hexdigest(),
            seal_sha256=hashlib.sha256(seal_raw).hexdigest(),
        )
        del verification_raw
        verify_static_roots()
        runner = _load_frozen(
            OOF_RUNNER,
            STATIC_ROOTS["oof_v4r1_runner"][1],
            "frozen_oof_v4r1_runner_for_review",
        )
        sealer = _load_frozen(
            OOF_SEALER,
            STATIC_ROOTS["oof_v4r1_post_generation_sealer"][1],
            "frozen_oof_v4r1_sealer_for_review",
        )
        evidence = _canonical_sealer_evidence(runner, sealer)
        _crosscheck_canonical_sealer_evidence(verification, evidence)
        document = sealer.build_review(
            runner,
            evidence,
            seal_raw,
            seal,
            review_id=REVIEW_ID,
            issued_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    else:
        required = ["verification", "seal"]
        if args.issue_consumer_handoff:
            required.append("review")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit("missing required argument(s): " + ",".join(missing))
        verification_raw, verification = _read_canonical(
            args.verification, "handoff verification"
        )
        seal_raw, seal = _read_canonical(args.seal, "canonical V5 seal")
        verify_static_roots()
        runner = _load_frozen(
            OOF_RUNNER,
            STATIC_ROOTS["oof_v4r1_runner"][1],
            "frozen_oof_v4r1_runner_for_handoff",
        )
        if args.issue_review_request:
            if args.confirmation != REQUEST_CONFIRMATION:
                raise SystemExit("review-request confirmation mismatch")
            _require_current_atomic_ready("canonical review-request issuance")
            sealer = _load_frozen(
                OOF_SEALER,
                STATIC_ROOTS["oof_v4r1_post_generation_sealer"][1],
                "frozen_oof_v4r1_sealer_for_review_request",
            )
            evidence = _canonical_sealer_evidence(runner, sealer)
            _crosscheck_canonical_sealer_evidence(verification, evidence)
            document = build_review_request(
                verification_raw, verification, seal_raw, seal, runner
            )
            _require_current_atomic_ready(
                "canonical review-request output after sealer replay"
            )
        else:
            if args.confirmation != HANDOFF_CONFIRMATION:
                raise SystemExit("consumer-handoff confirmation mismatch")
            _require_current_atomic_ready("OOF consumer-handoff issuance")
            sealer = _load_frozen(
                OOF_SEALER,
                STATIC_ROOTS["oof_v4r1_post_generation_sealer"][1],
                "frozen_oof_v4r1_sealer_for_consumer_handoff",
            )
            evidence = _canonical_sealer_evidence(runner, sealer)
            _crosscheck_canonical_sealer_evidence(verification, evidence)
            review_raw, review = _read_canonical(
                args.review, "independent canonical V5 seal review"
            )
            document = build_consumer_handoff(
                verification_raw,
                verification,
                seal_raw,
                seal,
                review_raw,
                review,
                runner,
                seal_path=args.seal,
                review_path=args.review,
            )
            _require_current_atomic_ready(
                "OOF consumer-handoff output after sealer replay"
            )
    digest = _write_exclusive(args.output, document)
    print(
        json.dumps(
            {"status": document["status"], "output": str(args.output), "sha256": digest},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
