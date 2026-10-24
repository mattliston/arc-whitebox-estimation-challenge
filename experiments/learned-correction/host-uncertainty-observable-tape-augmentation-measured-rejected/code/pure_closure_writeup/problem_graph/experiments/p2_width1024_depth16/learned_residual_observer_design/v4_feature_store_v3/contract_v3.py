#!/usr/bin/env python3
"""Additive V3 bindings over the immutable V1/V2 feature-store contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "v4_feature_store"
V2 = HERE.parent / "v4_feature_store_v2"
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
V4 = HERE.parent / "v4_launch"
SOURCE_MANIFEST_PATH = HERE / "SOURCE_MANIFEST_V3.json"
RELEASE_SCHEMA_PATH = HERE / "RELEASE_SCHEMA_V3.json"
CLAIM_PATH = HERE / "PHASE2_OBSERVER_V4_FEATURE_STORE_V3_ONCE.CLAIM"
STAGING_PATH = HERE / "sealed_target_blind_feature_store_v3.staging"
OUTPUT_PATH = HERE / "sealed_target_blind_feature_store_v3"
SCRATCH_PATH = HERE / ".feature_store_v3_row_scratch"


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


V1_BINDINGS = {
    "independent_source_audit_hold": {
        "path": V1 / "INDEPENDENT_SOURCE_AUDIT.json",
        "sha256": "84deb6eaa93aeba4b8cadf609b6f30ced36b890cc1c52702c1fb6a94da21be42",
    },
    "source_manifest": {
        "path": V1 / "SOURCE_MANIFEST.json",
        "sha256": "0227139aa62f9d6afbbcfcf1d64b6c1e1751c77129aabd086a15d1863b7454db",
    },
    "static_receipt": {
        "path": V1 / "STATIC_PREFLIGHT_RECEIPT.json",
        "sha256": "dc3fe5d788de54939d81019dd9c1e5537901e06f8a5d5267030a26ec16e69bed",
    },
    "contract": {
        "path": V1 / "contract.py",
        "sha256": "9744e49d3e17d92a57debe939f67a3f677f03d93250eca8308eb9a219303083c",
    },
    "generator_with_bound_resource_defects": {
        "path": V1 / "generate_feature_store_once.py",
        "sha256": "9a40003e8de559bdf27842c42651dea3fcf4782f29019a9b2b6e788e610a619c",
    },
    "host_worker": {
        "path": V1 / "host_feature_worker.py",
        "sha256": "12333cd06b8870dc05783e95c45dbea073559139be210dba60108a6d34e872ce",
    },
    "provider": {
        "path": V1 / "sealed_feature_store_provider.py",
        "sha256": "a1e9c2144153003a167e7aa00ed449b501308a6eb82490db6a45fe28428734d7",
    },
    "artifact_verifier": {
        "path": V1 / "verify_sealed_feature_store.py",
        "sha256": "52a7524250cae522c051314644916f4a473c49a035b75125d69dcbd81d71a2b1",
    },
}

V2_BINDINGS = {
    "independent_source_audit_hold": {
        "path": V2 / "INDEPENDENT_SOURCE_AUDIT_V2.json",
        "sha256": "bad66c45d982eddd1599d2c9c20881c3c8b391a5e168cf8634e435600664fb20",
    },
    "source_manifest": {
        "path": V2 / "SOURCE_MANIFEST_V2.json",
        "sha256": "afac525276e7afb2c707c318ed25a37cdcbdbbd42a7612c8918d362981d08a9f",
    },
    "static_receipt": {
        "path": V2 / "STATIC_PREFLIGHT_RECEIPT_V2.json",
        "sha256": "0d0b477f505127036658a6536f4149a55089f4d17696078c3c90140ed6584fb4",
    },
    "injected_receipt": {
        "path": V2 / "INJECTED_RESOURCE_TEST_RECEIPT_V2.json",
        "sha256": "66926f8503ea98013885d81f0fccac2941c5d40efacc54a90e9e3e8b23001312",
    },
    "protocol": {
        "path": V2 / "PROTOCOL_V2.md",
        "sha256": "782584ee64123bd7d13a705bfcaa8ff01ec5601ec950da015794d828079c411e",
    },
    "release_schema": {
        "path": V2 / "RELEASE_SCHEMA_V2.json",
        "sha256": "be14baa0cb4e274778ee36619b61517d6bb3c94329bde4c006ee19472fdfb25b",
    },
    "static_contract": {
        "path": V2 / "STATIC_CONTRACT_V2.json",
        "sha256": "688c79a121ad953c97173131f31c4ab7744dea57e9c558a3525be8e36a83e119",
    },
    "contract": {
        "path": V2 / "contract_v2.py",
        "sha256": "705b3e7c7779dd550c196798c547858e72caf9a556ee85874c555c90099faa43",
    },
    "resource_safety_with_bound_launch_race": {
        "path": V2 / "resource_safety_v2.py",
        "sha256": "3970e755a0b7197ed341767deffb1da5b22dfd9dea473d1468d89fb4f5297785",
    },
    "generator_with_bound_launch_race": {
        "path": V2 / "generate_feature_store_once_v2.py",
        "sha256": "5119558ed5014d2d74f850386ba8698ae46b555ada4fedca0a10407acb5b51ab",
    },
    "cursor": {
        "path": V2 / "target_blind_cursor_v2.py",
        "sha256": "a6cd8e0e16397ce95c939ac1fb9f7bcf4fe74f2056c5d77b592e5526e3a694a3",
    },
    "host_worker": {
        "path": V2 / "host_feature_worker_v2.py",
        "sha256": "02e22d94e3b3f59c35411b6ca0c6c7fcca1bc7e0d8ac835fb229d72f097daf17",
    },
    "host_exec_gate": {
        "path": V2 / "host_worker_exec_gate_v2.py",
        "sha256": "9c88c954c1cfc78bd58824ae505ee70aac3514fe9d537367a7c8c6e0f81845ed",
    },
    "provider": {
        "path": V2 / "sealed_feature_store_provider_v2.py",
        "sha256": "62bb048be2640b355cc5773129837303218cae3b030deff7be109b7f59365c33",
    },
    "artifact_verifier": {
        "path": V2 / "verify_artifacts_v2.py",
        "sha256": "40a8ad36bd0ed346b97ebf7a7caf7d5dbda954e896821f6a83c758e207980e54",
    },
    "static_verifier": {
        "path": V2 / "verify_static_preflight_v2.py",
        "sha256": "a32676caf4466c4e7a8c8f2fd51d349abd44da1bddbb93419f9820a6abefb08a",
    },
    "synthetic_group_tree": {
        "path": V2 / "synthetic_group_tree_v2.py",
        "sha256": "be45cf1bf2db2fcc557663566e074c5a8270b200613c29e531a90e7953bcedcd",
    },
    "injected_test": {
        "path": V2 / "test_injected_resource_failures_v2.py",
        "sha256": "31f1f4938e52d0f7b0a6c3afb2d7e15cf32337234291afc10b5a1aedf08fb602",
    },
}


def _load_v1_contract() -> Any:
    source = V1_BINDINGS["contract"]["path"]
    if sha256_file(source) != V1_BINDINGS["contract"]["sha256"]:
        raise AssertionError("immutable V1 contract changed")
    spec = importlib.util.spec_from_file_location("phase2_v4_feature_store_v1_contract", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable V1 contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1_CONTRACT = _load_v1_contract()
CAPS = V1_CONTRACT.CAPS
DEPTH = V1_CONTRACT.DEPTH
WIDTH = V1_CONTRACT.WIDTH
LOCAL_FEATURES = V1_CONTRACT.LOCAL_FEATURES
ROLE_ORDER = V1_CONTRACT.ROLE_ORDER
ROLE_BINDINGS = V1_CONTRACT.ROLE_BINDINGS
FROZEN_BINDINGS = V1_CONTRACT.FROZEN_BINDINGS
EXPECTED_ENVIRONMENT = V1_CONTRACT.EXPECTED_ENVIRONMENT
ENVIRONMENT_PYTHON = V1_CONTRACT.ENVIRONMENT_PYTHON
PER_PROCESS_LIMIT_BYTES = V1_CONTRACT.PER_PROCESS_LIMIT_BYTES
MIN_FREE_BYTES = V1_CONTRACT.MIN_FREE_BYTES
ROW_SCHEMA = V1_CONTRACT.ROW_SCHEMA
FINAL_SCHEMA = V1_CONTRACT.FINAL_SCHEMA
HEX64 = V1_CONTRACT.HEX64
MAX_HOST_WORKERS = 2
MAX_INVOCATIONS = 8
WATCHDOG_INTERVAL_SECONDS = 0.05
MAX_WATCHDOG_GAP_SECONDS = 0.20
TERMINATION_GRACE_SECONDS = 5.0
QUIESCENCE_TIMEOUT_SECONDS = 5.0

canonical_json_bytes = V1_CONTRACT.canonical_json_bytes
fsync_directory = V1_CONTRACT.fsync_directory
write_exclusive_fsynced = V1_CONTRACT.write_exclusive_fsynced
load_exact_roster = V1_CONTRACT.load_exact_roster
raw_array_sha256 = V1_CONTRACT.raw_array_sha256
array_payload_sha256 = V1_CONTRACT.array_payload_sha256
row_relative_path = V1_CONTRACT.row_relative_path
ordered_ids_sha256 = V1_CONTRACT.ordered_ids_sha256
expected_row = V1_CONTRACT.expected_row
verify_frozen_bindings = V1_CONTRACT.verify_frozen_bindings


def validate_row_directory(row_dir: Path, *, role: str, role_rank: int) -> dict[str, Any]:
    """V1 row contract with an exact additive V3 worker-source binding."""

    expected = V1_CONTRACT.expected_row(role, role_rank)
    manifest_path = row_dir / "row_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROW_SCHEMA or manifest.get("status") != "SEALED_ROW_LOCAL_FEATURES":
        raise AssertionError("V3 row manifest schema/status changed")
    scalar = {
        "role": role, "role_rank": role_rank, "mlp_id": int(expected["mlp_id"]),
        "mlp_seed": int(expected["mlp_seed"]), "source_ordinal": int(expected["ordinal"]),
        "source_shard_index": int(expected["shard_index"]),
        "source_row_in_shard": int(expected["row_in_shard"]),
        "weights_raw_sha256": str(expected["weights_sha256"]), "target_values_decoded": 0,
    }
    for key, value in scalar.items():
        if manifest.get(key) != value:
            raise AssertionError(f"V3 row binding changed: {key}")
    if manifest.get("decoded_columns") != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
        raise AssertionError("V3 row source projection changed")
    if manifest.get("host_caps") != list(CAPS) or manifest.get("host_latch") != [True] * DEPTH:
        raise AssertionError("V3 row host cap/latch changed")
    files = manifest.get("files")
    expected_names = {"weights.npy"}
    for cap in CAPS:
        expected_names.update(
            {f"cp{cap}_raw_locals.npy", f"cp{cap}_host_rows.npy", f"cp{cap}_sigma.npy"}
        )
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("V3 row file membership changed")
    if set(path.name for path in row_dir.iterdir()) != expected_names | {"row_manifest.json"}:
        raise AssertionError("V3 row directory has unbound members")
    if manifest.get("worker_source_sha256") != sha256_file(HERE / "host_feature_worker_v3.py"):
        raise AssertionError("V3 row was produced by another worker source")
    if manifest.get("frozen_host_worker_sha256") != FROZEN_BINDINGS["v4_host_worker"]["sha256"]:
        raise AssertionError("V3 row binds another frozen host worker")
    if manifest.get("frozen_host_base_sha256") != FROZEN_BINDINGS["v3_host_base"]["sha256"]:
        raise AssertionError("V3 row binds another frozen host base")
    if manifest.get("worker_environment") != EXPECTED_ENVIRONMENT:
        raise AssertionError("V3 row environment changed")
    for counter in (
        "target_columns_projected", "target_values_decoded", "optimizers_constructed", "fits", "scores"
    ):
        if manifest.get(counter) != 0:
            raise AssertionError(f"V3 row counter is nonzero: {counter}")
    if not isinstance(manifest.get("worker_max_rss_bytes"), int) or not (
        0 < manifest["worker_max_rss_bytes"] < PER_PROCESS_LIMIT_BYTES
    ):
        raise AssertionError("V3 row worker RSS violates guard")
    if not isinstance(manifest.get("rlimit_as_installed"), bool):
        raise AssertionError("V3 row lacks the exact-environment RLIMIT capability receipt")
    if manifest.get("strict_group_rss_watchdog_required") is not True:
        raise AssertionError("V3 row does not bind the authoritative group RSS watchdog")

    arrays: dict[str, Any] = {}
    arrays["weights.npy"] = V1_CONTRACT._load_array(
        row_dir / "weights.npy", shape=(DEPTH, WIDTH, WIDTH)
    )
    if raw_array_sha256(arrays["weights.npy"]) != expected["weights_sha256"]:
        raise AssertionError("V3 stored weights differ from roster")
    for cap in CAPS:
        arrays[f"cp{cap}_raw_locals.npy"] = V1_CONTRACT._load_array(
            row_dir / f"cp{cap}_raw_locals.npy", shape=(DEPTH, WIDTH, LOCAL_FEATURES)
        )
        arrays[f"cp{cap}_host_rows.npy"] = V1_CONTRACT._load_array(
            row_dir / f"cp{cap}_host_rows.npy", shape=(DEPTH, WIDTH), nonnegative=True
        )
        arrays[f"cp{cap}_sigma.npy"] = V1_CONTRACT._load_array(
            row_dir / f"cp{cap}_sigma.npy", shape=(DEPTH, WIDTH), positive=True
        )
    for name, array in arrays.items():
        record = files[name]
        if record.get("sha256") != sha256_file(row_dir / name):
            raise AssertionError(f"V3 row file hash changed: {name}")
        if record.get("array_sha256") != array_payload_sha256(array):
            raise AssertionError(f"V3 row array hash changed: {name}")
        if record.get("bytes") != (row_dir / name).stat().st_size:
            raise AssertionError(f"V3 row file size changed: {name}")
    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {str(cap) for cap in CAPS}:
        raise AssertionError("V3 row host reports changed")
    for cap in CAPS:
        host = hosts[str(cap)]
        hashes = host.get("entering_cp_state_hashes")
        if not isinstance(hashes, list) or len(hashes) != DEPTH or not all(
            isinstance(value, str) and V1_CONTRACT.HEX64.fullmatch(value) for value in hashes
        ):
            raise AssertionError("V3 entering CP hash chain changed")
        chain = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
        if host.get("entering_cp_chain_sha256") != chain:
            raise AssertionError("V3 entering CP chain digest changed")
        for field in ("raw32_sha256", "edges_sha256", "host_sha256", "sigma_sha256"):
            if not V1_CONTRACT.HEX64.fullmatch(str(host.get(field, ""))):
                raise AssertionError(f"V3 host hash malformed: {field}")
        if host["raw32_sha256"] != files[f"cp{cap}_raw_locals.npy"]["array_sha256"]:
            raise AssertionError("V3 raw-local host hash mismatch")
        if host["host_sha256"] != files[f"cp{cap}_host_rows.npy"]["array_sha256"]:
            raise AssertionError("V3 host-row hash mismatch")
        if host["sigma_sha256"] != files[f"cp{cap}_sigma.npy"]["array_sha256"]:
            raise AssertionError("V3 sigma hash mismatch")
        if not isinstance(host.get("flops_used"), int) or host["flops_used"] <= 0:
            raise AssertionError("V3 host FLOP receipt absent")
        if not math.isfinite(float(host.get("wall_seconds", math.nan))) or host["wall_seconds"] <= 0:
            raise AssertionError("V3 host timing receipt absent")
    if hosts["1536"]["edges_sha256"] != hosts["1792"]["edges_sha256"]:
        raise AssertionError("V3 cap-independent edge hash differs")
    return manifest


def verify_v1_bindings() -> dict[str, str]:
    observed = {}
    for name, record in V1_BINDINGS.items():
        digest = sha256_file(Path(record["path"]))
        if digest != record["sha256"]:
            raise AssertionError(f"immutable V1 byte changed: {name}")
        observed[name] = digest
    return observed


def verify_v2_bindings() -> dict[str, str]:
    observed = {}
    for name, record in V2_BINDINGS.items():
        digest = sha256_file(Path(record["path"]))
        if digest != record["sha256"]:
            raise AssertionError(f"immutable V2 byte changed: {name}")
        observed[name] = digest
    audit = json.loads(
        V2_BINDINGS["independent_source_audit_hold"]["path"].read_text(encoding="utf-8")
    )
    if audit.get("status") != (
        "HOLD_PHASE2_OBSERVER_V4_FEATURE_STORE_V2_SOURCE_AUDIT__POPEN_REGISTRY_ABORT_RACE"
    ):
        raise AssertionError("immutable V2 independent HOLD status changed")
    return observed


def assert_runtime_absent() -> None:
    for path in (CLAIM_PATH, STAGING_PATH, OUTPUT_PATH, SCRATCH_PATH):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"V3 runtime path exists during source freeze: {path.name}")
