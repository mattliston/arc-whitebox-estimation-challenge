#!/usr/bin/env python3
"""Additive V2 bindings over the immutable V1 feature-store row contract."""

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
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
V4 = HERE.parent / "v4_launch"
SOURCE_MANIFEST_PATH = HERE / "SOURCE_MANIFEST_V2.json"
RELEASE_SCHEMA_PATH = HERE / "RELEASE_SCHEMA_V2.json"
CLAIM_PATH = HERE / "PHASE2_OBSERVER_V4_FEATURE_STORE_V2_ONCE.CLAIM"
STAGING_PATH = HERE / "sealed_target_blind_feature_store_v2.staging"
OUTPUT_PATH = HERE / "sealed_target_blind_feature_store_v2"
SCRATCH_PATH = HERE / ".feature_store_v2_row_scratch"


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
    """V1 row contract with an exact additive V2 worker-source binding."""

    expected = V1_CONTRACT.expected_row(role, role_rank)
    manifest_path = row_dir / "row_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROW_SCHEMA or manifest.get("status") != "SEALED_ROW_LOCAL_FEATURES":
        raise AssertionError("V2 row manifest schema/status changed")
    scalar = {
        "role": role, "role_rank": role_rank, "mlp_id": int(expected["mlp_id"]),
        "mlp_seed": int(expected["mlp_seed"]), "source_ordinal": int(expected["ordinal"]),
        "source_shard_index": int(expected["shard_index"]),
        "source_row_in_shard": int(expected["row_in_shard"]),
        "weights_raw_sha256": str(expected["weights_sha256"]), "target_values_decoded": 0,
    }
    for key, value in scalar.items():
        if manifest.get(key) != value:
            raise AssertionError(f"V2 row binding changed: {key}")
    if manifest.get("decoded_columns") != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
        raise AssertionError("V2 row source projection changed")
    if manifest.get("host_caps") != list(CAPS) or manifest.get("host_latch") != [True] * DEPTH:
        raise AssertionError("V2 row host cap/latch changed")
    files = manifest.get("files")
    expected_names = {"weights.npy"}
    for cap in CAPS:
        expected_names.update(
            {f"cp{cap}_raw_locals.npy", f"cp{cap}_host_rows.npy", f"cp{cap}_sigma.npy"}
        )
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("V2 row file membership changed")
    if set(path.name for path in row_dir.iterdir()) != expected_names | {"row_manifest.json"}:
        raise AssertionError("V2 row directory has unbound members")
    if manifest.get("worker_source_sha256") != sha256_file(HERE / "host_feature_worker_v2.py"):
        raise AssertionError("V2 row was produced by another worker source")
    if manifest.get("frozen_host_worker_sha256") != FROZEN_BINDINGS["v4_host_worker"]["sha256"]:
        raise AssertionError("V2 row binds another frozen host worker")
    if manifest.get("frozen_host_base_sha256") != FROZEN_BINDINGS["v3_host_base"]["sha256"]:
        raise AssertionError("V2 row binds another frozen host base")
    if manifest.get("worker_environment") != EXPECTED_ENVIRONMENT:
        raise AssertionError("V2 row environment changed")
    for counter in (
        "target_columns_projected", "target_values_decoded", "optimizers_constructed", "fits", "scores"
    ):
        if manifest.get(counter) != 0:
            raise AssertionError(f"V2 row counter is nonzero: {counter}")
    if not isinstance(manifest.get("worker_max_rss_bytes"), int) or not (
        0 < manifest["worker_max_rss_bytes"] < PER_PROCESS_LIMIT_BYTES
    ):
        raise AssertionError("V2 row worker RSS violates guard")
    if not isinstance(manifest.get("rlimit_as_installed"), bool):
        raise AssertionError("V2 row lacks the exact-environment RLIMIT capability receipt")
    if manifest.get("strict_group_rss_watchdog_required") is not True:
        raise AssertionError("V2 row does not bind the authoritative group RSS watchdog")

    arrays: dict[str, Any] = {}
    arrays["weights.npy"] = V1_CONTRACT._load_array(
        row_dir / "weights.npy", shape=(DEPTH, WIDTH, WIDTH)
    )
    if raw_array_sha256(arrays["weights.npy"]) != expected["weights_sha256"]:
        raise AssertionError("V2 stored weights differ from roster")
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
            raise AssertionError(f"V2 row file hash changed: {name}")
        if record.get("array_sha256") != array_payload_sha256(array):
            raise AssertionError(f"V2 row array hash changed: {name}")
        if record.get("bytes") != (row_dir / name).stat().st_size:
            raise AssertionError(f"V2 row file size changed: {name}")
    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {str(cap) for cap in CAPS}:
        raise AssertionError("V2 row host reports changed")
    for cap in CAPS:
        host = hosts[str(cap)]
        hashes = host.get("entering_cp_state_hashes")
        if not isinstance(hashes, list) or len(hashes) != DEPTH or not all(
            isinstance(value, str) and V1_CONTRACT.HEX64.fullmatch(value) for value in hashes
        ):
            raise AssertionError("V2 entering CP hash chain changed")
        chain = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
        if host.get("entering_cp_chain_sha256") != chain:
            raise AssertionError("V2 entering CP chain digest changed")
        for field in ("raw32_sha256", "edges_sha256", "host_sha256", "sigma_sha256"):
            if not V1_CONTRACT.HEX64.fullmatch(str(host.get(field, ""))):
                raise AssertionError(f"V2 host hash malformed: {field}")
        if host["raw32_sha256"] != files[f"cp{cap}_raw_locals.npy"]["array_sha256"]:
            raise AssertionError("V2 raw-local host hash mismatch")
        if host["host_sha256"] != files[f"cp{cap}_host_rows.npy"]["array_sha256"]:
            raise AssertionError("V2 host-row hash mismatch")
        if host["sigma_sha256"] != files[f"cp{cap}_sigma.npy"]["array_sha256"]:
            raise AssertionError("V2 sigma hash mismatch")
        if not isinstance(host.get("flops_used"), int) or host["flops_used"] <= 0:
            raise AssertionError("V2 host FLOP receipt absent")
        if not math.isfinite(float(host.get("wall_seconds", math.nan))) or host["wall_seconds"] <= 0:
            raise AssertionError("V2 host timing receipt absent")
    if hosts["1536"]["edges_sha256"] != hosts["1792"]["edges_sha256"]:
        raise AssertionError("V2 cap-independent edge hash differs")
    return manifest


def verify_v1_bindings() -> dict[str, str]:
    observed = {}
    for name, record in V1_BINDINGS.items():
        digest = sha256_file(Path(record["path"]))
        if digest != record["sha256"]:
            raise AssertionError(f"immutable V1 byte changed: {name}")
        observed[name] = digest
    return observed


def assert_runtime_absent() -> None:
    for path in (CLAIM_PATH, STAGING_PATH, OUTPUT_PATH, SCRATCH_PATH):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"V2 runtime path exists during source freeze: {path.name}")
