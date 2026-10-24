#!/usr/bin/env python3
"""Frozen contracts for the additive target-blind V4 feature store.

This module is deliberately stdlib-only at import time.  In particular, it
does not import either full-data reader or any role-target reader.  NumPy is
loaded only by post-claim row/artifact validation call sites.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
V4 = HERE.parent / "v4_launch"
V3 = HERE.parent / "v3"

SOURCE_MANIFEST_PATH = HERE / "SOURCE_MANIFEST.json"
RELEASE_SCHEMA_PATH = HERE / "RELEASE_SCHEMA.json"
CLAIM_PATH = HERE / "PHASE2_OBSERVER_V4_FEATURE_STORE_ONCE.CLAIM"
STAGING_PATH = HERE / "sealed_target_blind_feature_store.staging"
OUTPUT_PATH = HERE / "sealed_target_blind_feature_store"
SCRATCH_PATH = HERE / ".feature_store_row_scratch"

ROW_SCHEMA = "phase2-observer-v4-target-blind-feature-row-v1"
CANDIDATE_SCHEMA = "phase2-observer-v4-target-blind-feature-store-candidate-v1"
FINAL_SCHEMA = "phase2-observer-v4-sealed-target-blind-feature-store-v1"
GENERATION_RELEASE_SCHEMA = "phase2-observer-v4-feature-store-generation-release-v1"
FINALIZATION_RELEASE_SCHEMA = "phase2-observer-v4-feature-store-finalization-release-v1"

WIDTH = 1024
DEPTH = 16
LOCAL_FEATURES = 20
CAPS = (1536, 1792)
ROLE_ORDER = ("observer_fit", "observer_selection")
ROLE_BINDINGS = {
    "observer_fit": {
        "rows": 688,
        "ordered_mlp_ids_sha256": "ba3d59f54ffacf4fe11d8ae6cd0f7ee1726486176217acc68002acb93be89de0",
        "global_rank_begin": 12,
        "global_rank_end": 700,
    },
    "observer_selection": {
        "rows": 88,
        "ordered_mlp_ids_sha256": "d1c719c608cadf5d4f7d852d23a9d935a23de0eae09d5dc1710257963aa1e755",
        "global_rank_begin": 712,
        "global_rank_end": 800,
    },
}

FROZEN_BINDINGS = {
    "custody_post_seal_audit": {
        "path": DATA_GATE / "INDEPENDENT_PRODUCTION_CUSTODIAN_V3_POST_SEAL_AUDIT_PASS.json",
        "sha256": "09a8c90833fedd44c541f9b61c74c1d4b4414f52b5b009cfe1ac630791a505ab",
    },
    "v4_implementation_manifest": {
        "path": V4 / "IMPLEMENTATION_MANIFEST_V4.json",
        "sha256": "626c79ab8899b7234c5850f225127614cda13decc59bece2a3c6f71bc4949408",
    },
    "v4_independent_launch_audit": {
        "path": V4 / "INDEPENDENT_LAUNCH_AUDIT_V4.json",
        "sha256": "26c51eb8cfbd29fc2400d68a104571892b9df58e91d1a86d3f4a5c158487dd62",
    },
    "v4_feature_contract": {
        "path": V4 / "FEATURE_STORE_CONTRACT_V4.json",
        "sha256": "b49086be2e4e81c07770bdb72ce92fdbd208f8315bf7accb7a00f9a916a7e44b",
    },
    "v4_host_worker": {
        "path": V4 / "flopscope_nonzero_worker.py",
        "sha256": "d25c245bd1bc0b4aa9b6b2fa5ea0ec79b169a8e6beaf73c86542be04b6131656",
    },
    "v3_host_base": {
        "path": V3 / "estimator.py",
        "sha256": "f8d48e5fc05d4de53173302e6fd8334c46908dd659a39baaebf2b534f5bdbea6",
    },
    "v3_host_generalized": {
        "path": V3 / "estimator_generalized.py",
        "sha256": "0b18dba56b6790f21640314f5904c7cd91d165558d90bc5c265898bb632a9310",
    },
    "primary_host_wrapper": {
        "path": V3 / "estimator_primary_cp1536_h128.py",
        "sha256": "800b3833c41a6809e83250e347f4234e54db943f2482ee96290c6b6173d1c019",
    },
    "control_host_wrapper": {
        "path": V3 / "estimator_control_cp1792_h64.py",
        "sha256": "c1ff172be0acde9ad020718f068fe613b8d04f8ca765c8b33f284cabd4f63418",
    },
    "target_blind_full_reader": {
        "path": DATA_GATE / "full_data_gate.py",
        "sha256": "6828136668cece104521f1b34460bf9d06de427649852653352bb09d9e801bef",
    },
    "full_roster": {
        "path": DATA_GATE / "full_network_splits.json",
        "sha256": "ba325d5c9ee4382ebf856b2e8d15330852fba5ac5627665fcd81dd9a5ef461d1",
    },
    "full_authority_extension": {
        "path": DATA_GATE / "FULL_SPLIT_AUTHORITY_EXTENSION.json",
        "sha256": "d61677658462f22949d8242f7bb92b24341502678f499d793184002fd3b6027a",
    },
    "full_cache_audit": {
        "path": DATA_GATE / "FULL_SPLIT_CACHE_AUDIT.json",
        "sha256": "43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd",
    },
    "full_gate_structure_audit": {
        "path": DATA_GATE / "FULL_SPLIT_GATE_STRUCTURE_AUDIT.json",
        "sha256": "8e75e15b52d411be194becf5f2f956e09b6b62d7970b9d0ea2d47a21e4cd5b77",
    },
}

EXPECTED_ENVIRONMENT = {
    "python": "3.11.5",
    "numpy": "2.4.6",
    "pyarrow": "25.0.1",
    "flopscope": "0.12.0",
    "whestbench": "0.16.0",
}
ENVIRONMENT_PYTHON = DATA_GATE / ".venv" / "bin" / "python"

MAX_WORKERS = 2
MAX_INVOCATIONS = 8
SYSTEM_MEMORY_FRACTION_NUMERATOR = 1
SYSTEM_MEMORY_FRACTION_DENOMINATOR = 4
PER_PROCESS_LIMIT_BYTES = 8 * (1 << 30)
RSS_SAMPLE_INTERVAL_SECONDS = 0.20
TERMINATION_GRACE_SECONDS = 5.0
MIN_FREE_BYTES = 60 * (1 << 30)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_fsynced(path: Path, payload: bytes, mode: int = 0o400) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def verify_frozen_bindings() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, binding in FROZEN_BINDINGS.items():
        path = Path(binding["path"])
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise AssertionError(f"frozen binding changed: {name}")
        observed[name] = digest
    return observed


def ordered_ids_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(f"{int(row['mlp_id'])}\n" for row in rows).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def load_exact_roster() -> dict[str, list[dict[str, Any]]]:
    binding = FROZEN_BINDINGS["full_roster"]
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise AssertionError("full roster changed")
    roster = json.loads(path.read_text(encoding="utf-8"))
    if roster.get("target_values_decoded") != 0:
        raise AssertionError("roster claims target decoding")
    if roster.get("decoded_value_columns") != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
        raise AssertionError("roster is not the four-column target-blind roster")
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[int] = set()
    for role in ROLE_ORDER:
        rows = list(roster["observer_roles"][role])
        binding_role = ROLE_BINDINGS[role]
        if len(rows) != binding_role["rows"]:
            raise AssertionError(f"role row count changed: {role}")
        if ordered_ids_sha256(rows) != binding_role["ordered_mlp_ids_sha256"]:
            raise AssertionError(f"role order changed: {role}")
        expected_ranks = list(
            range(binding_role["global_rank_begin"], binding_role["global_rank_end"])
        )
        if [int(row["global_hash_rank"]) for row in rows] != expected_ranks:
            raise AssertionError(f"global ranks changed: {role}")
        for role_rank, row in enumerate(rows):
            if row.get("observer_role") != role:
                raise AssertionError(f"row role changed: {role}/{role_rank}")
            if row.get("decoded_columns") != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
                raise AssertionError("roster row is not target blind")
            if not HEX64.fullmatch(str(row.get("weights_sha256", ""))):
                raise AssertionError("roster weight hash is malformed")
            mlp_id = int(row["mlp_id"])
            if mlp_id in seen:
                raise AssertionError("fit/selection roster has duplicate mlp_id")
            seen.add(mlp_id)
        result[role] = rows
    if len(seen) != 776:
        raise AssertionError("fit/selection roster is not exactly 776 rows")
    return result


def expected_row(role: str, role_rank: int) -> dict[str, Any]:
    if role not in ROLE_ORDER:
        raise PermissionError("only fit/selection feature roles are permitted")
    rows = load_exact_roster()[role]
    if role_rank < 0 or role_rank >= len(rows):
        raise IndexError("feature role rank out of range")
    return rows[role_rank]


def array_payload_sha256(array: Any) -> str:
    """Match the frozen V4 `_array_hash` dtype/shape/payload convention."""

    import numpy as np  # post-claim/artifact-validation only

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def raw_array_sha256(array: Any) -> str:
    import numpy as np  # post-claim/artifact-validation only

    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def row_relative_path(role: str, role_rank: int) -> Path:
    if role not in ROLE_ORDER or role_rank < 0 or role_rank >= ROLE_BINDINGS[role]["rows"]:
        raise ValueError("invalid feature row address")
    return Path("rows") / role / f"{role_rank:04d}"


def _load_array(path: Path, *, shape: tuple[int, ...], nonnegative: bool = False, positive: bool = False) -> Any:
    import numpy as np  # post-claim/artifact-validation only

    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.dtype("<f4") or tuple(value.shape) != shape or not value.flags.c_contiguous:
        raise AssertionError(f"array schema changed: {path.name}")
    if not np.isfinite(value).all():
        raise AssertionError(f"nonfinite array: {path.name}")
    if nonnegative and np.any(value < 0):
        raise AssertionError(f"negative host value: {path.name}")
    if positive and np.any(value <= 0):
        raise AssertionError(f"nonpositive sigma: {path.name}")
    return value


def validate_row_directory(row_dir: Path, *, role: str, role_rank: int) -> dict[str, Any]:
    """Fully validate one completed row directory and every stored byte."""

    expected = expected_row(role, role_rank)
    manifest_path = row_dir / "row_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ROW_SCHEMA or manifest.get("status") != "SEALED_ROW_LOCAL_FEATURES":
        raise AssertionError("row manifest schema/status changed")
    scalar_expectations = {
        "role": role,
        "role_rank": role_rank,
        "mlp_id": int(expected["mlp_id"]),
        "mlp_seed": int(expected["mlp_seed"]),
        "source_ordinal": int(expected["ordinal"]),
        "source_shard_index": int(expected["shard_index"]),
        "source_row_in_shard": int(expected["row_in_shard"]),
        "weights_raw_sha256": str(expected["weights_sha256"]),
        "target_values_decoded": 0,
    }
    for key, value in scalar_expectations.items():
        if manifest.get(key) != value:
            raise AssertionError(f"row binding changed: {key}")
    if manifest.get("decoded_columns") != ["mlp_id", "mlp_name", "mlp_seed", "weights"]:
        raise AssertionError("row source projection changed")
    if manifest.get("host_caps") != list(CAPS):
        raise AssertionError("row host caps changed")
    if manifest.get("host_latch") != [True] * DEPTH:
        raise AssertionError("row host latch is not all true")

    files = manifest.get("files")
    expected_names = {"weights.npy"}
    for cap in CAPS:
        expected_names.update(
            {f"cp{cap}_raw_locals.npy", f"cp{cap}_host_rows.npy", f"cp{cap}_sigma.npy"}
        )
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("row file membership changed")
    if set(path.name for path in row_dir.iterdir()) != expected_names | {"row_manifest.json"}:
        raise AssertionError("row directory has unbound members")
    if manifest.get("worker_source_sha256") != sha256_file(HERE / "host_feature_worker.py"):
        raise AssertionError("row was produced by another worker source")
    if manifest.get("frozen_host_worker_sha256") != FROZEN_BINDINGS["v4_host_worker"]["sha256"]:
        raise AssertionError("row binds another frozen host worker")
    if manifest.get("frozen_host_base_sha256") != FROZEN_BINDINGS["v3_host_base"]["sha256"]:
        raise AssertionError("row binds another frozen host base")
    if manifest.get("worker_environment") != EXPECTED_ENVIRONMENT:
        raise AssertionError("row worker environment changed")
    for counter in (
        "target_columns_projected", "target_values_decoded",
        "optimizers_constructed", "fits", "scores",
    ):
        if manifest.get(counter) != 0:
            raise AssertionError(f"row counter is nonzero: {counter}")
    if not isinstance(manifest.get("worker_max_rss_bytes"), int) or not (
        0 < manifest["worker_max_rss_bytes"] < PER_PROCESS_LIMIT_BYTES
    ):
        raise AssertionError("row worker RSS violates the strict guard")

    arrays: dict[str, Any] = {}
    arrays["weights.npy"] = _load_array(row_dir / "weights.npy", shape=(DEPTH, WIDTH, WIDTH))
    if raw_array_sha256(arrays["weights.npy"]) != expected["weights_sha256"]:
        raise AssertionError("stored weights differ from the roster attestation")
    for cap in CAPS:
        arrays[f"cp{cap}_raw_locals.npy"] = _load_array(
            row_dir / f"cp{cap}_raw_locals.npy", shape=(DEPTH, WIDTH, LOCAL_FEATURES)
        )
        arrays[f"cp{cap}_host_rows.npy"] = _load_array(
            row_dir / f"cp{cap}_host_rows.npy", shape=(DEPTH, WIDTH), nonnegative=True
        )
        arrays[f"cp{cap}_sigma.npy"] = _load_array(
            row_dir / f"cp{cap}_sigma.npy", shape=(DEPTH, WIDTH), positive=True
        )
    for name, array in arrays.items():
        record = files[name]
        if record.get("sha256") != sha256_file(row_dir / name):
            raise AssertionError(f"row file hash changed: {name}")
        if record.get("array_sha256") != array_payload_sha256(array):
            raise AssertionError(f"row array hash changed: {name}")
        if record.get("bytes") != (row_dir / name).stat().st_size:
            raise AssertionError(f"row file size changed: {name}")

    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {str(cap) for cap in CAPS}:
        raise AssertionError("row host reports changed")
    for cap in CAPS:
        host = hosts[str(cap)]
        hashes = host.get("entering_cp_state_hashes")
        if not isinstance(hashes, list) or len(hashes) != DEPTH or not all(
            isinstance(value, str) and HEX64.fullmatch(value) for value in hashes
        ):
            raise AssertionError("entering CP hash chain changed")
        chain = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
        if host.get("entering_cp_chain_sha256") != chain:
            raise AssertionError("entering CP chain digest changed")
        for field in ("raw32_sha256", "edges_sha256", "host_sha256", "sigma_sha256"):
            if not HEX64.fullmatch(str(host.get(field, ""))):
                raise AssertionError(f"host hash is malformed: {field}")
        if host["raw32_sha256"] != files[f"cp{cap}_raw_locals.npy"]["array_sha256"]:
            raise AssertionError("raw-local host hash mismatch")
        if host["host_sha256"] != files[f"cp{cap}_host_rows.npy"]["array_sha256"]:
            raise AssertionError("host-row hash mismatch")
        if host["sigma_sha256"] != files[f"cp{cap}_sigma.npy"]["array_sha256"]:
            raise AssertionError("sigma hash mismatch")
        if not isinstance(host.get("flops_used"), int) or host["flops_used"] <= 0:
            raise AssertionError("host FLOP receipt is absent")
        if not math.isfinite(float(host.get("wall_seconds", math.nan))) or host["wall_seconds"] <= 0:
            raise AssertionError("host timing receipt is absent")
    if hosts["1536"]["edges_sha256"] != hosts["1792"]["edges_sha256"]:
        raise AssertionError("cap-independent edge hash differs across host caps")
    return manifest


def validate_candidate_metadata(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise AssertionError("wrong candidate manifest schema")
    if manifest.get("status") != "READY_FOR_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED":
        raise AssertionError("candidate status changed")
    if manifest.get("host_caps") != list(CAPS) or manifest.get("target_values_decoded") != 0:
        raise AssertionError("candidate host/target contract changed")
    if manifest.get("raw_full_dataset_target_projection") != "forbidden":
        raise AssertionError("candidate permits target projection")
    roles = manifest.get("roles", {})
    if set(roles) != set(ROLE_ORDER):
        raise AssertionError("candidate roles changed")
    for role in ROLE_ORDER:
        expected = ROLE_BINDINGS[role]
        roster_rows = load_exact_roster()[role]
        if roles[role].get("rows") != expected["rows"]:
            raise AssertionError("candidate role count changed")
        if roles[role].get("ordered_mlp_ids_sha256") != expected["ordered_mlp_ids_sha256"]:
            raise AssertionError("candidate role order changed")
        entries = roles[role].get("row_manifests")
        if not isinstance(entries, list) or len(entries) != expected["rows"]:
            raise AssertionError("candidate row membership changed")
        for rank, entry in enumerate(entries):
            if entry.get("role_rank") != rank:
                raise AssertionError("candidate row rank changed")
            if entry.get("mlp_id") != int(roster_rows[rank]["mlp_id"]):
                raise AssertionError("candidate row mlp_id changed")
            if entry.get("path") != str(row_relative_path(role, rank) / "row_manifest.json"):
                raise AssertionError("candidate row path changed")
            if not HEX64.fullmatch(str(entry.get("sha256", ""))):
                raise AssertionError("candidate row hash malformed")
    return manifest


def assert_runtime_paths_absent() -> None:
    for path in (CLAIM_PATH, STAGING_PATH, OUTPUT_PATH, SCRATCH_PATH):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"runtime path must be absent during source freeze: {path.name}")
