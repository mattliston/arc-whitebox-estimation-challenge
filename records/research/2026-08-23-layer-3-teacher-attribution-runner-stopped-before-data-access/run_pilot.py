#!/usr/bin/env python3
"""One-use, target-free runner for the corrected teacher feasibility pilot.

The default preflight reads zero dataset rows.  Execution is mechanically
blocked until a root release binds two independent audits and every frozen
source/asset hash.  A durable claim is consumed before the dataset module is
imported or an output directory is created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Iterable

import numpy as np

import pilot_core as core


HERE = Path(__file__).resolve().parent
DESIGN_DIR = HERE.parent
DATA_GATE = DESIGN_DIR.parent / "data_gate"
FOOTHOLD_DIR = DESIGN_DIR.parent / "exact_layer_foothold"
RELEASE_PATH = HERE / "PILOT_RELEASE.json"
RECEIPTS_DIR = HERE / "receipts"
RUNS_DIR = HERE / "runs"
REQUIRED_RELEASE_STATUS = "ROOT_AUTHORIZED_ONE_TARGET_FREE_TEACHER_PILOT"
REQUIRED_AUDIT_STATUSES = {
    "static_interface_audit": "PASS_FROZEN_TEACHER_PILOT_STATIC_MATH_AND_INTERFACE",
    "memory_resource_audit": "PASS_FROZEN_TEACHER_PILOT_MEMORY_AND_RESOURCE",
}
ZERO_COUNTERS = {
    "target_values_decoded": 0,
    "target_metrics_computed": 0,
    "fits": 0,
    "retries": 0,
    "sweeps": 0,
    "variants": 0,
    "submissions": 0,
}
FROZEN_LOCAL_FILES = {
    "predecessor_design": DESIGN_DIR / "DESIGN.md",
    "predecessor_static_resource": DESIGN_DIR / "STATIC_RESOURCE_ESTIMATE.json",
    "predecessor_independent_hold_audit": DESIGN_DIR / "INDEPENDENT_DESIGN_AUDIT.md",
    "sealed_foothold_diagnostic": FOOTHOLD_DIR / "DIAGNOSTIC.md",
    "sealed_foothold_verifier": FOOTHOLD_DIR / "verify_exact_layer_foothold.py",
    "design_v2_preserved": DESIGN_DIR / "DESIGN_V2.md",
    "design_v2_audited_law_correction": DESIGN_DIR / "DESIGN_V2_AUDITED_LAW_CORRECTION.md",
    "static_resource_v2": DESIGN_DIR / "STATIC_RESOURCE_ESTIMATE_V2.json",
    "preregistration_preserved": HERE / "PREREGISTRATION.md",
    "preregistration_v1law_correction": HERE / "PREREGISTRATION_V1LAW_CORRECTION.md",
    "seed_split_authority_v1": HERE / "SEED_SPLIT_AUTHORITY_V1.json",
    "interface_preserved": HERE / "PILOT_INTERFACE.json",
    "interface_v1law": HERE / "PILOT_INTERFACE_V1LAW.json",
    "pilot_resource": HERE / "PILOT_RESOURCE_ESTIMATE.json",
    "probe_generator_preserved": HERE / "build_probe_asset.py",
    "probe_spec_preserved": HERE / "PROBE_SPEC.json",
    "probe_bytes_preserved": HERE / "probe32.f32le",
    "probe_generator_v1law": HERE / "build_probe_asset_v1law.py",
    "probe_spec_v1law": HERE / "PROBE_SPEC_V1LAW.json",
    "probe_bytes_v1law": HERE / "probe32_v1law.f32le",
    "pilot_core": HERE / "pilot_core.py",
    "pilot_driver": HERE / "run_pilot.py",
    "static_verifier": HERE / "verify_pilot_static.py",
    "receipts_parent": HERE / "receipts" / "README.md",
    "runs_parent": HERE / "runs" / "README.md",
}
PREDECESSOR_HASHES = {
    "predecessor_design": "bb8ca6d673fef1c660c695e8baad14d9739a53efe9f4e22c7c52a15d4df25dd0",
    "predecessor_static_resource": "28d46b24db3eef68fce4f0fb96d66b59c774a647d75a0e6c9586cc2968b39da6",
    "predecessor_independent_hold_audit": "96254da6f8e05f494eef2b8e3fe3355e1b7427142f8d5c9c5a706eab28b903c4",
}
FOOTHOLD_HASHES = {
    "sealed_foothold_diagnostic": "1a0e3b63c454b543793d2e3195a61eda638b1a958f27651980093f62ec1acb0a",
    "sealed_foothold_verifier": "55a4019ecda1a4d35b47560794c67bb4af61ff590dca57fdb030810be59d3db2",
}
PRESERVED_SUCCESSOR_HASHES = {
    "design_v2_preserved": "4879bb5c7d77345a3abbcc1f6a815367e3ca9b64500ad11a1f6286f235ee9438",
    "static_resource_v2": "f237d8499dced153940a8253d7aa8ccb82e9aa410212ebf190368ad0c8ba513c",
    "preregistration_preserved": "89e6e2d4b24935792cddb50857cc1e56615b4c86144d8ba2dd850b153c046fd4",
    "interface_preserved": "32a7f445b0da65d9710596a1cd784081c14337d926dd658136236cb6c284ed75",
    "probe_generator_preserved": "c5470e902540cbb0927b776544b577acd39876fc96b2d8fc5c0a1369018e7cec",
    "probe_bytes_preserved": "a56354144e5ece82891f6c5a219bd8d70764e7a1098b139a42a50444944b40b0",
    "probe_spec_preserved": "f08b5bb3f0087cc1d370c16cdb7a4ac13eb1419a7ddabc4252095108b601d88e",
}
RESOURCE_ABORT_BYTES = 6442450944
RESOURCE_HIGHWATER = {
    "combined_bytes": 0,
    "process_peak_rss_bytes": 0,
    "mps_current_allocated_bytes": 0,
    "mps_driver_allocated_bytes": 0,
    "stage": "not_sampled",
}


def current_resource_usage() -> dict[str, int]:
    process_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss is KiB on Linux and bytes on macOS.
    if sys.platform.startswith("linux"):
        process_peak *= 1024
    mps_current = 0
    mps_driver = 0
    try:
        import torch  # type: ignore[import-not-found]

        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            mps_current = int(torch.mps.current_allocated_memory())
            mps_driver = int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return {
        "process_peak_rss_bytes": process_peak,
        "mps_current_allocated_bytes": mps_current,
        "mps_driver_allocated_bytes": mps_driver,
        # Deliberately conservative: shared MPS pages may also appear in RSS.
        "combined_bytes": process_peak + mps_current + mps_driver,
    }


def enforce_resource_limit(
    stage: str,
    *,
    limit_bytes: int = RESOURCE_ABORT_BYTES,
    reading: dict[str, int] | None = None,
) -> dict[str, int]:
    usage = current_resource_usage() if reading is None else dict(reading)
    required = {
        "process_peak_rss_bytes",
        "mps_current_allocated_bytes",
        "mps_driver_allocated_bytes",
        "combined_bytes",
    }
    if set(usage) != required or any(value < 0 for value in usage.values()):
        raise AssertionError("invalid resource-guard reading")
    if usage["combined_bytes"] >= limit_bytes:
        raise MemoryError(
            f"resource abort at {stage}: {usage['combined_bytes']} >= {limit_bytes}"
        )
    if usage["combined_bytes"] > RESOURCE_HIGHWATER["combined_bytes"]:
        RESOURCE_HIGHWATER.update(usage)
        RESOURCE_HIGHWATER["stage"] = stage
    return usage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def local_bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for label, path in FROZEN_LOCAL_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen file {label}: {path}")
        result[label] = sha256_file(path)
    if any(result[label] != digest for label, digest in PREDECESSOR_HASHES.items()):
        raise AssertionError("predecessor authority chain changed")
    if any(result[label] != digest for label, digest in FOOTHOLD_HASHES.items()):
        raise AssertionError("sealed exact-layer foothold binding changed")
    if any(
        result[label] != digest for label, digest in PRESERVED_SUCCESSOR_HASHES.items()
    ):
        raise AssertionError("preserved successor evidence changed")
    return result


def load_interface() -> dict[str, object]:
    interface = json.loads(
        (HERE / "PILOT_INTERFACE_V1LAW.json").read_text(encoding="utf-8")
    )
    if interface["status"] != "STAGED_EXECUTION_BLOCKED":
        raise AssertionError("interface status changed")
    if tuple(interface["arms"]["masks"]) != core.ARM_MASKS:
        raise AssertionError("interface/core arm mismatch")
    if tuple(interface["sampling"]["total_levels"]) != core.TOTAL_LEVELS:
        raise AssertionError("interface/core level mismatch")
    if interface["access_counters"] != ZERO_COUNTERS:
        raise AssertionError("interface access counters changed")
    if interface["namespace"] != "p2-teacher-v1":
        raise AssertionError("pilot namespace is not the audited v1 law")
    if interface["pilot_role"] != {
        "name": "mini_feasibility_only",
        "dataset_split": "mini",
        "allowed_ordinals": [0, 1],
        "full_split_roles_available": [],
        "target_roles_available": [],
    }:
        raise AssertionError("pilot role scope changed")
    source_role = interface["target_blind_source"]
    if (
        source_role["dataset_split"] != "mini"
        or source_role["allowed_ordinals"] != [0, 1]
        or source_role["full_split_iterator_available"]
        or source_role["target_role_available"]
    ):
        raise AssertionError("target-blind mini source role changed")
    observed_ordinals = [
        int(case["ordinal"])
        for case in interface["cases"]
        if case["kind"] == "target_blind_mini"
    ]
    if observed_ordinals != [0, 1]:
        raise AssertionError("pilot case ordinals changed")
    foothold = interface["sealed_exact_layer_foothold"]
    if foothold["diagnostic_sha256"] != FOOTHOLD_HASHES["sealed_foothold_diagnostic"]:
        raise AssertionError("interface foothold diagnostic binding changed")
    if foothold["verifier_sha256"] != FOOTHOLD_HASHES["sealed_foothold_verifier"]:
        raise AssertionError("interface foothold verifier binding changed")
    if foothold["deep_closure_attribution_starts_at_layer_1based"] != 3:
        raise AssertionError("deep closure onset changed")
    return interface


def load_probe(interface: dict[str, object]) -> np.ndarray:
    probe_binding = interface["probe"]
    spec = json.loads((HERE / probe_binding["spec_path"]).read_text(encoding="utf-8"))
    payload = (HERE / probe_binding["path"]).read_bytes()
    if len(payload) != int(spec["bytes"]):
        raise AssertionError("probe byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise AssertionError("probe payload hash mismatch")
    if spec["sha256"] != probe_binding["sha256"] or spec["literal"] != "p2-teacher-v1|probe32":
        raise AssertionError("probe audited-law binding mismatch")
    probe = np.frombuffer(payload, dtype="<f4").reshape(spec["shape"])
    gram = probe.astype(np.float64).T @ probe.astype(np.float64)
    if not np.array_equal(gram, np.eye(probe.shape[1])):
        raise AssertionError("probe is not exactly orthonormal in float64")
    return probe.copy()


def validate_external_source_hashes(interface: dict[str, object]) -> None:
    source = interface["target_blind_source"]
    loader = (HERE / source["path"]).resolve()
    authority = (HERE / source["authority_path"]).resolve()
    if loader != (DATA_GATE / "target_blind_mini.py").resolve():
        raise AssertionError("loader path escaped fixed data gate")
    if sha256_file(loader) != source["sha256"]:
        raise AssertionError("target-blind loader hash changed")
    if sha256_file(authority) != source["authority_sha256"]:
        raise AssertionError("dataset authority hash changed")
    host = interface["host"]
    host_path = (HERE / host["path"]).resolve()
    expected_host = (DESIGN_DIR.parent / "cp3_cap1792_k3_readout" / "estimator.py").resolve()
    if host_path != expected_host or sha256_file(host_path) != host["sha256"]:
        raise AssertionError("frozen host binding changed")


def preflight() -> dict[str, object]:
    interface = load_interface()
    bindings = local_bindings()
    validate_external_source_hashes(interface)
    probe = load_probe(interface)
    self_test = core.small_exact_self_test()
    release_present = RELEASE_PATH.exists()
    return {
        "schema": "phase2-teacher-pilot-zero-row-preflight-v1",
        "status": (
            "BLOCKED_PENDING_AUDITS_AND_ROOT_RELEASE"
            if not release_present
            else "RELEASE_PRESENT_NOT_EXECUTED"
        ),
        "bindings": bindings,
        "probe_sha256": core.payload_sha256(probe),
        "probe_shape": list(probe.shape),
        "small_exact_self_test": self_test,
        "release_present": release_present,
        "dataset_module_imported": False,
        "dataset_rows_decoded": 0,
        **ZERO_COUNTERS,
    }


def validate_release() -> tuple[dict[str, object], str, dict[str, str]]:
    """Validate every authority binding before the irreversible claim."""

    if not RELEASE_PATH.is_file():
        raise PermissionError("PILOT_RELEASE.json is absent; execution is blocked")
    release_bytes = RELEASE_PATH.read_bytes()
    release_sha = hashlib.sha256(release_bytes).hexdigest()
    release = json.loads(release_bytes)
    bindings = local_bindings()
    if release.get("status") != REQUIRED_RELEASE_STATUS:
        raise PermissionError("root release status is not execution authority")
    if release.get("scope") != "target_blind_teacher_pilot_only":
        raise PermissionError("release scope mismatch")
    if release.get("access_counters") != ZERO_COUNTERS:
        raise PermissionError("release does not bind every forbidden counter to zero")
    if release.get("frozen_sha256") != bindings:
        raise PermissionError("release frozen-source binding mismatch")
    release_id = str(release.get("release_id", ""))
    if len(release_id) != 64 or any(ch not in "0123456789abcdef" for ch in release_id):
        raise PermissionError("release_id must be a lowercase SHA-256 token")
    expected_claim = RECEIPTS_DIR / f"teacher-pilot-{release_id}.one-use-claim.json"
    if not RECEIPTS_DIR.is_dir() or not (RECEIPTS_DIR / "README.md").is_file():
        raise PermissionError("frozen receipts parent missing")
    if not RUNS_DIR.is_dir() or not (RUNS_DIR / "README.md").is_file():
        raise PermissionError("frozen runs parent missing")
    if Path(release.get("claim_path", "")).resolve() != expected_claim.resolve():
        raise PermissionError("release claim path mismatch")
    expected_output = RUNS_DIR / f"teacher-pilot-{release_id}"
    if Path(release.get("output_path", "")).resolve() != expected_output.resolve():
        raise PermissionError("release output path mismatch")
    if expected_claim.exists() or expected_output.exists():
        raise FileExistsError("one-use authority already consumed or output exists")

    for label, required_status in REQUIRED_AUDIT_STATUSES.items():
        audit_binding = release.get(label)
        if not isinstance(audit_binding, dict):
            raise PermissionError(f"missing {label} binding")
        audit_path = Path(str(audit_binding.get("path", ""))).resolve()
        if HERE not in audit_path.parents:
            raise PermissionError(f"{label} escaped pilot directory")
        if sha256_file(audit_path) != audit_binding.get("sha256"):
            raise PermissionError(f"{label} hash mismatch")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != required_status:
            raise PermissionError(f"{label} status mismatch")
        if audit.get("frozen_sha256") != bindings:
            raise PermissionError(f"{label} source binding mismatch")
        if audit.get("access_counters") != ZERO_COUNTERS:
            raise PermissionError(f"{label} access counters mismatch")
    return release, release_sha, bindings


def write_one_use_claim(
    release: dict[str, object], release_sha: str, bindings: dict[str, str]
) -> tuple[Path, str]:
    claim_path = Path(str(release["claim_path"])).resolve()
    if claim_path.parent != RECEIPTS_DIR.resolve() or not RECEIPTS_DIR.is_dir():
        raise PermissionError("fixed receipts parent missing")
    claim = {
        "schema": "phase2-teacher-pilot-one-use-claim-v1",
        "status": "CLAIM_CONSUMED_BEFORE_DATASET_IMPORT_AND_OUTPUT_CREATION",
        "release_sha256": release_sha,
        "release_id": release["release_id"],
        "scope": release["scope"],
        "frozen_sha256": bindings,
        "static_interface_audit": release["static_interface_audit"],
        "memory_resource_audit": release["memory_resource_audit"],
        "output_path": release["output_path"],
        "dataset_module_imported_before_claim": False,
        "dataset_rows_decoded_before_claim": 0,
        "access_counters_before_claim": ZERO_COUNTERS,
    }
    payload = canonical_json_bytes(claim)
    descriptor = os.open(
        claim_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short claim write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(claim_path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return claim_path, hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> str:
    enforce_resource_limit(f"before_write_{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short output write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> str:
    return atomic_bytes(path, canonical_json_bytes(value))


def atomic_array(path: Path, value: np.ndarray, dtype: str) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=dtype)
    payload = array.tobytes(order="C")
    digest = atomic_bytes(path, payload)
    return {
        "path": path.name,
        "sha256": digest,
        "dtype": dtype,
        "shape": list(array.shape),
        "bytes": len(payload),
    }


def raw_tape_digest(tape: core.RawTape) -> str:
    digest = hashlib.sha256()
    digest.update(str(tape.count).encode("ascii") + b"\0")
    for array in (
        tape.pre_powers,
        tape.post_sum,
        tape.post_raw2,
        tape.pre_sq_cross,
    ):
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def seal_raw_tape(directory: Path, tape: core.RawTape) -> dict[str, object]:
    enforce_resource_limit(f"before_seal_raw_{directory.name}")
    tape.validate()
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "pre_powers": atomic_array(directory / "pre_powers.f64le", tape.pre_powers, "<f8"),
        "post_sum": atomic_array(directory / "post_sum.f64le", tape.post_sum, "<f8"),
        "post_raw2": atomic_array(directory / "post_raw2.f64le", tape.post_raw2, "<f8"),
        "pre_sq_cross": atomic_array(
            directory / "pre_sq_cross.f64le", tape.pre_sq_cross, "<f8"
        ),
    }
    manifest = {
        "schema": "phase2-teacher-raw-bank-level-v1",
        "count": tape.count,
        "raw_tape_sha256": raw_tape_digest(tape),
        "arrays": arrays,
        **ZERO_COUNTERS,
    }
    manifest["manifest_sha256"] = atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_raw_tape(directory: Path) -> core.RawTape:
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise AssertionError("raw bank manifest is not canonical JSON")
    if manifest.get("schema") != "phase2-teacher-raw-bank-level-v1":
        raise AssertionError("raw bank manifest schema changed")
    if any(manifest.get(label) != value for label, value in ZERO_COUNTERS.items()):
        raise AssertionError("raw bank manifest counter changed")

    def load(label: str) -> np.ndarray:
        spec = manifest["arrays"][label]
        path = directory / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"raw bank payload changed: {label}")
        return np.fromfile(path, dtype=spec["dtype"]).reshape(spec["shape"])

    tape = core.RawTape(
        count=int(manifest["count"]),
        pre_powers=load("pre_powers"),
        post_sum=load("post_sum"),
        post_raw2=load("post_raw2"),
        pre_sq_cross=load("pre_sq_cross"),
    )
    tape.validate()
    if raw_tape_digest(tape) != manifest["raw_tape_sha256"]:
        raise AssertionError("raw bank semantic hash mismatch")
    return tape


def raw_bank_binding(
    output: Path,
    sampler: str,
    bank: str,
    level: int,
) -> dict[str, object]:
    """Bind a derived level to its exact canonical bank and provenance bytes."""

    if sampler not in {"rqmc", "iid"} or bank not in core.BANKS:
        raise AssertionError("invalid raw-bank binding coordinates")
    directory = output / sampler / f"bank_{bank}" / f"N{level}"
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise AssertionError("raw bank manifest is not canonical JSON")
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    provenance_path = output / sampler / f"bank_{bank}_provenance.json"
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    if canonical_json_bytes(provenance) != provenance_bytes:
        raise AssertionError("raw bank provenance is not canonical JSON")
    if (
        provenance.get("schema") != "phase2-teacher-scramble-provenance-v1"
        or provenance.get("sampler") != sampler
        or provenance.get("bank") != bank
    ):
        raise AssertionError("raw bank provenance coordinates changed")
    if any(provenance.get(label) != value for label, value in ZERO_COUNTERS.items()):
        raise AssertionError("raw bank provenance counter changed")
    level_binding = provenance.get("levels", {}).get(str(level))
    if not isinstance(level_binding, dict):
        raise AssertionError("raw bank provenance level is absent")
    expected_level_binding = dict(manifest)
    expected_level_binding["manifest_sha256"] = manifest_file_sha256
    if level_binding != expected_level_binding:
        raise AssertionError("raw bank manifest/provenance binding mismatch")

    return {
        "sampler": sampler,
        "bank": bank,
        "total_level": level,
        "relative_directory": f"{sampler}/bank_{bank}/N{level}",
        "manifest_file_sha256": manifest_file_sha256,
        "provenance_file_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "raw_tape_sha256": manifest["raw_tape_sha256"],
        "count": manifest["count"],
        "array_sha256": {
            label: manifest["arrays"][label]["sha256"]
            for label in (
                "pre_powers",
                "post_sum",
                "post_raw2",
                "pre_sq_cross",
            )
        },
    }


def import_target_blind_loader_after_claim():
    loader_path = DATA_GATE / "target_blind_mini.py"
    spec = importlib.util.spec_from_file_location("p2_target_blind_mini", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct fixed target-blind loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weights_sha256(weights: np.ndarray) -> str:
    value = core.canonical_f32le(weights)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def load_cases_after_claim(interface: dict[str, object]) -> list[dict[str, object]]:
    cases = interface["cases"]
    depth = int(interface["shape"]["depth"])
    width = int(interface["shape"]["width"])
    result: list[dict[str, object]] = [
        {
            "label": cases[0]["label"],
            "network_seed": int(cases[0]["mlp_seed"]),
            "weights": np.broadcast_to(
                np.eye(width, dtype=np.float32), (depth, width, width)
            ),
            "identity": cases[0],
        }
    ]
    loader = import_target_blind_loader_after_claim()
    expected_by_ordinal = {int(case["ordinal"]): case for case in cases[1:]}
    found: set[int] = set()
    for row in loader.iter_target_blind_mini():
        if row.ordinal not in expected_by_ordinal:
            if row.ordinal > max(expected_by_ordinal):
                break
            continue
        expected = expected_by_ordinal[row.ordinal]
        identity = {
            "ordinal": row.ordinal,
            "mlp_id": row.mlp_id,
            "mlp_name": row.mlp_name,
            "mlp_seed": row.mlp_seed,
        }
        for label in ("ordinal", "mlp_id", "mlp_name", "mlp_seed"):
            if identity[label] != expected[label]:
                raise AssertionError(f"cached case identity mismatch: {label}")
        if tuple(row.decoded_columns) != tuple(
            interface["target_blind_source"]["allowed_value_columns"]
        ):
            raise AssertionError("physical projection allow-list changed")
        observed_hash = weights_sha256(row.weights)
        if observed_hash != expected["weights_sha256"]:
            raise AssertionError("cached case weight hash mismatch")
        result.append(
            {
                "label": f"mini{row.ordinal}_{row.mlp_name}",
                "network_seed": row.mlp_seed,
                "weights": row.weights.copy(),
                "identity": identity | {"weights_sha256": observed_hash},
            }
        )
        found.add(row.ordinal)
        if found == set(expected_by_ordinal):
            break
    if found != set(expected_by_ordinal) or len(result) != 3:
        raise AssertionError("did not load exactly the two frozen mini rows")
    return result


def stream_snapshots_cpu(
    weights: np.ndarray, blocks: Iterable[np.ndarray], prefixes: tuple[int, ...]
) -> dict[int, core.RawTape]:
    depth, width, _ = weights.shape
    tape = core.RawTape.zeros(depth, width)
    snapshots: dict[int, core.RawTape] = {}
    for block in blocks:
        enforce_resource_limit("cpu_before_sample_block")
        core.accumulate_cpu_block(tape, weights, block)
        enforce_resource_limit("cpu_after_sample_block")
        if tape.count in prefixes:
            enforce_resource_limit(f"cpu_before_nested_prefix_{tape.count}")
            snapshots[tape.count] = tape.copy()
            enforce_resource_limit(f"cpu_after_nested_prefix_{tape.count}")
        elif any(tape.count > prefix for prefix in prefixes if prefix not in snapshots):
            raise AssertionError("CPU block crossed a nested prefix")
    if tuple(snapshots) != prefixes:
        raise AssertionError("CPU stream missed a nested prefix")
    return snapshots


def sampler_bank_products(
    output: Path,
    weights: np.ndarray,
    network_seed: int,
    sampler: str,
    backend: str,
) -> dict[str, object]:
    depth, width, _ = weights.shape
    per_replicate = core.PER_REPLICATE_PREFIXES
    bank_manifests: dict[str, dict[str, object]] = {}
    for bank_name, replicates in core.BANKS.items():
        pooled = {prefix: core.RawTape.zeros(depth, width) for prefix in per_replicate}
        provenance: list[dict[str, object]] = []
        for replicate in replicates:
            blocks = core.iter_normal_blocks(
                width=width,
                network_seed=network_seed,
                sampler=sampler,
                replicate=replicate,
                count=per_replicate[-1],
                batch_size=256,
            )
            if backend == "mps":
                snapshots = core.accumulate_mps_stream(
                    weights, blocks, per_replicate, guard=enforce_resource_limit
                )
            elif backend == "cpu":
                snapshots = stream_snapshots_cpu(weights, blocks, per_replicate)
            else:
                raise AssertionError("unrecognized backend")
            level_receipts = []
            for prefix in per_replicate:
                enforce_resource_limit(
                    f"before_pool_{sampler}_{bank_name}_rep{replicate}_N{prefix * 8}"
                )
                pooled[prefix].add_(snapshots[prefix])
                enforce_resource_limit(
                    f"after_pool_{sampler}_{bank_name}_rep{replicate}_N{prefix * 8}"
                )
                level_receipts.append(
                    {
                        "per_replicate_prefix": prefix,
                        "total_level": prefix * 8,
                        "raw_tape_sha256": raw_tape_digest(snapshots[prefix]),
                    }
                )
            provenance.append(
                {
                    "replicate": replicate,
                    "seed": core.stream_seed(network_seed, sampler, replicate),
                    "nested_single_stream": True,
                    "levels": level_receipts,
                }
            )
        bank_levels: dict[str, object] = {}
        for prefix in per_replicate:
            total_level = prefix * 8
            directory = output / sampler / f"bank_{bank_name}" / f"N{total_level}"
            bank_levels[str(total_level)] = seal_raw_tape(directory, pooled[prefix])
        provenance_receipt = {
            "schema": "phase2-teacher-scramble-provenance-v1",
            "sampler": sampler,
            "bank": bank_name,
            "backend": backend,
            "replicates": provenance,
            "levels": bank_levels,
            **ZERO_COUNTERS,
        }
        provenance_path = output / sampler / f"bank_{bank_name}_provenance.json"
        bank_manifests[bank_name] = {
            "provenance_sha256": atomic_json(provenance_path, provenance_receipt),
            "levels": bank_levels,
        }
    return bank_manifests


def action_products(
    host: core.HostTape,
    teacher: core.TeacherTape,
    weights: np.ndarray,
    probe: np.ndarray,
) -> dict[str, np.ndarray]:
    covariance_diagonal = []
    covariance_offdiagonal = []
    covariance_total = []
    covariance_probe = []
    consumer_covariance_mse = []
    consumer_next_variance_error = []
    host_next_K21_probe_error = []
    for layer in range(weights.shape[0] - 1):
        covariance = core.covariance_action_split(
            host.post_covariance[layer],
            teacher.post_covariance[layer],
            weights[layer + 1],
            probe,
        )
        covariance_diagonal.append(covariance.diagonal)
        covariance_offdiagonal.append(covariance.offdiagonal)
        covariance_total.append(covariance.total)
        covariance_probe.append(covariance.probe)
        consumer_post = core.teacher_compatible_K21_consumer(
            teacher.pre_mean[layer],
            teacher.pre_covariance[layer],
            teacher.pre_K3[layer],
            teacher.pre_K21[layer],
        )
        consumer_covariance_mse.append(
            np.mean((consumer_post - teacher.post_covariance[layer]) ** 2)
        )
        consumer_next = core.next_variance_action(consumer_post, weights[layer + 1])
        direct_next = teacher.pre_variance[layer + 1]
        consumer_next_variance_error.append(consumer_next - direct_next)
        direct_K21_probe = teacher.pre_K21[layer + 1] @ probe.astype(np.float64)
        host_next_K21_probe_error.append(
            host.pre_K21_probe[layer + 1] - direct_K21_probe
        )
    return {
        "covariance_diagonal": np.stack(covariance_diagonal),
        "covariance_offdiagonal": np.stack(covariance_offdiagonal),
        "covariance_total": np.stack(covariance_total),
        "covariance_probe": np.stack(covariance_probe),
        "consumer_post_covariance_mse": np.asarray(
            consumer_covariance_mse, dtype=np.float64
        ),
        "consumer_next_variance_error": np.stack(consumer_next_variance_error),
        "host_next_K21_probe_error": np.stack(host_next_K21_probe_error),
    }


def analytic_foothold_control(
    host: core.HostTape,
    teacher: core.TeacherTape,
    predictions: np.ndarray,
) -> dict[str, object]:
    """Separate exact L1/L2 source controls from deep closure attribution."""

    layer1 = {
        "authority": "exact_Gaussian_preactivation_and_exact_Gaussian_ReLU",
        "host_pre_mean_mse": float(
            np.mean((host.pre_mean[0] - teacher.pre_mean[0]) ** 2)
        ),
        "host_pre_covariance_mse": float(
            np.mean((host.pre_covariance[0] - teacher.pre_covariance[0]) ** 2)
        ),
        "host_0000_ReLU_mean_mse": float(
            np.mean((predictions[0, 0] - teacher.post_mean[0]) ** 2)
        ),
        "all_teacher_1111_ReLU_mean_mse": float(
            np.mean((predictions[15, 0] - teacher.post_mean[0]) ** 2)
        ),
        "interpretation": "exact calibration control only",
    }
    if teacher.pre_mean.shape[0] < 2:
        return {"layer1": layer1, "layer2": None}
    layer2 = {
        "authority": "exact_pre_mean_and_pre_covariance_from_layer1_Gaussian_ReLU",
        "host_pre_mean_mse": float(
            np.mean((host.pre_mean[1] - teacher.pre_mean[1]) ** 2)
        ),
        "host_pre_covariance_mse": float(
            np.mean((host.pre_covariance[1] - teacher.pre_covariance[1]) ** 2)
        ),
        "host_0000_vs_direct_teacher_ReLU_mean_mse": float(
            np.mean((predictions[0, 1] - teacher.post_mean[1]) ** 2)
        ),
        "all_teacher_1111_vs_direct_teacher_ReLU_mean_mse": float(
            np.mean((predictions[15, 1] - teacher.post_mean[1]) ** 2)
        ),
        "interpretation": (
            "near-exact readout foothold; no L1/L2 substitution gain may be "
            "generalized to deep closure capacity"
        ),
    }
    return {"layer1": layer1, "layer2": layer2}


def derive_and_seal_level(
    output: Path,
    sampler: str,
    level: int,
    weights: np.ndarray,
    host: core.HostTape,
    probe: np.ndarray,
) -> dict[str, object]:
    enforce_resource_limit(f"before_derive_{sampler}_N{level}")
    source_raw_banks = {
        bank: raw_bank_binding(output, sampler, bank, level)
        for bank in ("A", "B")
    }
    raw_a = load_raw_tape(output / sampler / "bank_A" / f"N{level}")
    raw_b = load_raw_tape(output / sampler / "bank_B" / f"N{level}")
    teacher_a = core.normalize_raw_tape(raw_a, weights, guard=enforce_resource_limit)
    teacher_b = core.normalize_raw_tape(raw_b, weights, guard=enforce_resource_limit)
    teacher_full = core.normalize_raw_tape(
        core.merge_raw_tapes([raw_a, raw_b]), weights, guard=enforce_resource_limit
    )
    enforce_resource_limit(f"after_normalize_{sampler}_N{level}")
    prediction_a = core.factorial_predictions(host, teacher_a)
    prediction_b = core.factorial_predictions(host, teacher_b)
    prediction_full = core.factorial_predictions(host, teacher_full)
    prediction_error = core.bank_error(prediction_a, prediction_b)
    action_a = action_products(host, teacher_a, weights, probe)
    action_b = action_products(host, teacher_b, weights, probe)
    action_full = action_products(host, teacher_full, weights, probe)

    level_dir = output / sampler / f"derived_N{level}"
    level_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, object] = {}
    for label, array in {
        "predictions_bank_A": prediction_a,
        "predictions_bank_B": prediction_b,
        "predictions_full_eight": prediction_full,
        "prediction_D2_over_4": prediction_error.full_eight_proxy,
        "prediction_D2_over_2": prediction_error.conservative,
        "teacher_full_pre_mean": teacher_full.pre_mean,
        "teacher_full_pre_variance": teacher_full.pre_variance,
        "teacher_full_pre_K3": teacher_full.pre_K3,
        "teacher_full_pre_K4": teacher_full.pre_K4,
        "teacher_full_pre_K21_probe": np.einsum(
            "lij,jr->lir", teacher_full.pre_K21, probe.astype(np.float64)
        ),
    }.items():
        arrays[label] = atomic_array(level_dir / f"{label}.f64le", array, "<f8")

    action_errors: dict[str, object] = {}
    conservative_values = [float(np.max(np.mean(prediction_error.conservative, axis=2)))]
    for label in sorted(action_full):
        full_array = np.asarray(action_full[label], dtype=np.float64)
        error = core.bank_error(action_a[label], action_b[label])
        arrays[f"action_{label}_full"] = atomic_array(
            level_dir / f"action_{label}_full.f64le", full_array, "<f8"
        )
        arrays[f"action_{label}_D2_over_4"] = atomic_array(
            level_dir / f"action_{label}_D2_over_4.f64le",
            error.full_eight_proxy,
            "<f8",
        )
        arrays[f"action_{label}_D2_over_2"] = atomic_array(
            level_dir / f"action_{label}_D2_over_2.f64le", error.conservative, "<f8"
        )
        reduction_axes = tuple(range(1, full_array.ndim))
        signal_by_layer = (
            np.mean(full_array * full_array, axis=reduction_axes)
            if reduction_axes
            else full_array * full_array
        )
        conservative_by_layer = (
            np.mean(error.conservative, axis=reduction_axes)
            if reduction_axes
            else error.conservative
        )
        aggregate_by_layer = (
            np.mean(full_array, axis=reduction_axes)
            if reduction_axes
            else full_array
        )
        limits = np.maximum(1.0e-12, 0.1 * signal_by_layer)
        within = conservative_by_layer <= limits
        conservative_values.append(float(np.max(conservative_by_layer)))
        action_errors[label] = {
            "per_source_layer": [
                {
                    "source_layer_1based": index + 1,
                    "signal_mse": float(signal_by_layer[index]),
                    "conservative_error_mse": float(conservative_by_layer[index]),
                    "limit": float(limits[index]),
                    "within_limit": bool(within[index]),
                    "aggregate_sign": int(np.sign(aggregate_by_layer[index])),
                }
                for index in range(full_array.shape[0])
            ],
            "all_layers_within_limit": bool(np.all(within)),
        }

    manifest = {
        "schema": "phase2-teacher-derived-level-v1",
        "sampler": sampler,
        "total_level": level,
        "bank_counts": {"A": raw_a.count, "B": raw_b.count},
        "full_count": teacher_full.count,
        "pool_before_nonlinear": True,
        "source_raw_banks": source_raw_banks,
        "arm_masks": list(core.ARM_MASKS),
        "arrays": arrays,
        "action_certificates": action_errors,
        "analytic_foothold_control": analytic_foothold_control(
            host, teacher_full, prediction_full
        ),
        "maximum_relevant_conservative_mse": max(conservative_values),
        **ZERO_COUNTERS,
    }
    manifest["manifest_sha256"] = atomic_json(level_dir / "manifest.json", manifest)
    return manifest


def saved_input_blocks(
    directory: Path, width: int, network_seed: int, sampler: str
) -> dict[int, list[np.ndarray]]:
    result: dict[int, list[np.ndarray]] = {}
    for replicate in range(core.REPLICATES):
        enforce_resource_limit(f"before_saved_input_{sampler}_rep{replicate}")
        blocks = list(
            core.iter_normal_blocks(
                width=width,
                network_seed=network_seed,
                sampler=sampler,
                replicate=replicate,
                count=core.PER_REPLICATE_PREFIXES[0],
                batch_size=256,
            )
        )
        inputs = np.concatenate(blocks, axis=0)
        atomic_array(
            directory / f"{sampler}_rep{replicate}_N2048.f32le", inputs, "<f4"
        )
        enforce_resource_limit(f"after_saved_input_{sampler}_rep{replicate}")
        result[replicate] = [inputs[index : index + 256] for index in range(0, 2048, 256)]
    return result


def pooled_saved_inputs(
    weights: np.ndarray,
    saved: dict[int, list[np.ndarray]],
    backend: str,
) -> tuple[dict[str, core.RawTape], float]:
    started = time.perf_counter()
    pools: dict[str, core.RawTape] = {}
    for bank, replicates in core.BANKS.items():
        pool = core.RawTape.zeros(weights.shape[0], weights.shape[1])
        for replicate in replicates:
            if backend == "cpu":
                snapshots = stream_snapshots_cpu(weights, saved[replicate], (2048,))
            else:
                snapshots = core.accumulate_mps_stream(
                    weights,
                    saved[replicate],
                    (2048,),
                    guard=enforce_resource_limit,
                )
            pool.add_(snapshots[2048])
        pools[bank] = pool
    return pools, time.perf_counter() - started


def parity_summary(
    pools: dict[str, core.RawTape],
    weights: np.ndarray,
    host: core.HostTape,
    probe: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    teachers = {
        bank: core.normalize_raw_tape(raw, weights, guard=enforce_resource_limit)
        for bank, raw in pools.items()
    }
    predictions = {
        bank: core.factorial_predictions(host, teacher)
        for bank, teacher in teachers.items()
    }
    arrays = {
        "prediction_A": predictions["A"],
        "prediction_B": predictions["B"],
        "pre_mean_A": teachers["A"].pre_mean,
        "pre_mean_B": teachers["B"].pre_mean,
        "pre_variance_A": teachers["A"].pre_variance,
        "pre_variance_B": teachers["B"].pre_variance,
        "pre_K3_A": teachers["A"].pre_K3,
        "pre_K3_B": teachers["B"].pre_K3,
        "pre_K4_A": teachers["A"].pre_K4,
        "pre_K4_B": teachers["B"].pre_K4,
    }
    for bank in ("A", "B"):
        actions = action_products(host, teachers[bank], weights, probe)
        for label, array in actions.items():
            arrays[f"action_{label}_{bank}"] = np.asarray(array, dtype=np.float64)
    hashes = {label: core.payload_sha256(array) for label, array in arrays.items()}
    summary = {
        "schema": "phase2-teacher-saved-input-parity-summary-v1",
        "array_hashes": hashes,
        "summary_payload_sha256": hashlib.sha256(canonical_json_bytes(hashes)).hexdigest(),
    }
    return summary, arrays


def mps_gate(
    output: Path,
    weights: np.ndarray,
    network_seed: int,
    host: core.HostTape,
    host_wall_s: float,
    probe: np.ndarray,
) -> dict[str, object]:
    """Run saved-input CPU authority and conditional MPS parity/throughput."""

    sampler_results: dict[str, object] = {}
    overall_pass = True
    for sampler in ("rqmc", "iid"):
        input_dir = output / "saved_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        materialize_started = time.perf_counter()
        saved = saved_input_blocks(input_dir, weights.shape[1], network_seed, sampler)
        materialize_wall = time.perf_counter() - materialize_started
        cpu_started = time.perf_counter()
        cpu_pools, _ = pooled_saved_inputs(weights, saved, "cpu")
        cpu_summary, cpu_arrays = parity_summary(cpu_pools, weights, host, probe)
        atomic_json(output / f"PARITY_{sampler}_CPU.json", cpu_summary)
        cpu_wall = time.perf_counter() - cpu_started
        try:
            import torch  # type: ignore[import-not-found]

            available = bool(
                torch.backends.mps.is_built() and torch.backends.mps.is_available()
            )
        except Exception:
            available = False
        if not available:
            sampler_results[sampler] = {
                "status": "CPU_ONLY_MPS_UNAVAILABLE",
                "saved_input_materialization_wall_s": materialize_wall,
                "cpu_wall_s": cpu_wall,
                "cpu_summary": cpu_summary,
            }
            overall_pass = False
            continue

        # Three untimed warmups. Synchronization is also inside every snapshot.
        import torch  # type: ignore[import-not-found]

        for _ in range(3):
            core.accumulate_mps_stream(
                weights[:1],
                saved[0][:1],
                (256,),
                guard=enforce_resource_limit,
            )
        repeats = []
        repeat_arrays = []
        repeat_walls = []
        for _ in range(2):
            torch.mps.synchronize()
            started = time.perf_counter()
            mps_pools, _ = pooled_saved_inputs(weights, saved, "mps")
            mps_summary, mps_arrays = parity_summary(
                mps_pools, weights, host, probe
            )
            atomic_json(
                output / f"PARITY_{sampler}_MPS_repeat{len(repeats)}.json",
                mps_summary,
            )
            torch.mps.synchronize()
            repeat_walls.append(time.perf_counter() - started)
            repeats.append(mps_summary)
            repeat_arrays.append(mps_arrays)
        repeat_equal = repeats[0]["summary_payload_sha256"] == repeats[1]["summary_payload_sha256"]
        prediction_mse = float(
            np.mean(
                (cpu_arrays["prediction_A"] - repeat_arrays[0]["prediction_A"]) ** 2
            )
            + np.mean(
                (cpu_arrays["prediction_B"] - repeat_arrays[0]["prediction_B"]) ** 2
            )
        ) / 2.0
        diagonal_checks = []
        for label in (
            "pre_mean_A",
            "pre_mean_B",
            "pre_variance_A",
            "pre_variance_B",
            "pre_K3_A",
            "pre_K3_B",
            "pre_K4_A",
            "pre_K4_B",
        ):
            reference = cpu_arrays[label]
            observed = repeat_arrays[0][label]
            bank_partner = label[:-1] + ("B" if label.endswith("A") else "A")
            bank_se = np.abs(reference - cpu_arrays[bank_partner]) / 2.0
            discrepancy = np.abs(reference - observed)
            tolerance = np.maximum(1.0e-10, 0.1 * bank_se)
            absolute = float(np.max(discrepancy))
            mask = np.abs(reference) >= 1.0e-6
            relative = float(
                np.max(np.abs(reference[mask] - observed[mask]) / np.abs(reference[mask]))
            ) if np.any(mask) else 0.0
            diagonal_checks.append(
                {
                    "label": label,
                    "maximum_absolute": absolute,
                    "maximum_relative_above_1e_6": relative,
                    "maximum_bank_se_fraction_tolerance": float(np.max(tolerance)),
                    "within_bank_se_or_absolute_tolerance": bool(
                        np.all(discrepancy <= tolerance)
                    ),
                    "within_relative_tolerance": bool(
                        relative <= 1.0e-4
                    ),
                    "within_fixed_tolerance": bool(
                        np.all(discrepancy <= tolerance) and relative <= 1.0e-4
                    ),
                }
            )
        action_checks = []
        for label in sorted(name for name in cpu_arrays if name.startswith("action_") and name.endswith("_A")):
            partner = label[:-1] + "B"
            reference = cpu_arrays[label]
            observed = repeat_arrays[0][label]
            bank_se = np.abs(reference - cpu_arrays[partner]) / 2.0
            absolute_floor = 1.0e-9 if "K21" in label else 1.0e-10
            tolerance = np.maximum(absolute_floor, 0.1 * bank_se)
            discrepancy = np.abs(reference - observed)
            action_checks.append(
                {
                    "label": label,
                    "absolute_floor": absolute_floor,
                    "maximum_discrepancy": float(np.max(discrepancy)),
                    "maximum_tolerance": float(np.max(tolerance)),
                    "within_tolerance": bool(np.all(discrepancy <= tolerance)),
                }
            )
        invariant_checks = []
        for bank, pools_for_device in (("CPU", cpu_pools), ("MPS", mps_pools)):
            for bank_name, raw in pools_for_device.items():
                teacher = core.normalize_raw_tape(raw, weights)
                diagonal_error = float(
                    np.max(
                        np.abs(
                            np.diagonal(teacher.pre_K21, axis1=1, axis2=2)
                            - teacher.pre_K3
                        )
                    )
                )
                symmetry_error = float(
                    np.max(
                        np.abs(
                            teacher.post_covariance
                            - np.transpose(teacher.post_covariance, (0, 2, 1))
                        )
                    )
                )
                minimum_eigenvalue = float(
                    min(
                        np.linalg.eigvalsh(layer_covariance)[0]
                        for layer_covariance in teacher.post_covariance
                    )
                )
                invariant_checks.append(
                    {
                        "device": bank,
                        "bank": bank_name,
                        "K21_diagonal_K3_max_abs": diagonal_error,
                        "post_covariance_symmetry_max_abs": symmetry_error,
                        "post_covariance_minimum_eigenvalue": minimum_eigenvalue,
                        "finite": bool(
                            np.isfinite(teacher.pre_K21).all()
                            and np.isfinite(teacher.post_covariance).all()
                        ),
                        "pass": (
                            diagonal_error == 0.0
                            and symmetry_error <= 1.0e-12
                            and minimum_eigenvalue >= -1.0e-10
                        ),
                    }
                )
        # Timing includes fixed host and saved-input materialization/I/O in both paths.
        cpu_end_to_end = host_wall_s + materialize_wall + cpu_wall
        mps_end_to_end = host_wall_s + materialize_wall + min(repeat_walls)
        speedup = cpu_end_to_end / mps_end_to_end
        passed = bool(
            repeat_equal
            and prediction_mse <= 1.0e-10
            and all(item["within_fixed_tolerance"] for item in diagonal_checks)
            and all(item["within_tolerance"] for item in action_checks)
            and all(item["pass"] for item in invariant_checks)
            and speedup >= 1.5
        )
        overall_pass &= passed
        sampler_results[sampler] = {
            "status": "PASS_MPS" if passed else "CPU_ONLY_MPS_GATE_FAILED",
            "saved_input_materialization_wall_s": materialize_wall,
            "cpu_wall_s": cpu_wall,
            "mps_repeat_wall_s": repeat_walls,
            "cpu_end_to_end_wall_s": cpu_end_to_end,
            "mps_end_to_end_wall_s": mps_end_to_end,
            "end_to_end_speedup": speedup,
            "repeat_summary_hash_equal": repeat_equal,
            "prediction_disagreement_mse": prediction_mse,
            "diagonal_checks": diagonal_checks,
            "action_checks": action_checks,
            "invariant_checks": invariant_checks,
            "cpu_summary": cpu_summary,
            "mps_summaries": repeats,
        }
    result = {
        "schema": "phase2-teacher-pilot-mps-gate-v1",
        "status": "MPS" if overall_pass else "CPU_ONLY",
        "samplers": sampler_results,
        **ZERO_COUNTERS,
    }
    atomic_json(output / "MPS_GATE.json", result)
    return result


def synthetic_exact_errors(
    teacher_a: core.TeacherTape,
    teacher_b: core.TeacherTape,
    teacher_full: core.TeacherTape,
    host: core.HostTape,
) -> dict[str, object]:
    enforce_resource_limit("synthetic_exact_before_expected_arrays")
    exact = core.relu_standard_normal_exact()
    depth, width = teacher_full.pre_mean.shape
    expected_mean = np.full((depth, width), exact["mean"])
    expected_variance = np.full((depth, width), exact["variance"])
    expected_K3 = np.full((depth, width), exact["K3"])
    expected_K4 = np.full((depth, width), exact["K4"])
    expected_mean[0] = 0.0
    expected_variance[0] = 1.0
    expected_K3[0] = 0.0
    expected_K4[0] = 0.0
    expected_post_mean = np.full((depth, width), exact["mean"])
    expected_post_covariance = np.broadcast_to(
        exact["variance"] * np.eye(width), (depth, width, width)
    )
    expected_K21 = np.zeros((depth, width, width), dtype=np.float64)
    diagonal = np.arange(width)
    expected_K21[1:, diagonal, diagonal] = exact["K3"]
    enforce_resource_limit("synthetic_exact_after_expected_arrays")
    expected_teacher = core.TeacherTape(
        count=teacher_full.count,
        pre_mean=expected_mean,
        pre_covariance=teacher_full.pre_covariance,
        pre_variance=expected_variance,
        pre_K3=expected_K3,
        pre_K4=expected_K4,
        pre_K21=expected_K21,
        post_mean=expected_post_mean,
        post_covariance=expected_post_covariance,
    )
    observed_predictions = core.factorial_predictions(host, teacher_full)
    exact_predictions = core.factorial_predictions(host, expected_teacher)
    bank_prediction_error = core.bank_error(
        core.factorial_predictions(host, teacher_a),
        core.factorial_predictions(host, teacher_b),
    )
    enforce_resource_limit("synthetic_exact_after_all16_predictions")
    fields = {
        "pre_mean": (teacher_full.pre_mean, teacher_a.pre_mean, teacher_b.pre_mean, expected_mean),
        "pre_variance": (
            teacher_full.pre_variance,
            teacher_a.pre_variance,
            teacher_b.pre_variance,
            expected_variance,
        ),
        "pre_K3": (teacher_full.pre_K3, teacher_a.pre_K3, teacher_b.pre_K3, expected_K3),
        "pre_K4": (teacher_full.pre_K4, teacher_a.pre_K4, teacher_b.pre_K4, expected_K4),
        "post_mean": (
            teacher_full.post_mean,
            teacher_a.post_mean,
            teacher_b.post_mean,
            expected_post_mean,
        ),
        "post_covariance": (
            teacher_full.post_covariance,
            teacher_a.post_covariance,
            teacher_b.post_covariance,
            expected_post_covariance,
        ),
        "pre_K21": (
            teacher_full.pre_K21,
            teacher_a.pre_K21,
            teacher_b.pre_K21,
            expected_K21,
        ),
        "all16_predictions": (
            observed_predictions,
            core.factorial_predictions(host, teacher_a),
            core.factorial_predictions(host, teacher_b),
            exact_predictions,
        ),
    }
    checks: dict[str, object] = {}
    all_pass = True
    for label, (observed, bank_a, bank_b, expected) in fields.items():
        exact_mse = float(np.mean((observed - expected) ** 2))
        conservative_mse = float(np.mean((bank_a - bank_b) ** 2) / 2.0)
        limit = max(1.0e-12, 25.0 * conservative_mse)
        passed = exact_mse <= limit
        all_pass &= passed
        checks[label] = {
            "exact_mse": exact_mse,
            "bank_D2_over_2_mse": conservative_mse,
            "limit_25x_bank_or_1e_12": limit,
            "pass": passed,
        }
    return {
        "status": "PASS_EXACT_IDENTITY_CONTROL" if all_pass else "FAIL_EXACT_IDENTITY_CONTROL",
        "checks": checks,
        "all16_prediction_D2_over_4_mse": float(
            np.mean(bank_prediction_error.full_eight_proxy)
        ),
        "all16_prediction_D2_over_2_mse": float(
            np.mean(bank_prediction_error.conservative)
        ),
        "pass": all_pass,
    }


def run_case(
    run_root: Path,
    case: dict[str, object],
    probe: np.ndarray,
) -> dict[str, object]:
    case_output = run_root / str(case["label"])
    case_output.mkdir(parents=True, exist_ok=False)
    weights = core.canonical_f32le(case["weights"])
    host_started = time.perf_counter()
    enforce_resource_limit(f"before_host_{case['label']}")
    host = core.frozen_cp1792_host(weights, probe, guard=enforce_resource_limit)
    enforce_resource_limit(f"after_host_{case['label']}")
    host_wall = time.perf_counter() - host_started
    gate = mps_gate(
        case_output,
        weights,
        int(case["network_seed"]),
        host,
        host_wall,
        probe,
    )
    backend = "mps" if gate["status"] == "MPS" else "cpu"
    samplers: dict[str, object] = {}
    for sampler in ("rqmc", "iid"):
        bank_receipts = sampler_bank_products(
            case_output,
            weights,
            int(case["network_seed"]),
            sampler,
            backend,
        )
        derived = {}
        for level in core.TOTAL_LEVELS:
            derived[str(level)] = derive_and_seal_level(
                case_output, sampler, level, weights, host, probe
            )
        samplers[sampler] = {"banks": bank_receipts, "derived": derived}

    U = [
        float(samplers["rqmc"]["derived"][str(level)]["maximum_relevant_conservative_mse"])
        for level in core.TOTAL_LEVELS
    ]
    extrapolation = core.extrapolate_certificate(*U)
    signs_stable = True
    for label in samplers["rqmc"]["derived"][str(1 << 18)]["action_certificates"]:
        layer16 = samplers["rqmc"]["derived"][str(1 << 16)]["action_certificates"][label]["per_source_layer"]
        layer18 = samplers["rqmc"]["derived"][str(1 << 18)]["action_certificates"][label]["per_source_layer"]
        if len(layer16) != len(layer18):
            raise AssertionError("per-layer action certificate length changed")
        signs_stable &= all(
            left["aggregate_sign"] == right["aggregate_sign"]
            for left, right in zip(layer16, layer18, strict=True)
        )

    action_limits_pass = all(
        certificate["all_layers_within_limit"]
        for certificate in samplers["rqmc"]["derived"][str(1 << 18)][
            "action_certificates"
        ].values()
    )

    exact_errors = None
    if str(case["label"]).startswith("synthetic_identity"):
        raw_a = load_raw_tape(case_output / "rqmc" / "bank_A" / "N262144")
        raw_b = load_raw_tape(case_output / "rqmc" / "bank_B" / "N262144")
        teacher_a = core.normalize_raw_tape(
            raw_a, weights, guard=enforce_resource_limit
        )
        teacher_b = core.normalize_raw_tape(
            raw_b, weights, guard=enforce_resource_limit
        )
        teacher_full = core.normalize_raw_tape(
            core.merge_raw_tapes([raw_a, raw_b]),
            weights,
            guard=enforce_resource_limit,
        )
        exact_errors = synthetic_exact_errors(
            teacher_a, teacher_b, teacher_full, host
        )
    case_resource_gate_pass = bool(
        extrapolation["pass"]
        and signs_stable
        and action_limits_pass
        and gate["status"] == "MPS"
        and (exact_errors is None or exact_errors["pass"])
    )
    result = {
        "schema": "phase2-teacher-pilot-case-result-v1",
        "identity": case["identity"],
        "weights_sha256": weights_sha256(weights),
        "host_wall_s": host_wall,
        "selected_backend": backend,
        "mps_gate_status": gate["status"],
        "samplers": samplers,
        "rqmc_extrapolation": extrapolation,
        "last_two_level_action_signs_stable": signs_stable,
        "all_N262144_action_layers_within_limit": action_limits_pass,
        "synthetic_exact_errors": exact_errors,
        "full_extraction_resource_gate_pass": case_resource_gate_pass,
        "resource_highwater": dict(RESOURCE_HIGHWATER),
        **ZERO_COUNTERS,
    }
    result["result_sha256"] = atomic_json(case_output / "CASE_RESULT.json", result)
    return result


def execute() -> dict[str, object]:
    # Everything above this claim is zero-row, source-only validation.
    enforce_resource_limit("initial_preclaim")
    release, release_sha, bindings = validate_release()
    interface = load_interface()
    validate_external_source_hashes(interface)
    probe = load_probe(interface)
    claim_path, claim_sha = write_one_use_claim(release, release_sha, bindings)
    # Authority is now consumed.  No retry is legal after any subsequent failure.
    cases = load_cases_after_claim(interface)
    output = Path(str(release["output_path"])).resolve()
    if output.parent != RUNS_DIR.resolve():
        raise PermissionError("fixed run parent mismatch")
    if not RUNS_DIR.is_dir() or not (RUNS_DIR / "README.md").is_file():
        raise PermissionError("frozen runs parent missing after claim; authority consumed")
    enforce_resource_limit("before_output_child_creation")
    output.mkdir(parents=False, exist_ok=False)
    free_bytes = shutil.disk_usage(output).free
    resource_plan = json.loads(
        (HERE / "PILOT_RESOURCE_ESTIMATE.json").read_text(encoding="utf-8")
    )
    if free_bytes < int(resource_plan["persistent_bytes"]["required_free_workspace"]):
        raise RuntimeError("insufficient workspace after one-use claim; authority consumed")

    results = []
    for case in cases:
        results.append(run_case(output, case, probe))
        enforce_resource_limit(
            f"after_case_{case['label']}",
            limit_bytes=int(resource_plan["peak_memory"]["abort_threshold_bytes"]),
        )
    full_extraction_pass = all(
        bool(case_result["full_extraction_resource_gate_pass"])
        for case_result in results
    )
    final = {
        "schema": "phase2-teacher-target-free-pilot-result-v1",
        "status": "TARGET_FREE_PILOT_COMPLETE_NO_SCIENTIFIC_SELECTION",
        "release_sha256": release_sha,
        "claim_path": str(claim_path),
        "claim_sha256": claim_sha,
        "bindings": bindings,
        "cases": results,
        "full_extraction_resource_decision": (
            "GO_FULL_EXTRACTION_RESOURCE_ONLY__SEPARATE_AUTHORITY_REQUIRED"
            if full_extraction_pass
            else "STOP_CERTIFICATE_OR_RESOURCE_IMPLAUSIBLE"
        ),
        "full_extraction_resource_gate_pass": full_extraction_pass,
        "full_extraction_still_requires_separate_authority": True,
        "dataset_rows_decoded": 2,
        "decoded_columns": interface["target_blind_source"]["allowed_value_columns"],
        "resource_highwater": dict(RESOURCE_HIGHWATER),
        **ZERO_COUNTERS,
    }
    final["result_sha256"] = atomic_json(output / "PILOT_RESULT.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.preflight:
        raise SystemExit("choose exactly one mode")
    result = execute() if args.execute else preflight()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
