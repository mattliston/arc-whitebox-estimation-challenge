#!/usr/bin/env python3
"""Claim-gated V7 worker for the frozen V5 Layer-3 pilot science."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from runtime_contract import (
    attest_exact_environment,
    canonical_json_bytes,
    load_and_verify_source_manifest,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
P2 = HERE.parents[2]
RELEASE_PATH = HERE / "ROOT_RELEASE.json"
REHEARSAL_PATH = HERE / "SYNTHETIC_REHEARSAL.json"
LOADER_PATH = P2 / "data_gate/target_blind_mini_v2.py"
HOST_PATH = P2 / "cp3_cap1792_k3_readout/estimator.py"
LOADER_SHA256 = "6778a3216377a4a7e588df71a08fb0e182dc7a14d42ab393c2ac80200cccf0e1"
HOST_SHA256 = "035e3bd2fd08d3358f5424fc67f10281e5559e65fa9c6d6dce97791108b176dc"
CHECKPOINT_ROOT = P2 / (
    "cp3_full_birth_cap_ladder/mini100_seal_v3/receipts/bundles/"
    "mini100-cp3caps-v3-3eb980d14b36e443614427f3c1d781763cca1c38f88a2231b5559bdac32fe4c0/"
    "checkpoints"
)
CHECKPOINT_BINDINGS = (
    {
        "path": CHECKPOINT_ROOT / "row-000.json",
        "checkpoint_sha256": "0ce9844974bdf28ad627c717be3d4c6ff60263a29b011abf0a7ac4df145c3940",
        "ordinal": 0,
        "mlp_id": 0,
        "mlp_name": "logan-fitzgerald",
        "mlp_seed": 6319981554997072999,
        "weights_shape": (16, 1024, 1024),
        "weights_sha256": "d524e662eca3d117e9ba164fc3ae51a790dd675c585914a895ec69725fd880d0",
    },
    {
        "path": CHECKPOINT_ROOT / "row-001.json",
        "checkpoint_sha256": "4747f2b356e65b6b1d440ea47a1042575b05c4c056f9618005ae6ab248541bee",
        "ordinal": 1,
        "mlp_id": 1,
        "mlp_name": "william-graves",
        "mlp_seed": 2392109381927064353,
        "weights_shape": (16, 1024, 1024),
        "weights_sha256": "41f9a2a1d715abfc9f0ca535b515fb0aa0ae0edda0f022b3f76d61abfc69ae54",
    },
)
ZERO_SCIENTIFIC_COUNTERS = {
    "target_values_decoded": 0,
    "target_metrics_computed": 0,
    "fits": 0,
    "scores": 0,
    "variants": 0,
    "sweeps": 0,
    "submissions": 0,
    "graph_edits": 0,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_claimed_release(
    claimed_release_sha256: str, invocation_tombstone_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    if not RELEASE_PATH.is_file() or sha256_file(RELEASE_PATH) != claimed_release_sha256:
        raise PermissionError("claimed root release bytes changed")
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    manifest = load_and_verify_source_manifest(str(release["source_manifest_sha256"]))
    tombstone = Path(str(release["invocation_tombstone_path"])).resolve()
    if not tombstone.is_file() or sha256_file(tombstone) != invocation_tombstone_sha256:
        raise PermissionError("invocation tombstone missing or changed")
    tombstone_value = json.loads(tombstone.read_text(encoding="utf-8"))
    if (
        tombstone_value.get("status")
        != "ONE_INVOCATION_IRREVERSIBLY_CONSUMED_BEFORE_WORKER_CREATION"
        or tombstone_value.get("release_sha256") != claimed_release_sha256
        or tombstone_value.get("source_manifest_sha256") != manifest["manifest_sha256"]
        or tombstone_value.get("future_invocations_remaining") != 0
        or tombstone_value.get("retry_authorized") is not False
    ):
        raise PermissionError("invocation tombstone contract mismatch")
    for label in ("data_claim_path", "staging_path", "result_path"):
        if Path(str(release[label])).exists():
            raise FileExistsError(f"V7 path already exists: {label}")
    if release.get("ordinals") != [0, 1]:
        raise PermissionError("V7 ordinal scope changed")
    return release, manifest


def run_preclaim_gates(manifest: dict[str, object]) -> dict[str, object]:
    """No physical loader module is imported by this gate."""
    environment = attest_exact_environment()
    if not REHEARSAL_PATH.is_file():
        raise RuntimeError("frozen synthetic rehearsal receipt absent")
    rehearsal = json.loads(REHEARSAL_PATH.read_text(encoding="utf-8"))
    if (
        rehearsal.get("status")
        != "PASS_V7_EXACT_ENV_PREBOUND_ROWS_AND_SMALL_SCIENCE_REHEARSAL"
        or rehearsal.get("execution_authority") is not False
        or rehearsal.get("release_present") is not False
        or any(rehearsal["access_counters"].get(key) != 0 for key in (
            "physical_snapshot_calls", "physical_parquet_files_opened",
            "physical_dataset_iterator_calls", "physical_dataset_rows_decoded",
            "teacher_artifacts_generated", "target_values_decoded",
            "target_metrics_computed", "fits", "scores", "variants", "sweeps",
            "submissions", "graph_edits",
        ))
    ):
        raise RuntimeError("synthetic rehearsal receipt inadmissible")
    import science_core as core

    observed_small = core.synthetic_self_test()
    if observed_small != rehearsal.get("frozen_v5_science_runtime"):
        raise RuntimeError("small science runtime replay changed")
    if sha256_file(LOADER_PATH) != LOADER_SHA256 or sha256_file(HOST_PATH) != HOST_SHA256:
        raise RuntimeError("physical loader or frozen host source changed")
    return {
        "environment": environment,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "synthetic_rehearsal_sha256": sha256_file(REHEARSAL_PATH),
        "small_science_runtime": observed_small,
        "physical_loader_module_imported": False,
        "physical_snapshot_calls": 0,
        "physical_parquet_files_opened": 0,
        "physical_dataset_iterator_calls": 0,
        "physical_dataset_rows_decoded": 0,
        **ZERO_SCIENTIFIC_COUNTERS,
    }


def durable_data_claim(
    release: dict[str, object], release_sha: str, tombstone_sha: str,
    preclaim: dict[str, object],
) -> tuple[Path, str]:
    path = Path(str(release["data_claim_path"])).resolve()
    if path.parent != (HERE / "receipts").resolve():
        raise PermissionError("data claim path escaped frozen parent")
    claim = {
        "schema": "phase2-layer3-attribution-v7-data-claim-v1",
        "status": "DATA_CLAIM_CONSUMED_AFTER_SYNTHETIC_GATES_BEFORE_PHYSICAL_LOADER_IMPORT",
        "release_id": release["release_id"],
        "release_sha256": release_sha,
        "invocation_tombstone_sha256": tombstone_sha,
        "source_manifest_sha256": release["source_manifest_sha256"],
        "scope": release["scope"],
        "ordinals": [0, 1],
        "preclaim_gates": preclaim,
        "physical_loader_module_imported_before_claim": False,
        "physical_dataset_iterator_calls_before_claim": 0,
        "physical_dataset_rows_decoded_before_claim": 0,
        "future_invocations_remaining": 0,
        "retry_authorized": False,
        **ZERO_SCIENTIFIC_COUNTERS,
    }
    payload = canonical_json_bytes(claim)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short V7 data-claim write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return path, hashlib.sha256(payload).hexdigest()


def import_physical_loader_after_claim():
    if sha256_file(LOADER_PATH) != LOADER_SHA256:
        raise RuntimeError("target-blind loader changed after claim")
    loader = load_module("p2_l3_v7_physical_loader_after_claim", LOADER_PATH)
    if not callable(getattr(loader, "iter_target_blind_mini", None)):
        raise RuntimeError("target-blind iterator missing")
    return loader


def load_exact_two_rows_after_claim(loader) -> list[object]:
    iterator = loader.iter_target_blind_mini()
    rows = [next(iterator), next(iterator)]
    allowed = ("mlp_id", "mlp_name", "mlp_seed", "weights")
    for ordinal, row in enumerate(rows):
        if (
            row.ordinal != ordinal
            or tuple(row.decoded_columns) != allowed
            or row.weights.shape != (16, 1024, 1024)
            or str(row.weights.dtype) != "float32"
            or row.weights.flags.writeable
            or not row.weights.flags.c_contiguous
        ):
            raise AssertionError("physical target-blind row contract changed")
    return rows


def attest_decoded_row_against_binding(
    row: object, binding: dict[str, object], checkpoint: dict[str, object]
) -> dict[str, object]:
    """Bind decoded identity and physical weight bytes before any science."""
    allowed = ("mlp_id", "mlp_name", "mlp_seed", "weights")
    expected_identity = {
        "mlp_id": binding["mlp_id"],
        "mlp_name": binding["mlp_name"],
        "mlp_seed": binding["mlp_seed"],
    }
    observed_identity = {
        "mlp_id": int(row.mlp_id),
        "mlp_name": str(row.mlp_name),
        "mlp_seed": int(row.mlp_seed),
    }
    observed_weights_sha256 = hashlib.sha256(
        memoryview(row.weights).cast("B")
    ).hexdigest()
    if (
        int(row.ordinal) != binding["ordinal"]
        or tuple(row.decoded_columns) != allowed
        or observed_identity != expected_identity
        or tuple(row.weights.shape) != tuple(binding["weights_shape"])
        or str(row.weights.dtype) != "float32"
        or row.weights.flags.writeable
        or not row.weights.flags.c_contiguous
        or observed_weights_sha256 != binding["weights_sha256"]
        or checkpoint.get("ordinal") != binding["ordinal"]
        or checkpoint.get("identity") != expected_identity
        or checkpoint.get("weights") != {
            "dtype": "float32-le",
            "shape": list(binding["weights_shape"]),
            "sha256": binding["weights_sha256"],
        }
        or checkpoint.get("source", {}).get("decoded_columns") != list(allowed)
    ):
        raise AssertionError("decoded row differs from prebound CP16384 checkpoint")
    return {
        "status": "PASS_V7_PREBOUND_IDENTITY_AND_WEIGHT_BYTES",
        "ordinal": binding["ordinal"],
        **expected_identity,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "weights_sha256": observed_weights_sha256,
    }


def verify_decoded_rows_against_prebound_checkpoints(
    rows: list[object],
) -> list[dict[str, object]]:
    if len(rows) != len(CHECKPOINT_BINDINGS):
        raise AssertionError("V7 requires exactly two decoded rows")
    attestations = []
    for row, binding in zip(rows, CHECKPOINT_BINDINGS, strict=True):
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["checkpoint_sha256"]:
            raise RuntimeError("prebound CP16384 checkpoint bytes changed")
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        attestations.append(
            attest_decoded_row_against_binding(row, binding, checkpoint)
        )
    return attestations


def frozen_host_parity(host, weights, core) -> dict[str, object]:
    if sha256_file(HOST_PATH) != HOST_SHA256:
        raise RuntimeError("frozen host changed before parity")
    module = load_module("p2_l3_v7_frozen_host_after_claim", HOST_PATH)
    fake = SimpleNamespace(width=1024, depth=3, weights=weights[:3])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = core.canonical_f32(module.Estimator().predict(fake, 0))
    expected = core.canonical_f32(host.reported_rows)
    error = float(abs(observed.astype("float64") - expected).max())
    if error > 3.0e-7:
        raise AssertionError(f"frozen host parity failed: {error}")
    return {
        "status": "PASS_FROZEN_CP1792_HOST_PARITY_THROUGH_L3",
        "max_abs": error,
        "observed_sha256": core.hash_array(observed),
        "expected_sha256": core.hash_array(expected),
    }


def run_science(rows: list[object], preclaim: dict[str, object]) -> dict[str, object]:
    import science_core as core

    probe = core.walsh_probe()
    raw: dict[int, dict[int, dict[str, object]]] = {}
    networks = []
    for ordinal, row in enumerate(rows):
        weights = core.canonical_f32(row.weights)
        host = core.build_host_state(weights, probe)
        parity = frozen_host_parity(host, weights, core)
        streams_by_prefix = {prefix: [] for prefix in core.PRODUCTION_PREFIXES}
        for replicate in range(16):
            snapshots = core.stream_prefix_statistics(
                weights[:3], int(row.mlp_seed), replicate, probe,
                core.PRODUCTION_PREFIXES,
            )
            for prefix in core.PRODUCTION_PREFIXES:
                streams_by_prefix[prefix].append(snapshots[prefix])
        raw[ordinal] = {
            prefix: core.quartet_metrics(host, streams_by_prefix[prefix], weights[2])
            for prefix in core.PRODUCTION_PREFIXES
        }
        networks.append({
            "ordinal": ordinal,
            "mlp_id": int(row.mlp_id),
            "mlp_name": str(row.mlp_name),
            "mlp_seed": int(row.mlp_seed),
            "weights_sha256": core.hash_array(weights),
            "host_reported_sha256": core.hash_array(host.reported_rows),
            "host_parity": parity,
            "exact_controls": host.exact_controls,
            "prefix_diagnostics": raw[ordinal],
        })
    analysis = core.analyze_two_ordinals(raw)
    return {
        "schema": "phase2-layer3-attribution-v7-result-v1",
        "status": "COMPLETE_TWO_ORDINAL_TARGET_BLIND_LAYER3_ATTRIBUTION",
        "science_binding": "V5_BYTE_EXACT_V7_PREBOUND_ROWS_NO_FIT_NO_SCORE",
        "ordinals": [0, 1],
        "networks": networks,
        "analysis": analysis,
        "preclaim_gate": preclaim,
        "physical_dataset_iterator_calls": 1,
        "physical_dataset_rows_decoded": 2,
        "teacher_artifacts_generated": 1,
        **ZERO_SCIENTIFIC_COUNTERS,
    }


def publish_atomic_result(release: dict[str, object], result: dict[str, object]) -> str:
    staging = Path(str(release["staging_path"])).resolve()
    final = Path(str(release["result_path"])).resolve()
    if staging.parent != (HERE / "runs").resolve() or final.parent != staging.parent:
        raise PermissionError("result paths escaped frozen parent")
    if staging.exists() or final.exists():
        raise FileExistsError("staging or final result already exists")
    payload = canonical_json_bytes(result)
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short V7 result staging write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.link(staging, final)
    parent = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
        os.unlink(staging)
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(payload).hexdigest()


def execute(claimed_release_sha256: str, invocation_tombstone_sha256: str) -> dict[str, object]:
    release, manifest = validate_claimed_release(
        claimed_release_sha256, invocation_tombstone_sha256
    )
    preclaim = run_preclaim_gates(manifest)
    claim_path, claim_sha = durable_data_claim(
        release, claimed_release_sha256, invocation_tombstone_sha256, preclaim
    )
    loader = import_physical_loader_after_claim()
    rows = load_exact_two_rows_after_claim(loader)
    row_attestations = verify_decoded_rows_against_prebound_checkpoints(rows)
    result = run_science(rows, preclaim)
    result.update({
        "release_id": release["release_id"],
        "release_sha256": claimed_release_sha256,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "invocation_tombstone_sha256": invocation_tombstone_sha256,
        "data_claim_sha256": claim_sha,
        "decoded_row_attestations": row_attestations,
    })
    result_sha = publish_atomic_result(release, result)
    if not claim_path.is_file():
        raise RuntimeError("data claim vanished after publication")
    return {
        "status": "V7_RESULT_ATOMICALLY_PUBLISHED",
        "result_path": release["result_path"],
        "result_sha256": result_sha,
        "data_claim_sha256": claim_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claimed-release-sha256", required=True)
    parser.add_argument("--invocation-tombstone-sha256", required=True)
    args = parser.parse_args()
    result = execute(
        args.claimed_release_sha256, args.invocation_tombstone_sha256
    )
    sys.stdout.buffer.write(canonical_json_bytes(result))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
