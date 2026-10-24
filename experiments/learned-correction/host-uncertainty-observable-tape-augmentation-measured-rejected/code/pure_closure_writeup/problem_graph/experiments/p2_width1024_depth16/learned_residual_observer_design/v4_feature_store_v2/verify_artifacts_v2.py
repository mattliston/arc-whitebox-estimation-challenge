#!/usr/bin/env python3
"""Target-blind all-row/resource verifier for a future V2 candidate store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contract_v2 import (
    CLAIM_PATH, FROZEN_BINDINGS, HERE, OUTPUT_PATH, PER_PROCESS_LIMIT_BYTES,
    ROLE_BINDINGS, ROLE_ORDER, SOURCE_MANIFEST_PATH, STAGING_PATH,
    V1_BINDINGS, array_payload_sha256, row_relative_path, sha256_file,
    validate_row_directory,
)


def preflight() -> dict[str, Any]:
    return {
        "status": "PASS_TARGET_FREE_V2_ARTIFACT_VERIFIER_PREFLIGHT__NO_STORE_OPENED",
        "claim_present": CLAIM_PATH.exists(),
        "candidate_present": (STAGING_PATH / "candidate_manifest.json").exists(),
        "final_store_present": OUTPUT_PATH.exists(),
        "feature_rows_opened_or_hashed": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def _metadata(path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("schema") != "phase2-observer-v4-target-blind-feature-store-v2-candidate-v1":
        raise AssertionError("wrong V2 candidate schema")
    if candidate.get("status") != "READY_FOR_V2_INDEPENDENT_ARTIFACT_AUDIT__NOT_FINAL_SEALED":
        raise AssertionError("wrong V2 candidate status")
    if candidate.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise AssertionError("V2 candidate binds another source freeze")
    if candidate.get("provider_source_sha256") != sha256_file(HERE / "sealed_feature_store_provider_v2.py"):
        raise AssertionError("V2 candidate binds another provider")
    exact = {
        "host_caps": [1536, 1792],
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v1_hold_audit_sha256": V1_BINDINGS["independent_source_audit_hold"]["sha256"],
        "raw_full_dataset_target_projection": "forbidden",
        "target_values_decoded": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "fits": 0,
        "scores": 0,
    }
    for key, value in exact.items():
        if candidate.get(key) != value:
            raise AssertionError(f"V2 candidate binding changed: {key}")
    return candidate


def _resources(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    logs = candidate.get("resource_sample_logs")
    if not isinstance(logs, list) or not logs:
        raise AssertionError("V2 candidate lacks invocation resource logs")
    maximum_gap = 0.0
    total_samples = 0
    seen_paths: set[str] = set()
    for log_index, record in enumerate(logs):
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise AssertionError("V2 resource log address is malformed or duplicated")
        seen_paths.add(relative)
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
            raise AssertionError("V2 resource log escapes staging or is not regular")
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            raise AssertionError("V2 resource log hash/size changed")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise AssertionError("V2 resource log is empty")
        previous = None
        for sample_index, row in enumerate(rows):
            if int(row["aggregate_rss_bytes"]) * 4 >= int(row["physical_memory_bytes"]):
                raise AssertionError(
                    f"V2 aggregate boundary violated at log {log_index} sample {sample_index}"
                )
            if any(int(value) >= PER_PROCESS_LIMIT_BYTES for value in row["group_rss_bytes"].values()):
                raise AssertionError(
                    f"V2 group boundary violated at log {log_index} sample {sample_index}"
                )
            now = float(row["monotonic_seconds"])
            if previous is not None:
                maximum_gap = max(maximum_gap, now - previous)
            previous = now
        total_samples += len(rows)
    if maximum_gap > 0.20:
        raise AssertionError("V2 continuous watchdog gap exceeded 0.20 seconds")
    receipt_path = root / "GENERATION_RECEIPT_V2.json"
    if sha256_file(receipt_path) != candidate["generation_receipt_sha256"]:
        raise AssertionError("V2 generation receipt hash changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("decoded_full_rows") != 1000 or receipt.get("selected_rows") != 776:
        raise AssertionError("V2 generation receipt row counts changed")
    if not receipt.get("all_host_groups_quiescent") or not receipt["cursor_group_quiescence"]["three_empty_polls"]:
        raise AssertionError("V2 generation receipt lacks group quiescence")
    if float(receipt["watchdog"]["maximum_sample_gap_seconds"]) > 0.20:
        raise AssertionError("V2 receipt watchdog gap exceeds bound")
    return {
        "invocation_logs": len(logs),
        "samples": total_samples,
        "maximum_intra_invocation_sample_gap_seconds": maximum_gap,
    }


def verify(candidate_path: Path, verify_edges: bool) -> dict[str, Any]:
    if not CLAIM_PATH.is_file() or CLAIM_PATH.is_symlink():
        raise PermissionError("V2 artifact verification requires consumed claim")
    if candidate_path.resolve() != (STAGING_PATH / "candidate_manifest.json").resolve():
        raise PermissionError("V2 verifier accepts only exact staging candidate")
    candidate = _metadata(candidate_path)
    resources = _resources(candidate_path.parent, candidate)
    provider = None
    if verify_edges:
        import sealed_feature_store_provider_v2 as provider
    rows = 0; bytes_hashed = 0; edges_verified = 0
    for role in ROLE_ORDER:
        entries = candidate["roles"][role]["row_manifests"]
        if len(entries) != ROLE_BINDINGS[role]["rows"]:
            raise AssertionError("V2 candidate role membership changed")
        for rank, entry in enumerate(entries):
            row_dir = candidate_path.parent / row_relative_path(role, rank)
            if entry.get("role_rank") != rank or entry.get("path") != str(row_relative_path(role, rank) / "row_manifest.json"):
                raise AssertionError("V2 candidate row address changed")
            if sha256_file(row_dir / "row_manifest.json") != entry["sha256"]:
                raise AssertionError("V2 row manifest hash changed")
            manifest = validate_row_directory(row_dir, role=role, role_rank=rank)
            bytes_hashed += (row_dir / "row_manifest.json").stat().st_size
            bytes_hashed += sum(int(record["bytes"]) for record in manifest["files"].values())
            if verify_edges:
                import numpy as np
                weights = np.load(row_dir / "weights.npy", mmap_mode="r", allow_pickle=False)
                edges = provider.SealedFeatureStore._edge_matrices(weights)
                digest = array_payload_sha256(edges)
                if any(manifest["hosts"][str(cap)]["edges_sha256"] != digest for cap in (1536, 1792)):
                    raise AssertionError("V2 reconstructed edge hash changed")
                edges_verified += 1; del weights, edges
            rows += 1
    if rows != 776:
        raise AssertionError("V2 artifact verifier did not cover 776 rows")
    return {
        "status": "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_V2_ARTIFACT_AUDIT" if verify_edges else "HOLD_V2_ALL_ROW_EDGE_RECONSTRUCTION_NOT_RUN",
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "rows_verified": rows, "bytes_hashed": bytes_hashed,
        "edge_tapes_reconstructed_and_hash_checked": edges_verified,
        "resources": resources, "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0, "fits": 0, "scores": 0,
        "this_receipt_is_not_finalization_authority": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", type=Path); parser.add_argument("--verify-edges", action="store_true"); args = parser.parse_args()
    print(json.dumps(preflight() if args.candidate is None else verify(args.candidate, args.verify_edges), indent=2, sort_keys=True))
