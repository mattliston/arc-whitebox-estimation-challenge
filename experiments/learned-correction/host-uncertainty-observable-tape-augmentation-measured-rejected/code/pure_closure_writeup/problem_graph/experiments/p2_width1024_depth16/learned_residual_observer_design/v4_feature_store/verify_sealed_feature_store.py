#!/usr/bin/env python3
"""Target-blind verifier for a future candidate or final feature store.

Default invocation is inert.  Candidate verification is only meaningful after
the generation claim exists; it never imports a full-data or role-target
reader.  `--verify-edges` is mandatory for an artifact-audit PASS because it
reconstructs and hashes every row's edge tape against the frozen worker hash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contract import (
    CLAIM_PATH,
    OUTPUT_PATH,
    PER_PROCESS_LIMIT_BYTES,
    ROLE_BINDINGS,
    ROLE_ORDER,
    SOURCE_MANIFEST_PATH,
    STAGING_PATH,
    FROZEN_BINDINGS,
    HERE,
    array_payload_sha256,
    row_relative_path,
    sha256_file,
    validate_candidate_metadata,
    validate_row_directory,
)


def preflight() -> dict[str, Any]:
    return {
        "status": "PASS_TARGET_FREE_ARTIFACT_VERIFIER_PREFLIGHT__NO_STORE_OPENED",
        "claim_present": CLAIM_PATH.exists(),
        "candidate_present": (STAGING_PATH / "candidate_manifest.json").exists(),
        "final_store_present": OUTPUT_PATH.exists(),
        "feature_rows_opened_or_hashed": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "scores": 0,
    }


def _validate_resources(root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    samples_path = root / "resource_samples.jsonl"
    if sha256_file(samples_path) != candidate["resource_samples_sha256"]:
        raise AssertionError("resource-sample log hash changed")
    samples = []
    for line_number, line in enumerate(samples_path.read_text(encoding="utf-8").splitlines(), 1):
        sample = json.loads(line)
        physical = int(sample["physical_memory_bytes"])
        tree = int(sample["process_tree_rss_bytes"])
        if tree * 4 >= physical:
            raise AssertionError(f"system-memory boundary violated in sample {line_number}")
        worker_values = [int(value) for value in sample["worker_subtree_rss_bytes"].values()]
        if any(value >= PER_PROCESS_LIMIT_BYTES for value in worker_values):
            raise AssertionError(f"worker-memory boundary violated in sample {line_number}")
        if int(sample["running_workers"]) > 2:
            raise AssertionError("bounded-parallelism contract violated")
        samples.append(sample)
    if not samples:
        raise AssertionError("resource-sample log is empty")
    receipt_path = root / "GENERATION_RECEIPT.json"
    if sha256_file(receipt_path) != candidate["execution_receipt_sha256"]:
        raise AssertionError("generation receipt hash changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE_TARGET_BLIND_FEATURE_ROWS__HOLD_FINAL_SEAL_FOR_INDEPENDENT_AUDIT":
        raise AssertionError("generation receipt status changed")
    if receipt.get("total_validated_rows") != 776:
        raise AssertionError("generation receipt row count changed")
    if any(receipt.get(key) != 0 for key in ("target_values_decoded", "fits", "scores")):
        raise AssertionError("generation receipt claims target/fit/score activity")
    return {
        "samples": len(samples),
        "peak_process_tree_rss_bytes": max(int(row["process_tree_rss_bytes"]) for row in samples),
        "peak_worker_subtree_rss_bytes": max(
            (
                max((int(value) for value in row["worker_subtree_rss_bytes"].values()), default=0)
                for row in samples
            ),
            default=0,
        ),
        "physical_memory_bytes": int(samples[0]["physical_memory_bytes"]),
    }


def verify_candidate(candidate_path: Path, *, verify_edges: bool) -> dict[str, Any]:
    if not CLAIM_PATH.is_file() or CLAIM_PATH.is_symlink():
        raise PermissionError("candidate verification requires the consumed generation claim")
    if candidate_path.resolve() != (STAGING_PATH / "candidate_manifest.json").resolve():
        raise PermissionError("verifier accepts only the exact staging candidate")
    candidate = validate_candidate_metadata(candidate_path)
    if candidate.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise AssertionError("candidate binds another source freeze")
    if candidate.get("provider_source_sha256") != sha256_file(HERE / "sealed_feature_store_provider.py"):
        raise AssertionError("candidate binds another provider source")
    exact_bindings = {
        "target_blind_full_source_sha256": FROZEN_BINDINGS["target_blind_full_reader"]["sha256"],
        "full_cache_audit_sha256": FROZEN_BINDINGS["full_cache_audit"]["sha256"],
        "roster_sha256": FROZEN_BINDINGS["full_roster"]["sha256"],
        "v4_implementation_manifest_sha256": FROZEN_BINDINGS["v4_implementation_manifest"]["sha256"],
        "v4_launch_audit_sha256": FROZEN_BINDINGS["v4_independent_launch_audit"]["sha256"],
        "custody_post_seal_audit_sha256": FROZEN_BINDINGS["custody_post_seal_audit"]["sha256"],
    }
    for key, value in exact_bindings.items():
        if candidate.get(key) != value:
            raise AssertionError(f"candidate frozen binding changed: {key}")
    resources = _validate_resources(candidate_path.parent, candidate)

    provider = None
    if verify_edges:
        import sealed_feature_store_provider as provider

    rows = 0
    bytes_hashed = 0
    edge_tapes_reconstructed = 0
    for role in ROLE_ORDER:
        for rank, entry in enumerate(candidate["roles"][role]["row_manifests"]):
            row_dir = candidate_path.parent / row_relative_path(role, rank)
            row_manifest_path = row_dir / "row_manifest.json"
            if sha256_file(row_manifest_path) != entry["sha256"]:
                raise AssertionError("candidate row-manifest digest changed")
            row_manifest = validate_row_directory(row_dir, role=role, role_rank=rank)
            bytes_hashed += row_manifest_path.stat().st_size
            bytes_hashed += sum(int(record["bytes"]) for record in row_manifest["files"].values())
            if verify_edges:
                import numpy as np

                weights = np.load(row_dir / "weights.npy", mmap_mode="r", allow_pickle=False)
                edges = provider.SealedFeatureStore._edge_matrices(weights)
                edge_hash = array_payload_sha256(edges)
                for cap in (1536, 1792):
                    if row_manifest["hosts"][str(cap)]["edges_sha256"] != edge_hash:
                        raise AssertionError("provider edge reconstruction differs from frozen host")
                edge_tapes_reconstructed += 1
                del edges, weights
            rows += 1
    if rows != 776:
        raise AssertionError("artifact verifier did not cover exactly 776 rows")
    status = (
        "PASS_PHASE2_OBSERVER_V4_FEATURE_STORE_ARTIFACT_AUDIT"
        if verify_edges
        else "HOLD_EDGE_RECONSTRUCTION_NOT_VERIFIED_FOR_ALL_ROWS"
    )
    return {
        "status": status,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "rows_verified": rows,
        "bytes_hashed": bytes_hashed,
        "edge_tapes_reconstructed_and_hash_checked": edge_tapes_reconstructed,
        "resources": resources,
        "roles": {role: ROLE_BINDINGS[role]["rows"] for role in ROLE_ORDER},
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "this_receipt_is_not_finalization_authority": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--verify-edges", action="store_true")
    args = parser.parse_args()
    if args.candidate is None:
        report = preflight()
    else:
        report = verify_candidate(args.candidate, verify_edges=args.verify_edges)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
