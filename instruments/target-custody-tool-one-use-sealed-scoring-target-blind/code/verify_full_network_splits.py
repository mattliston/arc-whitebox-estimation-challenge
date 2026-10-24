#!/usr/bin/env python3
"""Verify the frozen full-network roster without reading the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_full_network_splits import DOMAIN, ROLE_SEGMENTS, SPLIT_SIZES, assignment_hash
from full_data_gate import AUTHORITY, FULL, HERE, WEIGHT_COLUMNS


DEFAULT_MANIFEST = HERE / "full_network_splits.json"


def _identity(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["global_hash_rank"],
        row["mlp_id"],
        row["mlp_seed"],
        row["weights_sha256"],
        row["primary_split"],
        row["observer_role"],
    )


def verify(path: Path) -> dict[str, object]:
    if assignment_hash(0) != (
        "22ba5941825921f557dc0abb7efc979b168e63300302421f7a2d9eaabffb92ac"
    ):
        raise AssertionError("teacher split-law test vector failed")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload["schema"] != "whest-phase2-network-split-v1":
        raise AssertionError("wrong full network split schema")
    if payload["commit"] != AUTHORITY["pinned_commit_sha"]:
        raise AssertionError("split is bound to the wrong commit")
    if payload["full_shard_manifest_sha256"] != FULL["manifest_sha256"]:
        raise AssertionError("split is bound to the wrong shard manifest")
    if bytes.fromhex(payload["assignment"]["domain_hex"]) != DOMAIN:
        raise AssertionError("split domain changed")
    if payload["assignment"]["formula"] != (
        "SHA256('p2-teacher-v1|v2-phase2|' + decimal_mlp_seed)"
    ):
        raise AssertionError("teacher assignment formula changed")
    if payload["assignment"]["assignment_inputs"] != ["mlp_seed"]:
        raise AssertionError("non-seed value entered teacher assignment")
    if payload["assignment"]["attestations_not_assignment_inputs"] != [
        "mlp_id",
        "weights_sha256",
    ]:
        raise AssertionError("teacher attestation/assignment boundary changed")
    if payload["decoded_value_columns"] != list(WEIGHT_COLUMNS):
        raise AssertionError("split builder did not use the exact target-blind projection")
    if payload["target_values_decoded"] != 0:
        raise AssertionError("split manifest claims target access")

    ranked: list[dict[str, object]] = []
    for split_name, expected_size in SPLIT_SIZES:
        rows = payload["splits"][split_name]
        if len(rows) != expected_size:
            raise AssertionError(f"wrong {split_name} count")
        for split_rank, row in enumerate(rows):
            if row["primary_split"] != split_name or row["split_rank"] != split_rank:
                raise AssertionError(f"wrong {split_name} rank annotation")
        ranked.extend(rows)
    if [row["global_hash_rank"] for row in ranked] != list(range(1000)):
        raise AssertionError("global hash ranks are not exactly 0..999")
    if ranked != sorted(
        ranked,
        key=lambda row: row["assignment_sha256"],
    ):
        raise AssertionError("rows are not in deterministic assignment order")

    for row in ranked:
        expected_assignment = assignment_hash(int(row["mlp_seed"]))
        if row["assignment_sha256"] != expected_assignment:
            raise AssertionError("assignment digest is not reproducible from seed")
        if row["decoded_columns"] != list(WEIGHT_COLUMNS):
            raise AssertionError("row decoded-column evidence changed")
        weight_hash = str(row["weights_sha256"])
        if len(weight_hash) != 64 or not set(weight_hash) <= set("0123456789abcdef"):
            raise AssertionError("invalid per-network weight SHA-256")
        if {"all_layer_means", "final_means"} & set(row):
            raise AssertionError("target value field entered network split")

    ids = [int(row["mlp_id"]) for row in ranked]
    seeds = [int(row["mlp_seed"]) for row in ranked]
    weights = [str(row["weights_sha256"]) for row in ranked]
    digests = [str(row["assignment_sha256"]) for row in ranked]
    if (
        len(set(ids)) != 1000
        or len(set(seeds)) != 1000
        or len(set(weights)) != 1000
        or len(set(digests)) != 1000
    ):
        raise AssertionError("IDs, seeds, assignment digests, or weights are not unique")

    expected_roles: dict[str, list[dict[str, object]]] = {}
    for role, begin, end in ROLE_SEGMENTS:
        expected_roles.setdefault(role, []).extend(ranked[begin:end])
        if any(row["observer_role"] != role for row in ranked[begin:end]):
            raise AssertionError(f"wrong role annotation in {begin}:{end}")
    if set(payload["observer_roles"]) != set(expected_roles):
        raise AssertionError("observer role names changed")
    for role, expected_rows in expected_roles.items():
        if [_identity(row) for row in payload["observer_roles"][role]] != [
            _identity(row) for row in expected_rows
        ]:
            raise AssertionError(f"observer role membership changed: {role}")

    expected_role_counts = {
        "observer_audit": 100,
        "observer_fit": 688,
        "observer_lockbox": 100,
        "observer_selection": 88,
        "retired_teacher_diagnostics": 24,
    }
    if payload["observer_role_counts"] != expected_role_counts:
        raise AssertionError("observer role counts changed")
    if payload["identity_attestation"] != {
        "weight_hash": "sha256 of C-contiguous float32[16,1024,1024] bytes",
        "unique_mlp_ids": 1000,
        "unique_mlp_seeds": 1000,
        "unique_assignment_hashes": 1000,
        "unique_weight_hashes": 1000,
    }:
        raise AssertionError("identity uniqueness attestation changed")

    return {
        "status": "PASS_FROZEN_FULL_NETWORK_SPLIT__ZERO_TARGET_VALUES",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "primary_counts": payload["counts"],
        "observer_role_counts": expected_role_counts,
        "unique_mlp_ids": len(set(ids)),
        "unique_mlp_seeds": len(set(seeds)),
        "unique_weight_hashes": len(set(weights)),
        "decoded_value_columns": list(WEIGHT_COLUMNS),
        "target_values_decoded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
