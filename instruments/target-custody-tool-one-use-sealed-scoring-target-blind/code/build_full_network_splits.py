#!/usr/bin/env python3
"""Create the frozen 700/100/100/100 Phase 2 network split.

Only the four target-blind columns are physically projected.  Assignment is a
stable sort of ``SHA256("p2-teacher-v1|v2-phase2|" + decimal_mlp_seed)``.
Targets, weights, IDs, names, and source ordering do not influence membership.
Per-network IDs and weight hashes are recorded solely as leakage-safe identity
attestations and must be unique.  Duplicate seeds or assignment digests fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from full_data_gate import AUTHORITY, FULL, HERE, WEIGHT_COLUMNS, iter_target_blind_full


DOMAIN = b"p2-teacher-v1|v2-phase2|"
SPLIT_SIZES = (
    ("train", 700),
    ("calibration", 100),
    ("validation", 100),
    ("test", 100),
)
ROLE_SEGMENTS = (
    ("retired_teacher_diagnostics", 0, 12),
    ("observer_fit", 12, 700),
    ("retired_teacher_diagnostics", 700, 712),
    ("observer_selection", 712, 800),
    ("observer_audit", 800, 900),
    ("observer_lockbox", 900, 1000),
)
DEFAULT_OUTPUT = HERE / "full_network_splits.json"


def assignment_hash(mlp_seed: int) -> str:
    payload = DOMAIN + str(mlp_seed).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_manifest() -> dict[str, object]:
    identities: list[dict[str, object]] = []
    for row in iter_target_blind_full():
        weight_hash = hashlib.sha256(memoryview(row.weights).cast("B")).hexdigest()
        identities.append(
            {
                "ordinal": row.ordinal,
                "shard_index": row.shard_index,
                "row_in_shard": row.row_in_shard,
                "mlp_id": row.mlp_id,
                "mlp_name": row.mlp_name,
                "mlp_seed": row.mlp_seed,
                "weights_sha256": weight_hash,
                "decoded_columns": list(WEIGHT_COLUMNS),
                "assignment_sha256": assignment_hash(row.mlp_seed),
            }
        )
    if len(identities) != sum(size for _, size in SPLIT_SIZES):
        raise AssertionError("full target-blind row count does not match split sizes")
    pairs = [(int(row["mlp_id"]), int(row["mlp_seed"])) for row in identities]
    if len(set(pairs)) != len(pairs):
        raise AssertionError("duplicate (mlp_id, mlp_seed) identity")
    ids = [int(row["mlp_id"]) for row in identities]
    seeds = [int(row["mlp_seed"]) for row in identities]
    weight_hashes = [str(row["weights_sha256"]) for row in identities]
    if len(set(ids)) != len(ids):
        raise AssertionError("full mlp_id values are not unique")
    if len(set(seeds)) != len(seeds):
        raise AssertionError("full mlp_seed values are not unique")
    if len(set(weight_hashes)) != len(weight_hashes):
        raise AssertionError("full network weight hashes are not unique")
    assignment_hashes = [str(row["assignment_sha256"]) for row in identities]
    if len(set(assignment_hashes)) != len(assignment_hashes):
        raise AssertionError("teacher assignment SHA-256 values are not unique")

    ranked = sorted(
        identities,
        key=lambda row: row["assignment_sha256"],
    )
    for global_rank, row in enumerate(ranked):
        row["global_hash_rank"] = global_rank

    splits: dict[str, list[dict[str, object]]] = {}
    cursor = 0
    for split_name, split_size in SPLIT_SIZES:
        selected = ranked[cursor : cursor + split_size]
        for split_rank, row in enumerate(selected):
            row["split_rank"] = split_rank
            row["primary_split"] = split_name
        splits[split_name] = selected
        cursor += split_size

    roles: dict[str, list[dict[str, object]]] = {}
    for role_name, begin, end in ROLE_SEGMENTS:
        roles.setdefault(role_name, []).extend(ranked[begin:end])
        for row in ranked[begin:end]:
            if "observer_role" in row:
                raise AssertionError("observer role slices overlap")
            row["observer_role"] = role_name
    if any("observer_role" not in row for row in ranked):
        raise AssertionError("observer role slices do not cover all networks")

    return {
        "schema": "whest-phase2-network-split-v1",
        "status": "FROZEN_TARGET_BLIND_NETWORK_SPLIT",
        "repository": AUTHORITY["repository"],
        "revision": AUTHORITY["revision"],
        "commit": AUTHORITY["pinned_commit_sha"],
        "full_shard_manifest_sha256": FULL["manifest_sha256"],
        "assignment": {
            "domain_hex": DOMAIN.hex(),
            "formula": "SHA256('p2-teacher-v1|v2-phase2|' + decimal_mlp_seed)",
            "input": "ASCII domain immediately followed by signed decimal mlp_seed",
            "ranking": "assignment_sha256 ascending; duplicate seeds/digests fail",
            "assignment_inputs": ["mlp_seed"],
            "attestations_not_assignment_inputs": ["mlp_id", "weights_sha256"],
            "slices": {name: size for name, size in SPLIT_SIZES},
        },
        "observer_role_segments": [
            {"role": role, "begin_inclusive": begin, "end_exclusive": end}
            for role, begin, end in ROLE_SEGMENTS
        ],
        "decoded_value_columns": list(WEIGHT_COLUMNS),
        "target_values_decoded": 0,
        "identity_attestation": {
            "weight_hash": "sha256 of C-contiguous float32[16,1024,1024] bytes",
            "unique_mlp_ids": len(set(ids)),
            "unique_mlp_seeds": len(set(seeds)),
            "unique_assignment_hashes": len(set(assignment_hashes)),
            "unique_weight_hashes": len(set(weight_hashes)),
        },
        "counts": {name: len(splits[name]) for name, _ in SPLIT_SIZES},
        "observer_role_counts": {
            role_name: len(rows) for role_name, rows in sorted(roles.items())
        },
        "splits": splits,
        "observer_roles": roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        if args.output.read_bytes() != encoded:
            raise AssertionError("frozen split manifest is not reproducible")
        print(
            json.dumps(
                {
                    "status": "PASS_REPRODUCIBLE_FULL_NETWORK_SPLIT",
                    "path": str(args.output),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "counts": manifest["counts"],
                    "target_values_decoded": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "WROTE_FROZEN_FULL_NETWORK_SPLIT",
                "path": str(args.output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "counts": manifest["counts"],
                "target_values_decoded": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
