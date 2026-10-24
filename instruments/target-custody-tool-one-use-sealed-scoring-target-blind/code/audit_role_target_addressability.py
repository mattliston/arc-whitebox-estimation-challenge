#!/usr/bin/env python3
"""Target-free audit of whether raw full Parquet can preserve role seals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from full_data_gate import FULL, HERE, _attest_shard_pointer, load_full_manifest, pinned_snapshot_path
from verify_full_network_splits import DEFAULT_MANIFEST, verify as verify_split


PROTECTED_ROLES = frozenset(("observer_audit", "observer_lockbox"))
TRAINING_ROLES = frozenset(("observer_fit", "observer_selection"))


def audit(split_path: Path) -> dict[str, object]:
    split_receipt = verify_split(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_rows: dict[tuple[int, int], dict[str, object]] = {}
    for primary_rows in split["splits"].values():
        for row in primary_rows:
            coordinate = (int(row["shard_index"]), int(row["row_in_shard"]))
            if coordinate in source_rows:
                raise AssertionError(f"duplicate source coordinate: {coordinate}")
            source_rows[coordinate] = row
    if len(source_rows) != int(FULL["n_mlps"]):
        raise AssertionError("split does not cover every source row")

    groups: list[dict[str, object]] = []
    role_groups: dict[str, list[int]] = {
        role: [] for role in split["observer_roles"]
    }
    snapshot = pinned_snapshot_path()
    global_group = 0
    for shard_index, expected in enumerate(load_full_manifest()):
        resolved = _attest_shard_pointer(snapshot / str(expected["path"]), expected)
        parquet = pq.ParquetFile(resolved)
        local_row = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            count = int(row_group.num_rows)
            rows = [
                source_rows[(shard_index, row_in_shard)]
                for row_in_shard in range(local_row, local_row + count)
            ]
            local_row += count
            roles = Counter(str(row["observer_role"]) for row in rows)
            primary = Counter(str(row["primary_split"]) for row in rows)
            for role in roles:
                role_groups[role].append(global_group)

            id_chunk = row_group.column(0)
            if id_chunk.path_in_schema != "mlp_id":
                raise AssertionError("mlp_id is not physical column zero")
            stats = id_chunk.statistics
            if stats is None or not stats.has_min_max:
                raise AssertionError("mlp_id row-group statistics unavailable")
            target_chunk = row_group.column(4)
            if not target_chunk.path_in_schema.startswith("all_layer_means"):
                raise AssertionError("all_layer_means physical column moved")
            groups.append(
                {
                    "global_row_group": global_group,
                    "shard_index": shard_index,
                    "row_group_index": row_group_index,
                    "rows": count,
                    "mlp_id_min": int(stats.min),
                    "mlp_id_max": int(stats.max),
                    "mlp_ids": sorted(int(row["mlp_id"]) for row in rows),
                    "observer_role_counts": dict(sorted(roles.items())),
                    "primary_split_counts": dict(sorted(primary.items())),
                    "role_pure": len(roles) == 1,
                    "network_independently_addressable": count == 1,
                    "all_layer_means_has_column_index": bool(
                        target_chunk.has_column_index
                    ),
                    "all_layer_means_has_offset_index": bool(
                        target_chunk.has_offset_index
                    ),
                }
            )
            global_group += 1
        if local_row != parquet.metadata.num_rows:
            raise AssertionError(f"row-group accounting failed for shard {shard_index}")

    for role, indices in role_groups.items():
        if len(indices) != len(set(indices)):
            raise AssertionError(f"duplicate role/row-group mapping for {role}")
    protected_training_collisions = [
        group["global_row_group"]
        for group in groups
        if set(group["observer_role_counts"]) & TRAINING_ROLES
        and set(group["observer_role_counts"]) & PROTECTED_ROLES
    ]
    independently_addressable = all(
        bool(group["network_independently_addressable"]) for group in groups
    )
    all_role_pure = all(bool(group["role_pure"]) for group in groups)
    safe_raw_role_projection = independently_addressable or all_role_pure
    if safe_raw_role_projection:
        status = "PASS_RAW_ROLE_SCOPED_ROW_GROUP_READS_AVAILABLE"
        action = "role-scoped raw reads may be implemented after a separate audit"
    else:
        status = "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
        action = (
            "a trusted data custodian must read the full target column once, join the "
            "frozen roster, emit separately hashed role-specific artifacts, and expose "
            "only authorized role files to training; audit and lockbox files remain "
            "outside the training environment"
        )

    role_summary = {}
    for role, indices in sorted(role_groups.items()):
        mixed = [index for index in indices if not groups[index]["role_pure"]]
        role_summary[role] = {
            "row_groups_touched": len(indices),
            "mixed_row_groups": len(mixed),
            "mixed_row_group_indices": mixed,
        }
    return {
        "schema": "whest-phase2-role-target-addressability-audit-v1",
        "status": status,
        "split_manifest_sha256": split_receipt["sha256"],
        "full_shard_manifest_sha256": FULL["manifest_sha256"],
        "full_shards": len(load_full_manifest()),
        "networks": len(source_rows),
        "row_groups": len(groups),
        "row_group_sizes": dict(
            sorted(Counter(int(group["rows"]) for group in groups).items())
        ),
        "network_independent_row_groups": sum(
            bool(group["network_independently_addressable"]) for group in groups
        ),
        "role_pure_row_groups": sum(bool(group["role_pure"]) for group in groups),
        "mlp_id_stats_available_row_groups": len(groups),
        "all_layer_means_column_index_available_row_groups": sum(
            bool(group["all_layer_means_has_column_index"]) for group in groups
        ),
        "all_layer_means_offset_index_available_row_groups": sum(
            bool(group["all_layer_means_has_offset_index"]) for group in groups
        ),
        "raw_role_projection_physically_safe": safe_raw_role_projection,
        "training_role_row_groups_colliding_with_audit_or_lockbox": (
            protected_training_collisions
        ),
        "role_summary": role_summary,
        "row_group_evidence": groups,
        "external_constraint_action": action,
        "decoded_value_columns": [],
        "inspected_statistics_columns": ["mlp_id"],
        "target_values_decoded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = audit(args.split_manifest)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.receipt)
    print(encoded, end="")


if __name__ == "__main__":
    main()
