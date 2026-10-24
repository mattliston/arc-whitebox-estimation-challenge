#!/usr/bin/env python3
"""Isolated synthetic [16,1024] roundtrip for the v3 custody chain.

This test creates its source and outputs only below a fresh /private/tmp root.
It never resolves a production cache path, authorization, claim, release, or
target value.  The synthetic target numbers are deterministic test patterns.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import authorized_full_targets_v2 as raw_reader
import role_target_artifact_codec_v1 as codec
import sealed_role_targets_v2 as sealed_reader


HERE = Path(__file__).resolve().parent
SYNTHETIC_TOKEN = "isolated-synthetic-parquet-v1"
PRODUCTION_PATHS = (
    HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM",
    HERE / "sealed_role_target_bundle.staging",
    HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V2.CLAIM",
    HERE / "sealed_role_target_bundle.v2.staging",
    HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V3.CLAIM",
    HERE / "sealed_role_target_bundle.v3.staging",
    HERE / "sealed_role_target_bundle",
)
TARGET_SHAPE = (16, 1024)
SOURCE_IDS = (4101, 4102, 4103, 4104, 4105, 4106, 4107)
ROLES = (
    "observer_fit",
    "observer_selection",
    "observer_audit",
    "observer_lockbox",
    "retired_teacher_diagnostics",
    "observer_fit",
    "observer_selection",
)
PRIMARY_SPLITS = ("train", "calibration", "validation", "test", "train", "train", "calibration")
WITHHELD_ROLES = {
    "observer_audit",
    "observer_lockbox",
    "retired_teacher_diagnostics",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def _assert_production_paths_absent() -> None:
    if any(path.exists() or path.is_symlink() for path in PRODUCTION_PATHS):
        raise AssertionError("production custody path exists during synthetic test")


def _array2d(values: np.ndarray) -> pa.FixedSizeListArray:
    primitive = pa.array(values.reshape(-1), type=pa.float32(), from_pandas=False)
    inner = pa.FixedSizeListArray.from_arrays(primitive, TARGET_SHAPE[1])
    return pa.FixedSizeListArray.from_arrays(inner, TARGET_SHAPE[0])


def _final_means(values: np.ndarray) -> pa.FixedSizeListArray:
    primitive = pa.array(values[:, -1, :].reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(primitive, TARGET_SHAPE[1])


def _synthetic_values() -> np.ndarray:
    base = np.arange(np.prod(TARGET_SHAPE), dtype=np.float32).reshape(TARGET_SHAPE)
    values = np.stack(
        [base * np.float32(1.0e-5) + np.float32(index) for index in range(len(SOURCE_IDS))],
        axis=0,
    )
    if values.shape != (len(SOURCE_IDS), *TARGET_SHAPE):
        raise AssertionError("synthetic target geometry changed")
    return np.ascontiguousarray(values, dtype=np.float32)


def _write_full_synthetic_source(path: Path, values: np.ndarray) -> None:
    target_storage = _array2d(values)
    target_metadata = codec.ARRAY2D_FIELD_METADATA
    # The denied weights column is deliberately tiny: its values are never
    # projected.  The compatibility contract under test is the full-size
    # [16,1024] all_layer_means physical column and exact eight-column names.
    weights = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(len(SOURCE_IDS), dtype=np.float32)), 1
    )
    arrays = [
        pa.array(SOURCE_IDS, type=pa.int32()),
        pa.array([f"synthetic-{value}" for value in SOURCE_IDS], type=pa.string()),
        pa.array([90_000 + index for index in range(len(SOURCE_IDS))], type=pa.int64()),
        weights,
        target_storage,
        _final_means(values),
        pa.array(np.linspace(0.1, 0.7, len(SOURCE_IDS)), type=pa.float64()),
        pa.array(["synthetic-only"] * len(SOURCE_IDS), type=pa.string()),
    ]
    schema = pa.schema(
        [
            pa.field("mlp_id", pa.int32(), nullable=False),
            pa.field("mlp_name", pa.string(), nullable=False),
            pa.field("mlp_seed", pa.int64(), nullable=False),
            pa.field("weights", weights.type, nullable=False),
            pa.field(
                "all_layer_means",
                target_storage.type,
                nullable=False,
                metadata=target_metadata,
            ),
            pa.field("final_means", arrays[5].type, nullable=False),
            pa.field("avg_variance", pa.float64(), nullable=False),
            pa.field("sampling_budget_breakdown", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=len(SOURCE_IDS),
        version="2.6",
    )
    parquet = pq.ParquetFile(path)
    expected = [
        "mlp_id",
        "mlp_name",
        "mlp_seed",
        "weights",
        "all_layer_means",
        "final_means",
        "avg_variance",
        "sampling_budget_breakdown",
    ]
    if parquet.schema_arrow.names != expected:
        raise AssertionError("synthetic source column order changed")
    if int(parquet.metadata.num_rows) != len(SOURCE_IDS):
        raise AssertionError("synthetic source row count changed")
    if int(parquet.metadata.num_row_groups) != 1:
        raise AssertionError("synthetic source must have one role-mixed row group")


def _expect_denied(iterator: Any, exception_type: type[BaseException]) -> None:
    try:
        next(iterator)
    except exception_type:
        return
    raise AssertionError("negative compatibility check unexpectedly succeeded")


def run() -> dict[str, Any]:
    _assert_production_paths_absent()
    if "WHEST_PHASE2_CUSTODIAN_BOUNDARY" in os.environ:
        raise AssertionError("production custodian boundary is set during synthetic test")
    old_token = os.environ.get(raw_reader.SYNTHETIC_COMPAT_ENV)
    values = _synthetic_values()
    membership = {
        mlp_id: (primary, role)
        for mlp_id, primary, role in zip(SOURCE_IDS, PRIMARY_SPLITS, ROLES)
    }
    role_ids = {
        role: [mlp_id for mlp_id, assigned in zip(SOURCE_IDS, ROLES) if assigned == role]
        for role in codec.PERSISTED_ROLES
    }

    with tempfile.TemporaryDirectory(
        prefix="whest-p2-synthetic-compat-", dir="/private/tmp"
    ) as temporary:
        root = Path(temporary)
        source_path = root / "full-synthetic.parquet"
        bundle = root / "sealed-synthetic-bundle"
        bundle.mkdir(mode=0o700)
        _write_full_synthetic_source(source_path, values)

        # The environment token is absent first, proving the raw test hook is
        # not a general local-file iterator.
        os.environ.pop(raw_reader.SYNTHETIC_COMPAT_ENV, None)
        _expect_denied(
            raw_reader.iter_synthetic_test_all_layer_means(
                parquet_paths=(source_path,),
                membership=membership,
                synthetic_root=root,
            ),
            PermissionError,
        )
        os.environ[raw_reader.SYNTHETIC_COMPAT_ENV] = SYNTHETIC_TOKEN

        retained: dict[str, dict[int, np.ndarray]] = {
            role: {} for role in codec.PERSISTED_ROLES
        }
        decoded_counts: dict[str, int] = {role: 0 for role in set(ROLES)}
        raw_rows = 0
        for row in raw_reader.iter_synthetic_test_all_layer_means(
            parquet_paths=(source_path,),
            membership=membership,
            synthetic_root=root,
        ):
            expected_index = SOURCE_IDS.index(row.mlp_id)
            if row.decoded_columns != raw_reader.TARGET_COLUMNS:
                raise AssertionError("synthetic raw projection columns changed")
            if not np.array_equal(row.all_layer_means, values[expected_index]):
                raise AssertionError("synthetic raw target roundtrip changed")
            decoded_counts[row.observer_role] += 1
            raw_rows += 1
            if row.observer_role in codec.PERSISTED_ROLES:
                retained[row.observer_role][row.mlp_id] = np.array(
                    row.all_layer_means, dtype=np.float32, order="C", copy=True
                )

        artifacts: dict[str, dict[str, Any]] = {}
        expected_role_values: dict[str, np.ndarray] = {}
        for role in ("observer_fit", "observer_selection"):
            role_values = np.stack(
                [retained[role][mlp_id] for mlp_id in role_ids[role]], axis=0
            )
            role_values = np.ascontiguousarray(role_values, dtype=np.float32)
            expected_role_values[role] = role_values
            artifacts[role] = codec.write_role_artifact(
                staging_path=bundle,
                role=role,
                ids=role_ids[role],
                values=role_values,
            )
        manifest = {
            "schema": sealed_reader.BUNDLE_SCHEMA,
            "status": "SYNTHETIC_COMPATIBILITY_ONLY_NOT_A_RELEASE",
            "synthetic_only": True,
            "target_columns": list(sealed_reader.TARGET_COLUMNS),
            "raw_full_dataset_included": False,
            "withheld_roles_included": False,
            "artifacts": artifacts,
            "fit_or_score_performed": False,
            "host_features_generated": False,
            "source_target_shape_per_row": list(TARGET_SHAPE),
        }
        manifest_path = codec.write_manifest_atomic(bundle, manifest)

        sealed_hashes: dict[str, str] = {}
        for role in ("observer_fit", "observer_selection"):
            rows = list(
                sealed_reader.iter_synthetic_test_role_targets(
                    role=role,
                    expected_ids=role_ids[role],
                    bundle_manifest_path=manifest_path,
                    synthetic_root=root,
                )
            )
            actual = np.stack([row.all_layer_means for row in rows], axis=0)
            if not np.array_equal(actual, expected_role_values[role]):
                raise AssertionError("sealed synthetic target roundtrip changed")
            if any(row.decoded_columns != sealed_reader.TARGET_COLUMNS for row in rows):
                raise AssertionError("sealed reader decoded-column contract changed")
            sealed_hashes[role] = _bytes_sha256(np.ascontiguousarray(actual))

        for denied_role in sorted(WITHHELD_ROLES):
            _expect_denied(
                sealed_reader.iter_synthetic_test_role_targets(
                    role=denied_role,
                    expected_ids=[SOURCE_IDS[ROLES.index(denied_role)]],
                    bundle_manifest_path=manifest_path,
                    synthetic_root=root,
                ),
                PermissionError,
            )
        if {path.name for path in bundle.iterdir()} != {
            "manifest.json",
            "observer_fit.parquet",
            "observer_selection.parquet",
        }:
            raise AssertionError("synthetic sealed bundle contains an unexpected file")
        if raw_rows != len(SOURCE_IDS) or sum(decoded_counts.values()) != len(SOURCE_IDS):
            raise AssertionError("synthetic raw decode accounting changed")

        report = {
            "status": "PASS_ISOLATED_SYNTHETIC_FULL_SHAPE_CUSTODY_ROUNDTRIP",
            "python": sys.version.split()[0],
            "pyarrow": pa.__version__,
            "numpy": np.__version__,
            "batch_name_api": "batch.schema.names",
            "synthetic_source": {
                "rows": len(SOURCE_IDS),
                "row_groups": 1,
                "all_layer_means_shape_per_row": list(TARGET_SHAPE),
                "all_layer_means_dtype": "float32",
                "sha256": _sha256_file(source_path),
                "target_values_c_order_sha256": _bytes_sha256(values),
                "projected_columns": list(raw_reader.TARGET_COLUMNS),
                "denied_columns_not_in_batches": sorted(raw_reader.DENIED_COLUMNS),
            },
            "raw_decode_accounting": {
                "synthetic_rows": raw_rows,
                "role_counts": dict(sorted(decoded_counts.items())),
                "production_rows": 0,
            },
            "sealed_artifacts": {
                role: {
                    "rows": artifacts[role]["rows"],
                    "parquet_sha256": artifacts[role]["sha256"],
                    "target_values_c_order_sha256": sealed_hashes[role],
                }
                for role in sorted(artifacts)
            },
            "bundle_manifest_sha256": _sha256_file(manifest_path),
            "withheld_roles_denied": sorted(WITHHELD_ROLES),
            "withheld_role_artifacts_written": 0,
            "fit_or_score_performed": False,
            "production_paths_touched": 0,
            "production_target_values_decoded": 0,
        }

    if old_token is None:
        os.environ.pop(raw_reader.SYNTHETIC_COMPAT_ENV, None)
    else:
        os.environ[raw_reader.SYNTHETIC_COMPAT_ENV] = old_token
    _assert_production_paths_absent()
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
