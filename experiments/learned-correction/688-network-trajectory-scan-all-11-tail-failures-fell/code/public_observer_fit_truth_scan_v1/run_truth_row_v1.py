#!/usr/bin/env python3
"""Score exact submitted Q24 on one public observer-fit MLP."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DATA_GATE = P2 / "data_gate"
AUTH = HERE / "TARGET_ACCESS_AUTHORIZATION.json"
TARGET_READER = DATA_GATE / "sealed_role_targets_v2.py"
TARGET_BUNDLE = DATA_GATE / "sealed_role_target_bundle/manifest.json"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
STORE_MANIFEST = STORE / "manifest.json"
ESTIMATOR = P2 / "fullbudget_k21_k4_closure_q24_v1/estimator.py"
RUNS = HERE / "runs"
FLOP_BUDGET = 2**41
LATE_THRESHOLD = 5e-7

HASHES = {
    TARGET_READER: "43cdc72d6bcd7b534d6c661d670b5e718af75bf482aff954ca6507d8d640cdbe",
    TARGET_BUNDLE: "2c0ea0bb09d0c86d46ca07fdde3563c4cda046a1fff152cdff26fde965e1f7e6",
    STORE_MANIFEST: "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50",
    ESTIMATOR: "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52",
}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def write_json_exclusive(path: Path, value) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(fd, payload[offset:])
            if count <= 0:
                raise OSError("short write")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)


def raw_sha256(np, array) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def load_target(np, role_rank: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    artifact = TARGET_BUNDLE.parent / "observer_fit.parquet"
    if sha256_file(artifact) != "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7":
        raise RuntimeError("observer-fit target artifact hash mismatch")
    parquet = pq.ParquetFile(artifact)
    if parquet.schema_arrow.names != ["mlp_id", "all_layer_means"]:
        raise RuntimeError("target physical projection changed")
    for index, batch in enumerate(parquet.iter_batches(
        batch_size=1,
        columns=["mlp_id", "all_layer_means"],
        use_threads=False,
    )):
        if index != role_rank:
            continue
        mlp_id = int(batch.column(0)[0].as_py())
        storage = batch.column(1)
        storage = getattr(storage, "storage", storage)
        while not pa.types.is_primitive(storage.type):
            storage = storage.flatten()
        target = storage.to_numpy(zero_copy_only=True).reshape(16, 1024)
        if target.dtype != np.float32 or not np.isfinite(target).all():
            raise RuntimeError("target geometry/finiteness changed")
        return mlp_id, np.ascontiguousarray(target).copy()
    raise RuntimeError("target role rank not found")


def layer_metrics(np, output, target):
    delta = output.astype(np.float64) - target.astype(np.float64)
    rows = []
    for layer in range(16):
        squared = delta[layer] * delta[layer]
        mse = float(np.mean(squared))
        rows.append({
            "layer": layer,
            "mse": mse,
            "rms": float(np.sqrt(mse)),
            "maximum_absolute_error": float(np.max(np.abs(delta[layer]))),
        })
    onset = next((row["layer"] for row in rows if row["mse"] > LATE_THRESHOLD), None)
    increments = [rows[layer]["mse"] - rows[layer - 1]["mse"] for layer in range(12, 16)]
    hit = bool(rows[15]["mse"] > LATE_THRESHOLD and onset in (12, 13, 14, 15) and max(increments) > 0.0)
    return {
        "layers": rows,
        "first_layer_above_5e_7": onset,
        "late_increments": increments,
        "late_max_increment": max(increments),
        "final_layer_mse": rows[15]["mse"],
        "max_layer_mse": max(row["mse"] for row in rows),
        "late_blowup_hit": hit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-rank", required=True, type=int)
    args = parser.parse_args()
    if not 0 <= args.role_rank < 688:
        raise ValueError("role rank outside observer_fit")
    for path, expected in HASHES.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"binding mismatch: {path}")

    RUNS.mkdir(exist_ok=True)
    row_dir = RUNS / f"row_{args.role_rank:04d}"
    row_dir.mkdir(exist_ok=False)
    write_json_exclusive(row_dir / "CLAIM.json", {
        "status": "CLAIMED_BEFORE_TARGET_DECODE_WEIGHT_OPEN_OR_ESTIMATOR_IMPORT",
        "role": "observer_fit",
        "role_rank": args.role_rank,
    })

    import numpy as np
    import flopscope as flops
    from whestbench.domain import MLP

    target_mlp_id, target = load_target(np, args.role_rank)
    store_manifest = json.loads(STORE_MANIFEST.read_text())
    entry = store_manifest["roles"]["observer_fit"]["row_manifests"][args.role_rank]
    row_manifest_path = STORE / entry["path"]
    if sha256_file(row_manifest_path) != entry["sha256"]:
        raise RuntimeError("weight row manifest hash mismatch")
    row_manifest = json.loads(row_manifest_path.read_text())
    if int(row_manifest["mlp_id"]) != target_mlp_id:
        raise RuntimeError("target/weight identity mismatch")
    weights_path = row_manifest_path.parent / "weights.npy"
    if sha256_file(weights_path) != row_manifest["files"]["weights.npy"]["sha256"]:
        raise RuntimeError("weights file hash mismatch")
    weights = np.load(weights_path, allow_pickle=False, mmap_mode="r")
    if weights.shape != (16, 1024, 1024) or weights.dtype != np.float32:
        raise RuntimeError("weight geometry mismatch")
    if raw_sha256(np, weights) != row_manifest["weights_raw_sha256"]:
        raise RuntimeError("weights raw hash mismatch")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(row_manifest["mlp_seed"]),
        name=f"public-observer-fit-rank-{args.role_rank}",
    )
    mlp.validate()

    estimator = load_module(f"submitted_q24_{args.role_rank}_{time.time_ns()}", ESTIMATOR)
    started = time.perf_counter()
    numerical_exception = None
    output = None
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as budget:
        try:
            output = estimator.Estimator().predict(mlp, FLOP_BUDGET)
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            numerical_exception = exc
    wall = time.perf_counter() - started
    operation_rows = budget.summary_dict(by_namespace=False).get("operations", {})
    operations = {
        name: {"calls": int(row.get("calls", 0)), "flops": int(row.get("flops", 0))}
        for name, row in sorted(operation_rows.items())
    }
    calls = sum(row["calls"] for row in operations.values())

    metrics = None
    finite = False
    if numerical_exception is None:
        output = np.asarray(output)
        if output.shape != (16, 1024) or output.dtype != np.float32:
            raise RuntimeError("output geometry mismatch")
        finite = bool(np.isfinite(output).all())
        if finite:
            output = np.ascontiguousarray(output).copy()
            metrics = layer_metrics(np, output, target)
            np.save(row_dir / "q24_output.npy", output, allow_pickle=False)
    hit = bool(numerical_exception is not None or not finite or metrics["late_blowup_hit"])
    result = {
        "schema": "phase2-public-observer-fit-submitted-q24-truth-row-v1",
        "status": "HIT" if hit else "NO_HIT",
        "role": "observer_fit",
        "role_rank": args.role_rank,
        "mlp_id": target_mlp_id,
        "weights_raw_sha256": row_manifest["weights_raw_sha256"],
        "estimator_sha256": HASHES[ESTIMATOR],
        "outcome": "NUMERICAL_FAILURE" if numerical_exception is not None else "FINITE" if finite else "NONFINITE",
        "exception_type": None if numerical_exception is None else type(numerical_exception).__name__,
        "exception_message": None if numerical_exception is None else str(numerical_exception),
        "wall_seconds": wall,
        "flops": int(budget.flops_used),
        "calls": calls,
        "operations": operations,
        "metrics": metrics,
        "hit": hit,
        "scope": {
            "decoded_columns": ["mlp_id", "all_layer_means"],
            "names_scores_board_joins": 0,
            "raw_full_dataset_access": 0,
        },
    }
    write_json_exclusive(row_dir / "result.json", result)
    print(json.dumps({
        "role_rank": args.role_rank,
        "mlp_id": target_mlp_id,
        "status": result["status"],
        "outcome": result["outcome"],
        "final_mse": None if metrics is None else metrics["final_layer_mse"],
        "onset": None if metrics is None else metrics["first_layer_above_5e_7"],
    }, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
