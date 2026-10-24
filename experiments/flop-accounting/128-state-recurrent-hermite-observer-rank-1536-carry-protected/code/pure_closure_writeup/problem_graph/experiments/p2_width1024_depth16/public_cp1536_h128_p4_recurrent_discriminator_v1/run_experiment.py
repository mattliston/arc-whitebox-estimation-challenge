#!/Users/matt/anaconda3/bin/python3
"""Frozen observer_fit-only H128/P4/C256 recurrent-Hermite discriminator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DESIGN = P2 / "learned_residual_observer_design"
V3 = DESIGN / "v3"
V6 = DESIGN / "v6_training_runtime"
FEATURE_ROOT = (
    DESIGN / "v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit"
)
TARGET = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FUSED_SOURCE = V6 / "fused_observer_v6.py"
OBSERVER_SOURCE = V3 / "observer_model.py"
STATIC_METER = V3 / "STATIC_METER_ESTIMATE_V3.json"
COMMON_METERS = V3 / "COMMON_DRIVER_METERS_V3.json"
TRAINING_SPEC = V6 / "TRAINING_SPEC_V6.json"
RUNTIME_PROJECTION = V6 / "RUNTIME_PROJECTION_V6.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
MANIFEST = HERE / "RUN_MANIFEST.json"
CHECKPOINT = HERE / "EPOCH_CHECKPOINT.pt"
RESULT = HERE / "RESULTS.json"

NETWORKS = 688
DEPTH = 16
WIDTH = 1024
LOCAL_FEATURES = 20
NORMALIZED_FEATURES = 19
FOLDS = 4
FOLD_SEED = 1_753_358_438
HELD_FOLD = 0
EPOCHS = 4
BATCH_SIZE = 2
HIDDEN = 128
CELL_WIDTH = 256
TILE_ROWS = 128
INIT_SEED = 4_671_727_249_568_777_182
PRIMARY_GATE = 1.0e-7
PROMOTION_BAND = 5.0e-7
SCALE_FLOOR = 2.0**-20
PROJECTED_FLOPS = 209_904_700_250
PHASE2_BUDGET = 2**41
MAX_FLOPS = 219_902_325_555
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 1 << 30

EXPECTED_HASHES = {
    OBSERVER_SOURCE: "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5",
    FUSED_SOURCE: "964fc8bb78f442ba1448d615770dbf915a7acf4503480520c87d0e0a2dd49350",
    STATIC_METER: "1a1c9971c0abc1cdfc185d03f59ec7e465843f6032948d4c4911dfe596f90a7c",
    COMMON_METERS: "07b7ea981656bcba5d4529d7e07c565b3af8625420c502abe7ff9e733f6b2c62",
    TRAINING_SPEC: "0595c14dc78feafef5602e3ff3c62b809c374f83e9da7547585ec8ab57a351a8",
    RUNTIME_PROJECTION: "c70129f7ad875ed4083a03b1267f7f9be40fbc71ff80d5ccb88dd46c8829c84c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def storage_bytes() -> int:
    return sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())


def enforce_resources(where: str) -> None:
    rss = resident_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"RSS bound exceeded at {where}: {rss}")
    used = storage_bytes()
    if used > MAX_STORAGE_BYTES:
        raise OSError(f"storage bound exceeded at {where}: {used}")


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def fold_of(mlp_id: int) -> int:
    payload = f"{FOLD_SEED}:{int(mlp_id)}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") % FOLDS


def load_targets() -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(TARGET, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    values = getattr(values, "storage", values)
    while not pa.types.is_primitive(values.type):
        values = values.flatten()
    targets = values.to_numpy(zero_copy_only=False).reshape(NETWORKS, DEPTH, WIDTH)
    if ids.dtype != np.int32 or targets.dtype != np.float32:
        raise AssertionError("observer_fit target schema changed")
    return np.ascontiguousarray(ids), np.ascontiguousarray(targets)


def row_dir(rank: int) -> Path:
    return FEATURE_ROOT / f"{rank:04d}"


def validate_row(rank: int, mlp_id: int) -> None:
    manifest = read_json(row_dir(rank) / "row_manifest.json")
    if manifest.get("role") != "observer_fit":
        raise AssertionError(f"rank {rank}: role mismatch")
    if int(manifest.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: role-rank mismatch")
    if int(manifest.get("mlp_id", -1)) != int(mlp_id):
        raise AssertionError(f"rank {rank}: mlp-id mismatch")


def checked_array(rank: int, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.load(row_dir(rank) / name, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.float32 or tuple(value.shape) != shape:
        raise AssertionError(f"rank {rank}: {name} schema changed")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"rank {rank}: {name} nonfinite")
    return value


def prepare_statistics(
    train_ranks: list[int], mlp_ids: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros((DEPTH, NORMALIZED_FEATURES), dtype=np.float64)
    m2 = np.zeros_like(mean)
    residual_sums = np.zeros(DEPTH - 1, dtype=np.float64)
    for index, rank in enumerate(train_ranks, start=1):
        validate_row(rank, int(mlp_ids[rank]))
        raw = checked_array(rank, "cp1536_raw_locals.npy", (DEPTH, WIDTH, LOCAL_FEATURES))
        host = checked_array(rank, "cp1536_host_rows.npy", (DEPTH, WIDTH))
        tail = np.asarray(raw[..., 1:], dtype=np.float64)
        batch_mean = tail.mean(axis=1, dtype=np.float64)
        centered = tail - batch_mean[:, None, :]
        batch_m2 = np.sum(centered * centered, axis=1, dtype=np.float64)
        batch_count = WIDTH
        if count == 0:
            mean[...] = batch_mean
            m2[...] = batch_m2
            count = batch_count
        else:
            delta = batch_mean - mean
            total = count + batch_count
            mean += delta * (np.float64(batch_count) / np.float64(total))
            m2 += batch_m2 + delta * delta * (
                np.float64(count) * np.float64(batch_count) / np.float64(total)
            )
            count = total
        residual = np.asarray(host[1:], dtype=np.float64) - targets[rank, 1:].astype(np.float64)
        residual_sums += np.sum(residual * residual, axis=1, dtype=np.float64)
        if index % 50 == 0 or index == len(train_ranks):
            print(f"PREP {index}/{len(train_ranks)} rss={resident_bytes()}", flush=True)
            enforce_resources(f"prep-{index}")
    expected_count = len(train_ranks) * WIDTH
    if count != expected_count:
        raise AssertionError("normalization count changed")
    scales = np.maximum(np.sqrt(m2 / np.float64(count)), np.float64(SCALE_FLOOR))
    residual_scales = np.maximum(
        np.sqrt(residual_sums / np.float64(expected_count)), np.float64(SCALE_FLOOR)
    ).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scales).all():
        raise FloatingPointError("invalid normalization statistics")
    if not np.isfinite(residual_scales).all() or not (residual_scales > 0).all():
        raise FloatingPointError("invalid residual scales")
    return np.ascontiguousarray(mean), np.ascontiguousarray(scales), np.ascontiguousarray(residual_scales)


def tensor_row(
    *,
    rank: int,
    expected_mlp_id: int,
    model: Any,
    observer: Any,
    centers: torch.Tensor,
    scales: torch.Tensor,
    masked_block: int | None,
    return_trace: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    validate_row(rank, expected_mlp_id)
    raw = checked_array(rank, "cp1536_raw_locals.npy", (DEPTH, WIDTH, LOCAL_FEATURES))
    host_array = checked_array(rank, "cp1536_host_rows.npy", (DEPTH, WIDTH))
    sigma_array = checked_array(rank, "cp1536_sigma.npy", (DEPTH, WIDTH))
    weights_array = checked_array(rank, "weights.npy", (DEPTH, WIDTH, WIDTH))
    raw_tensor = torch.from_numpy(np.asarray(raw))
    host = torch.from_numpy(np.asarray(host_array))
    sigma = torch.from_numpy(np.asarray(sigma_array))
    weights = torch.from_numpy(np.asarray(weights_array))
    standardized = observer.standardize_raw_features(raw_tensor, centers, scales)
    trace = model(
        standardized,
        weights,
        host,
        sigma,
        masked_optional_block=masked_block,
        return_trace=return_trace,
        fixed_inputs_verified=True,
    )
    return trace, host


def stable_model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_checkpoint(model: Any, optimizer: Any, completed_epoch: int, trace: list[dict[str, float]]) -> None:
    temporary = CHECKPOINT.with_name(f".{CHECKPOINT.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "completed_epoch": completed_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_trace": trace,
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
        },
        temporary,
    )
    temporary.replace(CHECKPOINT)


def load_checkpoint(model: Any, optimizer: Any) -> tuple[int, list[dict[str, float]]]:
    if not CHECKPOINT.is_file():
        return 0, []
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if state.get("preregistration_sha256") != sha256(PREREGISTRATION):
        raise AssertionError("checkpoint preregistration binding changed")
    if state.get("program_sha256") != sha256(Path(__file__)):
        raise AssertionError("checkpoint program binding changed")
    completed = int(state["completed_epoch"])
    if not 0 <= completed <= EPOCHS:
        raise AssertionError("checkpoint epoch invalid")
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    trace = list(state["training_trace"])
    if len(trace) != completed:
        raise AssertionError("checkpoint trace length changed")
    return completed, trace


def train(
    *,
    model: Any,
    observer: Any,
    optimizer: Any,
    centers: torch.Tensor,
    scales: torch.Tensor,
    residual_scales: torch.Tensor,
    train_ranks: list[int],
    mlp_ids: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, float]]:
    completed_epoch, training_trace = load_checkpoint(model, optimizer)
    if completed_epoch:
        print(f"RESUME completed_epoch={completed_epoch}", flush=True)
    train_set = set(train_ranks)
    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        started = time.monotonic()
        model.train()
        order = [position - 12 for position in observer.fit_epoch_order(epoch) if position - 12 in train_set]
        if len(order) != len(train_ranks) or len(set(order)) != len(train_ranks):
            raise AssertionError("filtered training permutation changed")
        loss_sum = 0.0
        grad_sum = 0.0
        grad_max = 0.0
        steps = 0
        for offset in range(0, len(order), BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            traces: list[dict[str, Any]] = []
            target_tensors: list[torch.Tensor] = []
            host_tensors: list[torch.Tensor] = []
            for rank in ranks:
                trace, host = tensor_row(
                    rank=rank,
                    expected_mlp_id=int(mlp_ids[rank]),
                    model=model,
                    observer=observer,
                    centers=centers,
                    scales=scales,
                    masked_block=observer.source_mask_block(epoch, rank + 12),
                    return_trace=True,
                )
                traces.append(trace)
                target_tensors.append(torch.from_numpy(targets[rank]))
                host_tensors.append(host)
            if len(ranks) == 2:
                loss = observer.exact_two_network_batch_loss(
                    traces, target_tensors, host_tensors, residual_scales
                )
            else:
                loss = observer.exact_training_loss(
                    traces[0], target_tensors[0], host_tensors[0], residual_scales
                )
            loss_value = float(loss.detach().cpu().item())
            grad_norm = float(observer.one_optimizer_step(model, loss))
            if not math.isfinite(loss_value) or not math.isfinite(grad_norm):
                raise FloatingPointError("training became nonfinite")
            loss_sum += loss_value
            grad_sum += grad_norm
            grad_max = max(grad_max, grad_norm)
            steps += 1
            if steps % 50 == 0 or offset + BATCH_SIZE >= len(order):
                print(
                    f"EPOCH {epoch}/{EPOCHS} step={steps}/{math.ceil(len(order)/BATCH_SIZE)} "
                    f"loss_mean={loss_sum/steps:.9g} rss={resident_bytes()}",
                    flush=True,
                )
                enforce_resources(f"epoch-{epoch}-step-{steps}")
        record = {
            "epoch": float(epoch),
            "mean_training_loss": loss_sum / steps,
            "mean_preclip_gradient_norm": grad_sum / steps,
            "maximum_preclip_gradient_norm": grad_max,
            "optimizer_steps": float(steps),
            "seconds": time.monotonic() - started,
        }
        training_trace.append(record)
        save_checkpoint(model, optimizer, epoch, training_trace)
        print(f"EPOCH_COMMIT {json.dumps(record, sort_keys=True)}", flush=True)
    return training_trace


def evaluate_ranks(
    *,
    ranks: list[int],
    model: Any,
    observer: Any,
    centers: torch.Tensor,
    scales: torch.Tensor,
    mlp_ids: np.ndarray,
    targets: np.ndarray,
    label: str,
    include_per_network: bool,
) -> dict[str, Any]:
    model.eval()
    candidate_sums = np.zeros(DEPTH, dtype=np.float64)
    host_sums = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[dict[str, Any]] = []
    maximum_correction = 0.0
    all_latched = True
    with torch.no_grad():
        for index, rank in enumerate(ranks, start=1):
            trace, host = tensor_row(
                rank=rank,
                expected_mlp_id=int(mlp_ids[rank]),
                model=model,
                observer=observer,
                centers=centers,
                scales=scales,
                masked_block=None,
                return_trace=True,
            )
            prediction = trace["predictions"].detach().cpu().numpy().astype(np.float64)
            host_array = host.detach().cpu().numpy().astype(np.float64)
            target = targets[rank].astype(np.float64)
            candidate_error = prediction - target
            host_error = host_array - target
            candidate_sums += np.sum(candidate_error * candidate_error, axis=1, dtype=np.float64)
            host_sums += np.sum(host_error * host_error, axis=1, dtype=np.float64)
            maximum_correction = max(
                maximum_correction,
                float(torch.max(torch.abs(trace["physical_correction"])).cpu().item()),
            )
            all_latched = all_latched and all(bool(value) for value in trace["latch"])
            if include_per_network:
                per_network.append(
                    {
                        "role_rank": rank,
                        "mlp_id": int(mlp_ids[rank]),
                        "candidate_final_mse": float(np.mean(candidate_error[-1] ** 2, dtype=np.float64)),
                        "host_final_mse": float(np.mean(host_error[-1] ** 2, dtype=np.float64)),
                    }
                )
            if index % 25 == 0 or index == len(ranks):
                print(f"EVAL {label} {index}/{len(ranks)} rss={resident_bytes()}", flush=True)
                enforce_resources(f"eval-{label}-{index}")
    denominator = np.float64(len(ranks) * WIDTH)
    candidate_layer = candidate_sums / denominator
    host_layer = host_sums / denominator
    return {
        "rows": len(ranks),
        "candidate_layer_mse": candidate_layer.tolist(),
        "candidate_final_mse": float(candidate_layer[-1]),
        "host_layer_mse": host_layer.tolist(),
        "host_final_mse": float(host_layer[-1]),
        "finite_complete_rollout": bool(all_latched and np.isfinite(candidate_layer).all()),
        "maximum_absolute_correction": maximum_correction,
        "per_network": per_network,
    }


def verify_frozen_receipts() -> None:
    preregistration = read_json(PREREGISTRATION)
    seal = read_json(SEAL)
    manifest = read_json(MANIFEST)
    if preregistration.get("status") != "frozen-before-any-protected-metric":
        raise AssertionError("preregistration status changed")
    if seal.get("status") != "frozen-before-any-protected-metric":
        raise AssertionError("seal status changed")
    if seal.get("preregistration_sha256") != sha256(PREREGISTRATION):
        raise AssertionError("preregistration hash changed")
    if seal.get("program_sha256") != sha256(Path(__file__)):
        raise AssertionError("program hash changed")
    if manifest.get("preregistration_sha256") != sha256(PREREGISTRATION):
        raise AssertionError("manifest preregistration binding changed")
    if manifest.get("program_sha256") != sha256(Path(__file__)):
        raise AssertionError("manifest program binding changed")
    if manifest.get("seal_sha256") != sha256(SEAL):
        raise AssertionError("manifest seal binding changed")
    for path, expected in EXPECTED_HASHES.items():
        if sha256(path) != expected:
            raise AssertionError(f"source authority changed: {path}")
    meter = read_json(STATIC_METER)["primary"]["actual_common_driver_meter"]
    if int(meter["recurrent_flops"]) != PROJECTED_FLOPS or PROJECTED_FLOPS > MAX_FLOPS:
        raise AssertionError("literal common-driver meter changed")


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    verify_frozen_receipts()
    if PROJECTED_FLOPS > MAX_FLOPS:
        raise AssertionError("projected estimator exceeds 0.1 score floor")
    fused = load_module(FUSED_SOURCE, "p2r0c1d2_fused_observer")
    observer = fused.OBSERVER
    determinism = observer.configure_cpu_determinism()
    mlp_ids, targets = load_targets()
    folds = np.asarray([fold_of(int(value)) for value in mlp_ids], dtype=np.int64)
    train_ranks = [int(rank) for rank in np.flatnonzero(folds != HELD_FOLD)]
    held_ranks = [int(rank) for rank in np.flatnonzero(folds == HELD_FOLD)]
    if len(train_ranks) != 541 or len(held_ranks) != 147:
        raise AssertionError("frozen split cardinality changed")
    started = time.monotonic()
    centers_array, scales_array, residual_array = prepare_statistics(train_ranks, mlp_ids, targets)
    centers = torch.from_numpy(centers_array)
    scales = torch.from_numpy(scales_array)
    residual_scales = torch.from_numpy(residual_array)
    model = fused.AcceleratedResidualObserver(
        arm="recurrent",
        hidden_channels=HIDDEN,
        cell_width=CELL_WIDTH,
        init_seed=INIT_SEED,
        tile_rows=TILE_ROWS,
    ).cpu()
    if sum(parameter.numel() for parameter in model.parameters()) != 171017:
        raise AssertionError("parameter count changed")
    optimizer = observer.build_adamw(model)
    model._v3_optimizer = optimizer
    training_trace = train(
        model=model,
        observer=observer,
        optimizer=optimizer,
        centers=centers,
        scales=scales,
        residual_scales=residual_scales,
        train_ranks=train_ranks,
        mlp_ids=mlp_ids,
        targets=targets,
    )
    # The protected targets are scored exactly once, only at the frozen epoch-4 state.
    held = evaluate_ranks(
        ranks=held_ranks,
        model=model,
        observer=observer,
        centers=centers,
        scales=scales,
        mlp_ids=mlp_ids,
        targets=targets,
        label="protected",
        include_per_network=True,
    )
    diagnostic_ranks = train_ranks[:32]
    diagnostic = evaluate_ranks(
        ranks=diagnostic_ranks,
        model=model,
        observer=observer,
        centers=centers,
        scales=scales,
        mlp_ids=mlp_ids,
        targets=targets,
        label="train-diagnostic",
        include_per_network=False,
    )
    raw = float(held["candidate_final_mse"])
    if raw < PRIMARY_GATE:
        disposition = "promote_full_architecture"
    elif raw <= PROMOTION_BAND:
        disposition = "promote_longer_fixed_training"
    else:
        disposition = "close_exact_four_epoch_schedule_architecture_remains_optimization_inconclusive"
    result = {
        "schema": "p2r0c1d2-cp1536-h128-p4-recurrent-discriminator-results-v1",
        "status": "complete",
        "graph_binding": {"node": "P2R0C1D2", "parent": "P2R0C1D"},
        "receipts": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
            "seal_sha256": sha256(SEAL),
            "manifest_sha256": sha256(MANIFEST),
            "target_sha256": sha256(TARGET),
            "model_tensor_sha256": stable_model_hash(model),
            "checkpoint_sha256": sha256(CHECKPOINT),
        },
        "split": {"train_rows": len(train_ranks), "protected_rows": len(held_ranks), "held_fold": HELD_FOLD},
        "architecture": {
            "host_cap": 1536,
            "hidden_channels": HIDDEN,
            "message_powers": 4,
            "cell_width": CELL_WIDTH,
            "parameters": 171017,
            "recurrent_transport": "streamed exact four-polynomial Hermite",
        },
        "meter": {
            "literal_common_driver_flops": PROJECTED_FLOPS,
            "phase2_budget_flops": PHASE2_BUDGET,
            "utilization_fraction": PROJECTED_FLOPS / PHASE2_BUDGET,
            "floor_margin_flops": MAX_FLOPS - PROJECTED_FLOPS,
        },
        "training": {
            "epochs": EPOCHS,
            "training_trace": training_trace,
            "residual_scales": residual_array.tolist(),
            "normalization_centers_sha256": hashlib.sha256(centers_array.astype("<f8", copy=False).tobytes()).hexdigest(),
            "normalization_scales_sha256": hashlib.sha256(scales_array.astype("<f8", copy=False).tobytes()).hexdigest(),
        },
        "metrics": {"protected": held, "train_diagnostic_32": diagnostic},
        "gate": {
            "threshold_raw_final_mse": PRIMARY_GATE,
            "observed_raw_final_mse": raw,
            "passed": bool(raw < PRIMARY_GATE),
            "disposition": disposition,
        },
        "resources": {
            "peak_rss_bytes": resident_bytes(),
            "storage_bytes_before_result": storage_bytes(),
            "wall_seconds": time.monotonic() - started,
            "determinism": determinism,
        },
        "scope_boundary": "Only observer_fit targets and target-blind observer_fit features were accessed. No observer_selection/private/public-name join, Q24 trajectory, network, or submission path was accessed.",
    }
    write_json_once(RESULT, result)
    enforce_resources("complete")
    print(
        f"COMPLETE protected_raw={raw:.12g} host={held['host_final_mse']:.12g} "
        f"gate={raw < PRIMARY_GATE} disposition={disposition}",
        flush=True,
    )


if __name__ == "__main__":
    main()
