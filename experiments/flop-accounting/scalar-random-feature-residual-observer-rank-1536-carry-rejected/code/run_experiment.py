#!/Users/matt/anaconda3/bin/python3
"""Frozen four-fold CP1536 scalar physical-transport RFF discriminator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
TARGET = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
CACHE = HERE / "cache"
RESULT = HERE / "CV_RESULTS.json"

NETWORKS = 688
DEPTH = 16
WIDTH = 1024
RAW_FEATURES = 20
BASE_FEATURES = 21
RANDOM_FEATURES = 128
DESIGN_FEATURES = 1 + BASE_FEATURES + RANDOM_FEATURES
FOLDS = 4
FOLD_SEED = 1_753_358_438
SAMPLE_SEED = 1_368_231_781
RANDOM_MAP_SEED = 1_592_654_542
SAMPLE_NEURONS = 32
RIDGE = 1.0e-3
CORRECTION_CLIP = 1.0 / 32.0
SCALE_FLOOR = 2.0**-20
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 1 << 30
RAW_GATE_COLUMN = 5
HOST_FLOPS = 199_018_976_258
PROJECTED_TOTAL_FLOPS = 199_548_139_522
PHASE2_BUDGET = 2**41
SCORE_FLOOR_MAX = 219_902_325_555
PRIMARY_GATE = 1.0e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value < 1 << 30:
        value *= 1024
    return value


def storage_bytes() -> int:
    return sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())


def enforce_resources(where: str) -> int:
    rss = resident_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"RSS bound exceeded at {where}: {rss}")
    used = storage_bytes()
    if used > MAX_STORAGE_BYTES:
        raise OSError(f"storage bound exceeded at {where}: {used}")
    return rss


def verify_seal() -> None:
    preregistration = read_json(PREREGISTRATION)
    seal = read_json(SEAL)
    if preregistration.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("preregistration status changed")
    if seal.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("seal status changed")
    if seal.get("preregistration_sha256") != sha256(PREREGISTRATION):
        raise AssertionError("preregistration hash changed")
    if seal.get("program_sha256") != sha256(Path(__file__)):
        raise AssertionError("program hash changed")


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


def fold_of(mlp_id: int) -> int:
    payload = f"{FOLD_SEED}:{int(mlp_id)}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") % FOLDS


def sampled_neurons(mlp_id: int) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64(np.uint64(SAMPLE_SEED) ^ np.uint64(int(mlp_id)))
    )
    return np.sort(generator.choice(WIDTH, size=SAMPLE_NEURONS, replace=False))


def random_maps() -> tuple[np.ndarray, np.ndarray, str]:
    weights = np.empty((DEPTH, RANDOM_FEATURES, BASE_FEATURES), dtype=np.float64)
    biases = np.empty((DEPTH, RANDOM_FEATURES), dtype=np.float64)
    digest = hashlib.sha256()
    for layer in range(DEPTH):
        generator = np.random.Generator(np.random.PCG64(RANDOM_MAP_SEED + layer))
        weights[layer] = generator.normal(
            0.0, 0.45, size=(RANDOM_FEATURES, BASE_FEATURES)
        )
        biases[layer] = generator.uniform(-1.0, 1.0, size=RANDOM_FEATURES)
        digest.update(weights[layer].astype("<f8", copy=False).tobytes())
        digest.update(biases[layer].astype("<f8", copy=False).tobytes())
    return weights, biases, digest.hexdigest()


def validate_feature_row(rank: int, mlp_id: int) -> None:
    manifest = read_json(row_dir(rank) / "row_manifest.json")
    if manifest.get("role") != "observer_fit":
        raise AssertionError(f"rank {rank}: role mismatch")
    if int(manifest.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: role-rank mismatch")
    if int(manifest.get("mlp_id", -1)) != int(mlp_id):
        raise AssertionError(f"rank {rank}: mlp-id mismatch")


def prepare_cache(
    mlp_ids: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "MANIFEST.json"
    paths = {
        "sample_raw": CACHE / "sample_raw.npy",
        "sample_sigma": CACHE / "sample_sigma.npy",
        "sample_error": CACHE / "sample_error.npy",
        "sample_teacher_message": CACHE / "sample_teacher_message.npy",
        "teacher_message": CACHE / "teacher_message.npy",
    }
    expected = {
        "schema": "cp1536-scalar-transport-rff-cache-v1",
        "networks": NETWORKS,
        "depth": DEPTH,
        "width": WIDTH,
        "sample_neurons": SAMPLE_NEURONS,
        "target_sha256": sha256(TARGET),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
    }
    if manifest_path.is_file():
        if read_json(manifest_path) != expected:
            raise AssertionError("cache manifest changed")
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError("manifested cache is incomplete")
    else:
        if any(path.exists() for path in paths.values()):
            raise FileExistsError("unmanifested partial cache preserved")
        sample_raw = np.lib.format.open_memmap(
            paths["sample_raw"], mode="w+", dtype=np.float32,
            shape=(NETWORKS, DEPTH, SAMPLE_NEURONS, RAW_FEATURES),
        )
        sample_sigma = np.lib.format.open_memmap(
            paths["sample_sigma"], mode="w+", dtype=np.float32,
            shape=(NETWORKS, DEPTH, SAMPLE_NEURONS),
        )
        sample_error = np.lib.format.open_memmap(
            paths["sample_error"], mode="w+", dtype=np.float32,
            shape=(NETWORKS, DEPTH, SAMPLE_NEURONS),
        )
        sample_teacher = np.lib.format.open_memmap(
            paths["sample_teacher_message"], mode="w+", dtype=np.float32,
            shape=(NETWORKS, DEPTH, SAMPLE_NEURONS),
        )
        teacher = np.lib.format.open_memmap(
            paths["teacher_message"], mode="w+", dtype=np.float32,
            shape=(NETWORKS, DEPTH, WIDTH),
        )
        for rank in range(NETWORKS):
            mlp_id = int(mlp_ids[rank])
            validate_feature_row(rank, mlp_id)
            directory = row_dir(rank)
            raw = np.load(directory / "cp1536_raw_locals.npy", mmap_mode="r")
            host = np.load(directory / "cp1536_host_rows.npy", mmap_mode="r")
            sigma = np.load(directory / "cp1536_sigma.npy", mmap_mode="r")
            weights = np.load(directory / "weights.npy", mmap_mode="r")
            if raw.shape != (DEPTH, WIDTH, RAW_FEATURES) or raw.dtype != np.float32:
                raise AssertionError(f"rank {rank}: raw schema changed")
            if host.shape != (DEPTH, WIDTH) or host.dtype != np.float32:
                raise AssertionError(f"rank {rank}: host schema changed")
            if sigma.shape != (DEPTH, WIDTH) or sigma.dtype != np.float32:
                raise AssertionError(f"rank {rank}: sigma schema changed")
            if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
                raise AssertionError(f"rank {rank}: weight schema changed")
            error = np.asarray(targets[rank] - host, dtype=np.float32)
            teacher[rank, 0] = 0.0
            for layer in range(1, DEPTH):
                transported = np.asarray(weights[layer].T @ error[layer - 1], dtype=np.float32)
                teacher[rank, layer] = np.clip(
                    raw[layer, :, RAW_GATE_COLUMN] * transported
                    / np.maximum(sigma[layer], np.float32(2.0**-30)),
                    -8.0,
                    8.0,
                )
            chosen = sampled_neurons(mlp_id)
            sample_raw[rank] = raw[:, chosen]
            sample_sigma[rank] = sigma[:, chosen]
            sample_error[rank] = error[:, chosen]
            sample_teacher[rank] = teacher[rank][:, chosen]
            if (rank + 1) % 50 == 0 or rank + 1 == NETWORKS:
                for value in (sample_raw, sample_sigma, sample_error, sample_teacher, teacher):
                    value.flush()
                print(f"prepare {rank + 1}/{NETWORKS}", flush=True)
                enforce_resources(f"prepare-{rank + 1}")
        for value in (sample_raw, sample_sigma, sample_error, sample_teacher, teacher):
            value.flush()
        del sample_raw, sample_sigma, sample_error, sample_teacher, teacher
        write_json_once(manifest_path, expected)
    return (
        np.load(paths["sample_raw"], mmap_mode="r"),
        np.load(paths["sample_sigma"], mmap_mode="r"),
        np.load(paths["sample_error"], mmap_mode="r"),
        np.load(paths["sample_teacher_message"], mmap_mode="r"),
        np.load(paths["teacher_message"], mmap_mode="r"),
        np.asarray([fold_of(int(value)) for value in mlp_ids], dtype=np.int64),
    )


def base_features(
    raw: np.ndarray,
    message: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    layer: int,
) -> np.ndarray:
    tail = np.clip((np.asarray(raw[..., 1:], dtype=np.float64) - center) / scale, -8.0, 8.0)
    depth = np.full(tail.shape[:-1] + (1,), 2.0 * layer / 15.0 - 1.0)
    msg = np.clip(np.asarray(message, dtype=np.float64), -8.0, 8.0)[..., None]
    return np.concatenate((depth, tail, msg), axis=-1)


def design_matrix(
    base: np.ndarray, map_weight: np.ndarray, map_bias: np.ndarray
) -> np.ndarray:
    flat = base.reshape(-1, BASE_FEATURES)
    random = np.tanh(flat @ map_weight.T + map_bias)
    return np.concatenate((np.ones((len(flat), 1)), flat, random), axis=1)


def fit_models(
    sample_raw: np.ndarray,
    sample_sigma: np.ndarray,
    sample_error: np.ndarray,
    sample_teacher: np.ndarray,
    folds: np.ndarray,
    map_weight: np.ndarray,
    map_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.empty((FOLDS, DEPTH, RAW_FEATURES - 1), dtype=np.float64)
    scales = np.empty_like(centers)
    betas = np.zeros((FOLDS, DEPTH, DESIGN_FEATURES), dtype=np.float64)
    for fold in range(FOLDS):
        train = np.flatnonzero(folds != fold)
        for layer in range(DEPTH):
            tail = np.asarray(sample_raw[train, layer, :, 1:], dtype=np.float64)
            centers[fold, layer] = tail.mean(axis=(0, 1), dtype=np.float64)
            scales[fold, layer] = np.maximum(
                tail.std(axis=(0, 1), dtype=np.float64), SCALE_FLOOR
            )
            if layer == 0:
                continue
            base = base_features(
                sample_raw[train, layer], sample_teacher[train, layer],
                centers[fold, layer], scales[fold, layer], layer,
            )
            design = design_matrix(base, map_weight[layer], map_bias[layer])
            target = (
                np.asarray(sample_error[train, layer], dtype=np.float64)
                / np.maximum(np.asarray(sample_sigma[train, layer], dtype=np.float64), 2.0**-30)
            ).reshape(-1)
            gram = (design.T @ design) / len(target)
            rhs = (design.T @ target) / len(target)
            penalty = np.eye(DESIGN_FEATURES, dtype=np.float64) * RIDGE
            penalty[0, 0] = 0.0
            betas[fold, layer] = np.linalg.solve(gram + penalty, rhs)
        print(f"fit fold {fold + 1}/{FOLDS}", flush=True)
        enforce_resources(f"fit-fold-{fold}")
    return centers, scales, betas


def normalized_prediction(
    raw_layer: np.ndarray,
    message: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    layer: int,
    map_weight: np.ndarray,
    map_bias: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    base = base_features(raw_layer, message, center, scale, layer)
    design = design_matrix(base, map_weight, map_bias)
    return np.clip(design @ beta, -CORRECTION_CLIP, CORRECTION_CLIP)


def evaluate(
    mlp_ids: np.ndarray,
    targets: np.ndarray,
    teacher_message: np.ndarray,
    folds: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    betas: np.ndarray,
    map_weight: np.ndarray,
    map_bias: np.ndarray,
) -> dict[str, object]:
    squared = {
        "host": np.zeros(DEPTH, dtype=np.float64),
        "teacher": np.zeros(DEPTH, dtype=np.float64),
        "autonomous": np.zeros(DEPTH, dtype=np.float64),
    }
    fold_squared = {
        name: np.zeros((FOLDS, DEPTH), dtype=np.float64) for name in squared
    }
    fold_counts = np.zeros(FOLDS, dtype=np.int64)
    per_network: list[dict[str, object]] = []
    maximum_abs_correction = 0.0
    for rank in range(NETWORKS):
        fold = int(folds[rank])
        directory = row_dir(rank)
        raw = np.load(directory / "cp1536_raw_locals.npy", mmap_mode="r")
        host = np.load(directory / "cp1536_host_rows.npy", mmap_mode="r")
        sigma = np.load(directory / "cp1536_sigma.npy", mmap_mode="r")
        weights = np.load(directory / "weights.npy", mmap_mode="r")
        target = targets[rank]
        teacher_correction = np.zeros((DEPTH, WIDTH), dtype=np.float64)
        autonomous_correction = np.zeros_like(teacher_correction)
        for layer in range(1, DEPTH):
            teacher_unit = normalized_prediction(
                raw[layer], teacher_message[rank, layer],
                centers[fold, layer], scales[fold, layer], layer,
                map_weight[layer], map_bias[layer], betas[fold, layer],
            )
            teacher_correction[layer] = np.asarray(sigma[layer], dtype=np.float64) * teacher_unit
            transported = np.asarray(
                weights[layer].T @ autonomous_correction[layer - 1].astype(np.float32),
                dtype=np.float64,
            )
            autonomous_message = np.clip(
                np.asarray(raw[layer, :, RAW_GATE_COLUMN], dtype=np.float64) * transported
                / np.maximum(np.asarray(sigma[layer], dtype=np.float64), 2.0**-30),
                -8.0,
                8.0,
            )
            autonomous_unit = normalized_prediction(
                raw[layer], autonomous_message,
                centers[fold, layer], scales[fold, layer], layer,
                map_weight[layer], map_bias[layer], betas[fold, layer],
            )
            autonomous_correction[layer] = (
                np.asarray(sigma[layer], dtype=np.float64) * autonomous_unit
            )
        maximum_abs_correction = max(
            maximum_abs_correction,
            float(np.max(np.abs(autonomous_correction))),
            float(np.max(np.abs(teacher_correction))),
        )
        predictions = {
            "host": np.maximum(np.asarray(host, dtype=np.float64), 0.0),
            "teacher": np.maximum(np.asarray(host, dtype=np.float64) + teacher_correction, 0.0),
            "autonomous": np.maximum(np.asarray(host, dtype=np.float64) + autonomous_correction, 0.0),
        }
        target64 = np.asarray(target, dtype=np.float64)
        row_metrics: dict[str, float] = {}
        for name, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"rank {rank}: nonfinite {name} prediction")
            layer_sum = np.square(prediction - target64).sum(axis=1, dtype=np.float64)
            squared[name] += layer_sum
            fold_squared[name][fold] += layer_sum
            row_metrics[f"{name}_final_mse"] = float(layer_sum[-1] / WIDTH)
        fold_counts[fold] += WIDTH
        per_network.append(
            {
                "role_rank": rank,
                "mlp_id": int(mlp_ids[rank]),
                "fold": fold,
                **row_metrics,
            }
        )
        if (rank + 1) % 50 == 0 or rank + 1 == NETWORKS:
            print(f"evaluate {rank + 1}/{NETWORKS}", flush=True)
            enforce_resources(f"evaluate-{rank + 1}")
    denominator = NETWORKS * WIDTH
    layer_mse = {name: (value / denominator).tolist() for name, value in squared.items()}
    folds_out = []
    for fold in range(FOLDS):
        folds_out.append(
            {
                "fold": fold,
                "networks": int(np.sum(folds == fold)),
                **{
                    f"{name}_final_mse": float(
                        fold_squared[name][fold, -1] / fold_counts[fold]
                    )
                    for name in squared
                },
            }
        )
    return {
        "layer_mse": layer_mse,
        "final_mse": {name: float(values[-1]) for name, values in layer_mse.items()},
        "folds": folds_out,
        "per_network": per_network,
        "maximum_absolute_correction": maximum_abs_correction,
        "finite_complete_rollout": True,
    }


def main() -> None:
    started = time.perf_counter()
    verify_seal()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if PROJECTED_TOTAL_FLOPS > SCORE_FLOOR_MAX:
        raise AssertionError("frozen architecture no longer fits the score floor")
    mlp_ids, targets = load_targets()
    sample_raw, sample_sigma, sample_error, sample_teacher, teacher, folds = prepare_cache(
        mlp_ids, targets
    )
    map_weight, map_bias, map_sha256 = random_maps()
    centers, scales, betas = fit_models(
        sample_raw, sample_sigma, sample_error, sample_teacher,
        folds, map_weight, map_bias,
    )
    metrics = evaluate(
        mlp_ids, targets, teacher, folds, centers, scales, betas,
        map_weight, map_bias,
    )
    final = metrics["final_mse"]
    assert isinstance(final, dict)
    teacher_pass = float(final["teacher"]) < PRIMARY_GATE
    primary_pass = float(final["autonomous"]) < PRIMARY_GATE
    result = {
        "schema": "public-cp1536-scalar-transport-rff-cv-results-v1",
        "status": "complete",
        "provenance": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
            "seal_sha256": sha256(SEAL),
            "target_sha256": sha256(TARGET),
            "fixed_random_map_sha256": map_sha256,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_bytes": resident_bytes(),
            "storage_bytes": storage_bytes(),
        },
        "split": {
            "role": "observer_fit",
            "networks": NETWORKS,
            "fold_counts": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
            "fit_neurons_per_network_per_layer": SAMPLE_NEURONS,
            "held_neurons_per_network_per_layer": WIDTH,
            "observer_selection_targets_opened": 0,
            "q24_trajectories_opened": 0,
            "board_name_joins": 0,
        },
        "architecture": {
            "host": "CP1536",
            "base_features": BASE_FEATURES,
            "random_features": RANDOM_FEATURES,
            "design_features": DESIGN_FEATURES,
            "ridge": RIDGE,
            "normalized_correction_clip": CORRECTION_CLIP,
            "training": "teacher-forced scalar physical transport",
            "evaluation": "teacher-forced and autonomous out-of-fold",
        },
        "projected_deployment": {
            "host_flops": HOST_FLOPS,
            "total_flops": PROJECTED_TOTAL_FLOPS,
            "phase2_budget": PHASE2_BUDGET,
            "fraction_of_budget": PROJECTED_TOTAL_FLOPS / PHASE2_BUDGET,
            "score_floor_margin": SCORE_FLOOR_MAX - PROJECTED_TOTAL_FLOPS,
            "adjusted_multiplier": 0.1,
            "literal_meter_status": "required only after accuracy pass",
        },
        "metrics": metrics,
        "decision_gates": {
            "finite": bool(metrics["finite_complete_rollout"]),
            "teacher_forced_below_1e_7": teacher_pass,
            "autonomous_below_1e_7": primary_pass,
        },
        "verdict": (
            "PROMOTE_EXACT_ARCHITECTURE_TO_LITERAL_METER"
            if primary_pass
            else "PROMOTE_EXPOSURE_TRAINED_SUCCESSOR"
            if teacher_pass
            else "CLOSE_EXACT_SCALAR_RFF_FORM"
        ),
        "scope_boundary": "No observer_selection/private target, Q24 trajectory, or public-board name was accessed. Failure closes only this exact scalar fixed-random-feature spelling, not the broader H128/P4 recurrent observer.",
    }
    write_json_once(RESULT, result)
    print(json.dumps({
        "final_mse": final,
        "decision_gates": result["decision_gates"],
        "verdict": result["verdict"],
        "projected_total_flops": PROJECTED_TOTAL_FLOPS,
        "elapsed_seconds": result["provenance"]["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
