#!/usr/bin/env python3
"""Exact CPU fit/selection core imported only after the V4 one-use claim.

This module contains no reader and no feature-store implementation.  Its sole
external interface accepts already authorized modules from the claim-owning
launcher.  Import constructs no model or optimizer and performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
OBSERVER_SHA256 = "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5"
SELECTION_SHA256 = "806a2933afebd369040413a17f3e8f2cee939b0e870da7ab728697ba64eed516"
CONFIGS = (
    (1536, 128, 256, "primary"),
    (1792, 64, 96, "host_fidelity_control"),
)
ARMS = ("local_no_recurrence", "recurrent")
FIT_ROWS = 688
SELECTION_ROWS = 88
WIDTH = 1024
DEPTH = 16
SCALE_FLOOR = float(2.0**-20)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name: str, path: Path, expected_sha256: str):
    if _sha256_file(path) != expected_sha256:
        raise AssertionError(f"frozen V3 dependency changed: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBSERVER = _load("v4_cpu_frozen_observer", V3 / "observer_model.py", OBSERVER_SHA256)
SELECTION = _load("v4_cpu_frozen_selection", V3 / "selection_and_promotion.py", SELECTION_SHA256)


@dataclass
class TrainingResult:
    receipt: dict[str, Any]
    selected_model: Any | None
    selected_centers: np.ndarray | None
    selected_scales: np.ndarray | None


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _validate_raw_row(row: Any, expected_mlp_id: int) -> np.ndarray:
    if int(_field(row, "mlp_id")) != expected_mlp_id:
        raise AssertionError("feature/target mlp_id mismatch")
    raw = np.asarray(_field(row, "raw_locals"))
    if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, 20) or not raw.flags.c_contiguous:
        raise ValueError("raw_locals must be contiguous float32[16,1024,20]")
    if not np.isfinite(raw).all():
        raise FloatingPointError("raw_locals are nonfinite")
    return raw


def _validate_feature_row(row: Any, expected_mlp_id: int) -> dict[str, np.ndarray]:
    raw = _validate_raw_row(row, expected_mlp_id)
    expected = {
        "edge_matrices": ((DEPTH, 4, WIDTH, WIDTH), np.float32),
        "host_rows": ((DEPTH, WIDTH), np.float32),
        "sigma": ((DEPTH, WIDTH), np.float32),
    }
    result = {"raw_locals": raw}
    for name, (shape, dtype) in expected.items():
        value = np.asarray(_field(row, name))
        if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} has wrong shape/dtype/layout")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} is nonfinite")
        result[name] = value
    if not (result["host_rows"] >= 0.0).all() or not (result["sigma"] > 0.0).all():
        raise FloatingPointError("host/sigma sign contract failed")
    hashes = tuple(_field(row, "entering_cp_state_hashes"))
    if len(hashes) != DEPTH or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise AssertionError("entering CP state hash chain changed")
    if tuple(bool(value) for value in _field(row, "host_latch")) != (True,) * DEPTH:
        raise AssertionError("feature host latched on a released training row")
    return result


def _load_targets(
    *, reader_module: Any, reader_authorization: Path, bundle_manifest: Path
) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for role, count in (("observer_fit", FIT_ROWS), ("observer_selection", SELECTION_ROWS)):
        rows = []
        for row in reader_module.iter_authorized_sealed_role_targets(
            role=role,
            authorization_path=reader_authorization,
            bundle_manifest_path=bundle_manifest,
        ):
            target = np.array(row.all_layer_means, dtype=np.float32, order="C", copy=True)
            if target.shape != (DEPTH, WIDTH) or not np.isfinite(target).all():
                raise FloatingPointError("authorized target row changed shape or finiteness")
            rows.append((int(row.mlp_id), target))
        if len(rows) != count or len({mlp_id for mlp_id, _ in rows}) != count:
            raise AssertionError(f"authorized {role} rows changed")
        result[role] = tuple(rows)
    if set(mlp_id for mlp_id, _ in result["observer_fit"]) & set(
        mlp_id for mlp_id, _ in result["observer_selection"]
    ):
        raise AssertionError("fit and selection identities overlap")
    return result


def _normalization(store: Any, cap: int, fit_targets: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(DEPTH, dtype=np.int64)
    means = np.zeros((DEPTH, 19), dtype=np.float64)
    m2 = np.zeros((DEPTH, 19), dtype=np.float64)
    for role_rank, (mlp_id, _) in enumerate(fit_targets):
        row = store.get_normalization_row(cap, role_rank)
        raw = _validate_raw_row(row, mlp_id)
        for layer0 in range(DEPTH):
            for neuron in range(WIDTH):
                count[layer0] += 1
                sample = raw[layer0, neuron, 1:].astype(np.float64)
                delta = sample - means[layer0]
                means[layer0] += delta / np.float64(count[layer0])
                m2[layer0] += delta * (sample - means[layer0])
    if not np.array_equal(count, np.full(DEPTH, FIT_ROWS * WIDTH, dtype=np.int64)):
        raise AssertionError("normalization stream count changed")
    scales = np.sqrt(m2 / count[:, None].astype(np.float64))
    scales = np.maximum(scales, np.float64(SCALE_FLOOR))
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or not (scales > 0.0).all():
        raise FloatingPointError("normalization statistics are invalid")
    return np.ascontiguousarray(means), np.ascontiguousarray(scales)


def _residual_scales(store: Any, cap: int, fit_targets: tuple[Any, ...]) -> np.ndarray:
    sums = np.zeros(DEPTH - 1, dtype=np.float64)
    for role_rank, (mlp_id, target) in enumerate(fit_targets):
        row = _validate_feature_row(store.get_row(cap, "observer_fit", role_rank), mlp_id)
        for layer0 in range(1, DEPTH):
            for neuron in range(WIDTH):
                error = np.float64(row["host_rows"][layer0, neuron]) - np.float64(target[layer0, neuron])
                sums[layer0 - 1] = np.float64(sums[layer0 - 1] + np.float64(error * error))
    rms = np.sqrt(sums / np.float64(FIT_ROWS * WIDTH))
    rms = np.maximum(rms, np.float64(SCALE_FLOOR)).astype(np.float32)
    if not np.isfinite(rms).all() or not (rms > 0.0).all():
        raise FloatingPointError("target-derived residual scales are invalid")
    return np.ascontiguousarray(rms)


def _torch_row(
    *, model: Any, row: dict[str, np.ndarray], centers: np.ndarray,
    scales: np.ndarray, masked_block: int | None,
) -> dict[str, Any]:
    raw = torch.from_numpy(row["raw_locals"])
    standardized = OBSERVER.standardize_raw_features(
        raw,
        torch.from_numpy(centers),
        torch.from_numpy(scales),
    )
    trace = model(
        standardized,
        torch.from_numpy(row["edge_matrices"]),
        torch.from_numpy(row["host_rows"]),
        torch.from_numpy(row["sigma"]),
        masked_optional_block=masked_block,
        return_trace=True,
    )
    if not all(trace["latch"]):
        raise FloatingPointError("observer latched during CPU training/evaluation")
    if not torch.equal(
        trace["physical_correction"][0],
        torch.zeros(WIDTH, dtype=torch.float32),
    ):
        raise AssertionError("layer-1 correction is not structurally zero")
    return trace


def _evaluate(
    *, model: Any, store: Any, cap: int, role: str, targets: tuple[Any, ...],
    centers: np.ndarray, scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = np.empty((len(targets), DEPTH, WIDTH), dtype=np.float32)
    hosts = np.empty_like(predictions)
    with torch.no_grad():
        for role_rank, (mlp_id, _) in enumerate(targets):
            row = _validate_feature_row(store.get_row(cap, role, role_rank), mlp_id)
            trace = _torch_row(
                model=model, row=row, centers=centers, scales=scales,
                masked_block=None,
            )
            predictions[role_rank] = trace["predictions"].cpu().numpy()
            hosts[role_rank] = row["host_rows"]
    return predictions, hosts


def _clone_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _fit_arm(
    *, cap: int, hidden: int, cell_width: int, arm: str, store: Any,
    targets: dict[str, tuple[Any, ...]], centers: np.ndarray,
    scales: np.ndarray, residual_scales: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    model = OBSERVER.ResidualObserver(
        arm=arm,
        init_seed=OBSERVER.INIT_SEED,
        parity_nonzero_mean_head=False,
        hidden_channels=hidden,
        cell_width=cell_width,
    ).cpu()
    optimizer = OBSERVER.build_adamw(model)
    model._v3_optimizer = optimizer
    checks: list[tuple[int, float]] = []
    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_predictions: np.ndarray | None = None
    best_hosts: np.ndarray | None = None
    non_improving = 0

    def checkpoint(epoch: int) -> bool:
        nonlocal best_metric, best_state, best_predictions, best_hosts, non_improving
        prediction, host = _evaluate(
            model=model, store=store, cap=cap, role="observer_selection",
            targets=targets["observer_selection"], centers=centers, scales=scales,
        )
        target_array = np.stack([target for _, target in targets["observer_selection"]])
        metric = float(SELECTION._ordered_mse(prediction, target_array, 15))
        checks.append((epoch, metric))
        if metric < best_metric:
            best_metric = metric
            best_state = _clone_state(model)
            best_predictions = prediction.copy()
            best_hosts = host.copy()
            non_improving = 0
        else:
            non_improving += 1
        return non_improving == SELECTION.PATIENCE_CHECKS or epoch == 160

    checkpoint(0)
    residual_tensor = torch.from_numpy(residual_scales)
    last_epoch = 0
    for epoch in range(1, 161):
        model.train()
        order = OBSERVER.fit_epoch_order(epoch)
        for offset in range(0, FIT_ROWS, 2):
            positions = order[offset : offset + 2]
            traces, target_tensors, host_tensors = [], [], []
            for position in positions:
                role_rank = position - 12
                mlp_id, target = targets["observer_fit"][role_rank]
                row = _validate_feature_row(
                    store.get_row(cap, "observer_fit", role_rank), mlp_id
                )
                traces.append(
                    _torch_row(
                        model=model, row=row, centers=centers, scales=scales,
                        masked_block=OBSERVER.source_mask_block(epoch, position),
                    )
                )
                target_tensors.append(torch.from_numpy(target))
                host_tensors.append(torch.from_numpy(row["host_rows"]))
            loss = OBSERVER.exact_two_network_batch_loss(
                traces, target_tensors, host_tensors, residual_tensor
            )
            OBSERVER.one_optimizer_step(model, loss)
        last_epoch = epoch
        if epoch % 4 == 0 and checkpoint(epoch):
            break
    decision = SELECTION.select_checkpoint(checks)
    if best_state is None or best_predictions is None or best_hosts is None:
        raise AssertionError("checkpoint selector retained no model")
    model.load_state_dict(best_state, strict=True)
    if decision["best_epoch"] not in {epoch for epoch, _ in checks}:
        raise AssertionError("checkpoint decision is inconsistent")
    return model, best_predictions, best_hosts, {
        "arm": arm,
        "cap": cap,
        "hidden_channels": hidden,
        "cell_width": cell_width,
        "checks": checks,
        "checkpoint_decision": decision,
        "last_trained_epoch": last_epoch,
        "selected_model_sha256": OBSERVER.model_sha256(model),
        "optimizer_constructed_after_claim": True,
    }


def _array_sha256(name: str, value: np.ndarray) -> str:
    return OBSERVER.numpy_array_sha256(name, np.ascontiguousarray(value))


def run_training_session(
    *, reader_module: Any, reader_authorization: Path, bundle_manifest: Path,
    feature_provider_module: Any, feature_manifest: Path,
) -> TrainingResult:
    """Execute the exact fit/selection session; caller must own the claim."""

    determinism = OBSERVER.configure_cpu_determinism()
    if torch.backends.mps.is_available() and torch.get_default_device().type == "mps":
        raise RuntimeError("MPS is forbidden for V4 training")
    if determinism["num_threads"] != 1 or determinism["num_interop_threads"] != 1:
        raise RuntimeError("CPU thread authority changed")
    targets = _load_targets(
        reader_module=reader_module,
        reader_authorization=reader_authorization,
        bundle_manifest=bundle_manifest,
    )
    store = feature_provider_module.open_feature_store(feature_manifest)
    store.verify_all_artifact_hashes_after_claim()
    target_array = np.stack([target for _, target in targets["observer_selection"]])
    predictions_by_cap: dict[int, dict[str, np.ndarray]] = {}
    models: dict[tuple[int, str], Any] = {}
    statistics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    arms_receipt: dict[str, Any] = {}
    stats_receipt: dict[str, Any] = {}
    for cap, hidden, cell_width, config_role in CONFIGS:
        centers, scales = _normalization(store, cap, targets["observer_fit"])
        residual_scales = _residual_scales(store, cap, targets["observer_fit"])
        statistics[cap] = (centers, scales)
        cap_predictions: dict[str, np.ndarray] = {}
        host_reference: np.ndarray | None = None
        for arm in ARMS:
            model, prediction, host, arm_receipt = _fit_arm(
                cap=cap, hidden=hidden, cell_width=cell_width, arm=arm,
                store=store, targets=targets, centers=centers, scales=scales,
                residual_scales=residual_scales,
            )
            if host_reference is None:
                host_reference = host
            elif not np.array_equal(host_reference, host):
                raise AssertionError("local/recurrent host bytes differ")
            models[(cap, arm)] = model
            cap_predictions[arm] = prediction
            arms_receipt[f"cp{cap}_{arm}"] = arm_receipt
        if host_reference is None:
            raise AssertionError("host reference is absent")
        predictions_by_cap[cap] = {
            "host": host_reference,
            "local_no_recurrence": cap_predictions["local_no_recurrence"],
            "recurrent": cap_predictions["recurrent"],
        }
        stats_receipt[str(cap)] = {
            "role": config_role,
            "centers_sha256": _array_sha256(f"cp{cap}/centers", centers),
            "scales_sha256": _array_sha256(f"cp{cap}/scales", scales),
            "residual_scales_sha256": _array_sha256(
                f"cp{cap}/residual_scales", residual_scales
            ),
        }
    launch = SELECTION.planned_launch_selection(predictions_by_cap, target_array)
    winner_cap = int(launch["winner_host_cap"])
    winner_arm = str(launch["winner_arm"])
    selected_model = None if winner_arm == "host" else models[(winner_cap, winner_arm)]
    selected_centers, selected_scales = statistics[winner_cap]
    receipt = {
        "schema": "phase2-observer-v4-cpu-fit-selection-session-v1",
        "status": "COMPLETE_CPU_FIT_SELECTION__UNPUBLISHED_PENDING_POSTFIT_PARITY_METER_AND_AUDIT",
        "configs": [
            {"cap": cap, "hidden_channels": hidden, "cell_width": width, "role": role}
            for cap, hidden, width, role in CONFIGS
        ],
        "arms": arms_receipt,
        "statistics": stats_receipt,
        "launch_selection": launch,
        "fit_rows": FIT_ROWS,
        "selection_rows": SELECTION_ROWS,
        "layer1_structural_zero": True,
        "layer2_non_regression_applied": True,
        "audit_rows_opened": 0,
        "lockbox_rows_opened": 0,
        "mps_used": False,
        "device": "cpu",
        "postfit_release_authorized": False,
    }
    return TrainingResult(
        receipt=receipt,
        selected_model=selected_model,
        selected_centers=selected_centers,
        selected_scales=selected_scales,
    )
