#!/Users/matt/anaconda3/bin/python3
"""One preregistered exposure-trained physical residual recurrence.

Only public ``observer_fit`` ranks 0:199 are opened.  Ranks 0:159 fit the
model and ranks 160:199 are scored once after the fixed final epoch.  The
separate observer-selection role is never addressed by this program.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
TARGET_FILE = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
OLD_HOST_ROOT = (
    P2 / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
)
PATCH_HOST_ROOT = (
    P2
    / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_near_floor_min"
)
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RESULT = HERE / "RESULT.json"
ARTIFACT = HERE / "MODEL.pt"

DEPTH = 16
WIDTH = 1024
LOCAL_FEATURES = 20
FIT_STOP = 160
VALID_STOP = 200
FIRST_CORRECTED_LAYER = 2
HIDDEN = 64
SEED = 0xE870_2026
THREADS = 2
BATCH_SIZE = 4
EPOCHS = 16
EXPOSURE_SCHEDULE = (
    0.0,
    0.0,
    0.25,
    0.25,
    0.50,
    0.50,
    0.75,
    0.75,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)
LEARNING_RATE = 1.5e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
TAIL_MSE_THRESHOLD = 5.0e-7
TRANSPORT_PENALTY = 1.0e-3
OLD_ESTIMATOR_SHA256 = (
    "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5"
)
PATCH_ESTIMATOR_SHA256 = (
    "26aca6ed1fb52ec431132f87739fe729ec6855cc2fa2c80642fadd1c4c786e5d"
)


def configure() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFF_FFFF)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(receipt: dict[str, object], name: str) -> float:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        raise AssertionError(f"receipt lacks metrics.{name}")
    return float(metrics[name])


def target_prefix(stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Open exactly the requested target prefix, never the next batch."""
    ids: list[int] = []
    values: list[np.ndarray] = []
    batches = pq.ParquetFile(TARGET_FILE).iter_batches(
        batch_size=1,
        columns=["mlp_id", "all_layer_means"],
        use_threads=False,
    )
    for batch in itertools.islice(batches, stop):
        ids.append(int(batch.column(0)[0].as_py()))
        storage = batch.column(1)
        storage = getattr(storage, "storage", storage)
        while not pa.types.is_primitive(storage.type):
            storage = storage.flatten()
        values.append(storage.to_numpy(zero_copy_only=True).reshape(DEPTH, WIDTH).copy())
    if len(values) != stop:
        raise RuntimeError(f"expected {stop} target rows, got {len(values)}")
    return (
        np.asarray(ids, dtype=np.int32),
        np.ascontiguousarray(np.stack(values), dtype=np.float32),
    )


def validate_receipt(
    receipt: dict[str, object], rank: int, mlp_id: int, expected_sha256: str
) -> None:
    if receipt.get("role") != "observer_fit":
        raise AssertionError(f"rank {rank}: role mismatch")
    if int(receipt.get("role_rank", -1)) != rank:
        raise AssertionError(f"rank {rank}: rank mismatch")
    if int(receipt.get("mlp_id", -1)) != mlp_id:
        raise AssertionError(f"rank {rank}: mlp_id mismatch")
    if receipt.get("estimator_sha256") != expected_sha256:
        raise AssertionError(f"rank {rank}: estimator hash mismatch")
    if receipt.get("exception_type") is not None:
        raise AssertionError(f"rank {rank}: failed host receipt")


def selected_host(rank: int, mlp_id: int) -> tuple[Path, str, float]:
    old_dir = OLD_HOST_ROOT / f"row_{rank:04d}"
    old = read_json(old_dir / "result.json")
    validate_receipt(old, rank, mlp_id, OLD_ESTIMATOR_SHA256)
    old_max = max(metric(old, "final_layer_mse"), metric(old, "max_layer_mse"))
    if old_max <= TAIL_MSE_THRESHOLD:
        return old_dir / "q24_output.npy", "old-trigger", old_max

    patch_dir = PATCH_HOST_ROOT / f"row_{rank:04d}"
    patch = read_json(patch_dir / "result.json")
    validate_receipt(patch, rank, mlp_id, PATCH_ESTIMATOR_SHA256)
    patch_max = max(metric(patch, "final_layer_mse"), metric(patch, "max_layer_mse"))
    if patch_max >= old_max:
        raise AssertionError(f"rank {rank}: near-floor-min host did not repair tail")
    return patch_dir / "q24_output.npy", "near-floor-min-repair", old_max


def load_panel() -> dict[str, object]:
    mlp_ids, target = target_prefix(VALID_STOP)
    hosts: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    weight_paths: list[Path] = []
    host_sources: list[str] = []
    old_maxes: list[float] = []
    for rank in range(VALID_STOP):
        feature_dir = FEATURE_ROOT / f"{rank:04d}"
        manifest = read_json(feature_dir / "row_manifest.json")
        mlp_id = int(mlp_ids[rank])
        if manifest.get("role") != "observer_fit":
            raise AssertionError(f"rank {rank}: feature role mismatch")
        if int(manifest.get("role_rank", -1)) != rank:
            raise AssertionError(f"rank {rank}: feature rank mismatch")
        if int(manifest.get("mlp_id", -1)) != mlp_id:
            raise AssertionError(f"rank {rank}: feature mlp_id mismatch")
        host_path, source, old_max = selected_host(rank, mlp_id)
        host = np.load(host_path, mmap_mode="r", allow_pickle=False)
        raw = np.load(
            feature_dir / "cp1792_raw_locals.npy", mmap_mode="r", allow_pickle=False
        )
        sigma = np.load(
            feature_dir / "cp1792_sigma.npy", mmap_mode="r", allow_pickle=False
        )
        weight_path = feature_dir / "weights.npy"
        if host.dtype != np.float32 or host.shape != (DEPTH, WIDTH):
            raise AssertionError(f"rank {rank}: host schema changed")
        if raw.dtype != np.float32 or raw.shape != (DEPTH, WIDTH, LOCAL_FEATURES):
            raise AssertionError(f"rank {rank}: raw schema changed")
        if sigma.dtype != np.float32 or sigma.shape != (DEPTH, WIDTH):
            raise AssertionError(f"rank {rank}: sigma schema changed")
        if not weight_path.is_file():
            raise FileNotFoundError(weight_path)
        hosts.append(np.asarray(host))
        raws.append(np.asarray(raw))
        sigmas.append(np.asarray(sigma))
        weight_paths.append(weight_path)
        host_sources.append(source)
        old_maxes.append(old_max)
    host_array = np.ascontiguousarray(np.stack(hosts), dtype=np.float32)
    return {
        "mlp_ids": mlp_ids,
        "target": target,
        "host": host_array,
        "error": np.ascontiguousarray(target - host_array, dtype=np.float32),
        "raw": np.ascontiguousarray(np.stack(raws), dtype=np.float32),
        "sigma": np.ascontiguousarray(np.stack(sigmas), dtype=np.float32),
        "weight_paths": weight_paths,
        "host_sources": host_sources,
        "old_maxes": old_maxes,
    }


def fit_moments(raw: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_tail = raw[:FIT_STOP, ..., 1:]
    centers = np.mean(fit_tail, axis=(0, 2), dtype=np.float64)
    second = np.mean(np.square(fit_tail, dtype=np.float64), axis=(0, 2), dtype=np.float64)
    scales = np.sqrt(np.maximum(second - centers * centers, (2.0**-20) ** 2))
    layer_rms = np.sqrt(
        np.mean(
            np.square(error[:FIT_STOP], dtype=np.float64),
            axis=(0, 2),
            dtype=np.float64,
        )
    )
    return (
        centers.astype(np.float32),
        scales.astype(np.float32),
        np.maximum(layer_rms, 2.0**-20).astype(np.float32),
    )


class LocalInnovationTransport(nn.Module):
    """The registered 6,688-parameter scalar physical recurrence."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))
        inputs = LOCAL_FEATURES + 2
        self.trunk = nn.Sequential(
            nn.Linear(inputs, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(DEPTH))
        nn.init.normal_(self.head_weight, mean=0.0, std=2.0e-3)
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH))

    def transport_gain(self) -> torch.Tensor:
        return 1.0 + 0.30 * torch.tanh(self.transport_delta)

    def innovation(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        message: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        rms = self.layer_rms[layer]
        message_rms = torch.clamp(message / rms, -8.0, 8.0)[..., None]
        message_sigma = torch.clamp(
            message / torch.clamp(sigma_layer * rms, min=2.0**-30),
            -8.0,
            8.0,
        )[..., None]
        features = torch.cat((raw_layer[..., :1], tail, message_rms, message_sigma), dim=-1)
        hidden = self.trunk(features)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[layer])
        standard = standard + self.head_bias[layer]
        return sigma_layer * rms * standard

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        *,
        truth: torch.Tensor | None = None,
        exposure: float = 1.0,
    ) -> torch.Tensor:
        if not 0.0 <= exposure <= 1.0:
            raise ValueError(exposure)
        if exposure < 1.0 and truth is None:
            raise ValueError("scheduled/teacher rollout requires truth")
        batch = raw.shape[0]
        state = torch.zeros(batch, WIDTH, dtype=raw.dtype, device=raw.device)
        outputs: list[torch.Tensor] = []
        gain = self.transport_gain()
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                state = torch.zeros_like(state)
                outputs.append(state)
                continue
            if layer == FIRST_CORRECTED_LAYER:
                prior = state
            elif exposure == 1.0:
                prior = state
            elif exposure == 0.0:
                prior = truth[:, layer - 1]
            else:
                prior = exposure * state + (1.0 - exposure) * truth[:, layer - 1]
            message = torch.bmm(prior[:, None, :], weights[:, layer]).squeeze(1)
            message = raw[:, layer, :, 5] * message
            local = self.innovation(raw[:, layer], sigma[:, layer], message, layer)
            state = gain[layer] * message + local
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def load_weights(paths: list[Path], ranks: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(
        np.stack(
            [np.load(paths[int(rank)], mmap_mode="r", allow_pickle=False) for rank in ranks]
        ),
        dtype=np.float32,
    )
    if array.shape != (len(ranks), DEPTH, WIDTH, WIDTH):
        raise AssertionError("weight schema changed")
    return torch.from_numpy(array)


def tensor_rows(array: np.ndarray, ranks: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array[ranks], dtype=np.float32))


def fit_model(
    model: LocalInnovationTransport, panel: dict[str, object]
) -> list[dict[str, float | int]]:
    raw = panel["raw"]
    sigma = panel["sigma"]
    error = panel["error"]
    paths = panel["weight_paths"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(paths, list)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * math.ceil(FIT_STOP / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=MIN_LEARNING_RATE
    )
    order_rng = np.random.default_rng(SEED ^ 0x5A17)
    layer_weights = torch.linspace(0.5, 1.5, DEPTH - FIRST_CORRECTED_LAYER)
    trace: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch, exposure in enumerate(EXPOSURE_SCHEDULE):
        order = order_rng.permutation(FIT_STOP)
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, FIT_STOP, BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            batch_raw = tensor_rows(raw, ranks)
            batch_sigma = tensor_rows(sigma, ranks)
            batch_truth = tensor_rows(error, ranks)
            batch_weights = load_weights(paths, ranks)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch_raw,
                batch_sigma,
                batch_weights,
                truth=batch_truth,
                exposure=exposure,
            )
            normalized = (
                prediction[:, FIRST_CORRECTED_LAYER:] - batch_truth[:, FIRST_CORRECTED_LAYER:]
            ) / model.layer_rms[FIRST_CORRECTED_LAYER:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            rollout = torch.mean(layer_weights * layer_mse)
            final_focus = layer_mse[-1]
            near_one = TRANSPORT_PENALTY * torch.mean(
                (model.transport_gain()[FIRST_CORRECTED_LAYER:] - 1.0).square()
            )
            loss = rollout + final_focus + near_one
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training objective at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            count = len(ranks)
            objective_sum += float(loss.detach()) * count
            examples += count
        record = {
            "epoch": epoch + 1,
            "exposure": exposure,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "transport_gain_min": float(model.transport_gain().detach().min()),
            "transport_gain_max": float(model.transport_gain().detach().max()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace


def score(
    model: LocalInnovationTransport,
    panel: dict[str, object],
    ranks: np.ndarray,
    *,
    exposure: float,
) -> dict[str, object]:
    raw = panel["raw"]
    sigma = panel["sigma"]
    error = panel["error"]
    paths = panel["weight_paths"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(paths, list)
    layer_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[float] = []
    maximum_absolute = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            truth = tensor_rows(error, group)
            prediction = model(
                tensor_rows(raw, group),
                tensor_rows(sigma, group),
                load_weights(paths, group),
                truth=truth if exposure < 1.0 else None,
                exposure=exposure,
            )
            finite = finite and bool(torch.isfinite(prediction).all())
            maximum_absolute = max(maximum_absolute, float(prediction.abs().max()))
            squared = (prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            per_network.extend(squared[:, -1].mean(dim=1).cpu().numpy().tolist())
    layer_mse = layer_sse / (len(ranks) * WIDTH)
    return {
        "exposure": exposure,
        "finite_full_rollout": finite,
        "maximum_absolute_correction": maximum_absolute,
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "per_network_final_mse": per_network,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def main() -> None:
    configure()
    if RESULT.exists() or ARTIFACT.exists():
        raise FileExistsError("registered experiment is single-run; output already exists")
    prereg = read_json(PREREGISTRATION)
    if prereg.get("status") != "frozen-before-metrics":
        raise AssertionError("PREREGISTRATION.json is not frozen")
    started = time.perf_counter()
    panel = load_panel()
    raw = panel["raw"]
    error = panel["error"]
    sources = panel["host_sources"]
    assert isinstance(raw, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(sources, list)
    centers, scales, layer_rms = fit_moments(raw, error)
    model = LocalInnovationTransport(centers, scales, layer_rms)
    if parameter_count(model) != 6688:
        raise AssertionError(f"unexpected parameter count {parameter_count(model)}")
    trace = fit_model(model, panel)

    # The fixed final checkpoint is scored once on development-validation.
    validation = np.arange(FIT_STOP, VALID_STOP)
    autonomous = score(model, panel, validation, exposure=1.0)
    teacher_forced = score(model, panel, validation, exposure=0.0)
    host_final = float(
        np.mean(
            np.square(error[FIT_STOP:VALID_STOP, -1], dtype=np.float64),
            dtype=np.float64,
        )
    )
    tail_ranks = [rank for rank, source in enumerate(sources) if source != "old-trigger"]
    result = {
        "schema": "public-exposure-trained-residual-state-v1",
        "status": "complete",
        "split": {
            "role": "observer_fit",
            "fit": [0, FIT_STOP],
            "development_validation_opened_once_after_fixed_epoch": [FIT_STOP, VALID_STOP],
            "observer_selection_targets_opened": 0,
            "observer_fit_targets_at_or_above_rank_200_opened": 0,
        },
        "data": {
            "tail_threshold": TAIL_MSE_THRESHOLD,
            "data_derived_repaired_tail_ranks": tail_ranks,
            "host_source_counts": {
                "old-trigger": sources.count("old-trigger"),
                "near-floor-min-repair": sources.count("near-floor-min-repair"),
            },
            "old_estimator_sha256": OLD_ESTIMATOR_SHA256,
            "near_floor_min_estimator_sha256": PATCH_ESTIMATOR_SHA256,
        },
        "architecture": {
            "parameters": parameter_count(model),
            "hidden": HIDDEN,
            "input_features": LOCAL_FEATURES + 2,
            "recurrence": "e_l=a_l*Phi_l*(e_{l-1}@W_l)+local_shared(raw20,message)",
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
        },
        "training": {
            "seed": SEED,
            "threads": THREADS,
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "exposure_schedule": list(EXPOSURE_SCHEDULE),
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "normalized all-corrected-layer rollout MSE + equal final-layer focus + near-one gain penalty",
            "checkpoint": "fixed final epoch; no validation checkpoint selection",
            "trace": trace,
        },
        "metrics": {
            "validation_host_final_mse": host_final,
            "validation_autonomous": autonomous,
            "validation_same_checkpoint_teacher_forced": teacher_forced,
            "autonomous_ratio_to_host": float(autonomous["final_mse"]) / host_final,
            "teacher_forced_ratio_to_host": float(teacher_forced["final_mse"]) / host_final,
        },
        "fatal_gates": {
            "finite_full_autonomous_rollout": bool(autonomous["finite_full_rollout"]),
            "beats_3_44e_8": float(autonomous["final_mse"]) < 3.44e-8,
            "reaches_1e_8": float(autonomous["final_mse"]) < 1.0e-8,
        },
        "provenance": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "program_sha256": sha256(Path(__file__)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    artifact = {
        "schema": "public-exposure-trained-residual-state-model-v1",
        "state_dict": model.state_dict(),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
    }
    torch.save(artifact, ARTIFACT)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "autonomous_final_mse": autonomous["final_mse"],
                "teacher_forced_final_mse": teacher_forced["final_mse"],
                "fatal_gates": result["fatal_gates"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
