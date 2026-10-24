#!/Users/matt/anaconda3/bin/python3
"""Frozen full-688 four-fold dynamic multistate residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
LOADER_DIR = P2 / "public_full688_local_innovation_transport_v1"
sys.path.insert(0, str(LOADER_DIR))
from dataset import (  # noqa: E402
    DEPTH,
    LOCAL_FEATURES,
    WIDTH,
    PublicResidualDataset,
)


PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
CACHE = HERE / "cache"
FOLD_DIR = HERE / "folds"
CV_RESULT = HERE / "CV_RESULTS.json"

SEED = 1_753_358_438
FOLDS = 4
CHANNELS = 4
HIDDEN = 96
FIRST_CORRECTED_LAYER = 2
THREADS = 2
BATCH_SIZE = 3
EPOCHS = 6
EXPOSURE_SCHEDULE = (0.0, 0.25, 0.50, 0.75, 1.0, 1.0)
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
AUX_STATE_PENALTY = 2.0e-5
IDENTITY_PENALTY = 5.0e-4
GRADIENT_CLIP = 5.0
PRIMARY_GATE = 3.123872816e-8
STRETCH_GATE = 1.0e-8
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
HOST_FLOPS = 2_126_977_459_133
PHASE2_BUDGET = 2_200_000_000_000


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


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux reports KiB.  Magnitude is not a valid
    # discriminator because a healthy Darwin process can be below 1 GiB.
    if sys.platform != "darwin":
        value *= 1024
    return value


def enforce_resources(where: str) -> int:
    rss = resident_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"RSS bound exceeded at {where}: {rss}")
    total = 0
    if HERE.exists():
        for path in HERE.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    if total > MAX_STORAGE_BYTES:
        raise OSError(f"storage bound exceeded at {where}: {total}")
    return rss


def verify_seal() -> dict[str, object]:
    if not SEAL.is_file():
        raise FileNotFoundError("PREREGISTRATION_SEAL.json is required")
    seal = read_json(SEAL)
    if seal.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("seal status changed")
    if seal.get("preregistration_sha256") != sha256(PREREGISTRATION):
        raise AssertionError("preregistration hash changed")
    if seal.get("program_sha256") != sha256(Path(__file__)):
        raise AssertionError("program hash changed")
    if seal.get("run_manifest_sha256") != sha256(RUN_MANIFEST):
        raise AssertionError("run manifest hash changed")
    loader_path = (HERE / str(seal.get("loader_path"))).resolve()
    if seal.get("loader_sha256") != sha256(loader_path):
        raise AssertionError("identity-safe dataset loader hash changed")
    prereg = read_json(PREREGISTRATION)
    if prereg.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("preregistration status changed")
    return seal


def fold_of(mlp_id: int) -> int:
    payload = f"{SEED}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % FOLDS


def readiness_or_raise(dataset: PublicResidualDataset) -> dict[str, object]:
    readiness = dataset.readiness()
    if not readiness["ready"]:
        raise RuntimeError(f"observer_fit scan/repair gate is not ready: {readiness}")
    if readiness["available"] != 688 or readiness["unpatched_tail_ranks"]:
        raise AssertionError("readiness contract changed")
    return readiness


def cache_manifest(dataset: PublicResidualDataset) -> dict[str, object]:
    readiness = readiness_or_raise(dataset)
    return {
        "schema": "full688-dynamic-multistate-cache-v1",
        "observer_fit_networks": dataset.total,
        "tail_threshold": readiness["tail_threshold"],
        "tail_ranks": readiness["data_derived_tail_ranks"],
        "repaired_tail_ranks": readiness["repaired_tail_ranks"],
        "old_estimator_sha256": dataset.expected_host_sha256,
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
    }


def prepare_cache(
    dataset: PublicResidualDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    readiness_or_raise(dataset)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "MANIFEST.json"
    expected_manifest = cache_manifest(dataset)
    error_path = CACHE / "patched_error.npy"
    sums_path = CACHE / "raw_sufficient_stats.npz"
    if manifest_path.is_file():
        if read_json(manifest_path) != expected_manifest:
            raise AssertionError("cache manifest no longer matches the frozen dataset")
        if not error_path.is_file() or not sums_path.is_file():
            raise FileNotFoundError("cache manifest exists but arrays are incomplete")
    else:
        if error_path.exists() or sums_path.exists():
            raise FileExistsError("unmanifested cache artifact")
        error_map = np.lib.format.open_memmap(
            error_path,
            mode="w+",
            dtype=np.float32,
            shape=(dataset.total, DEPTH, WIDTH),
        )
        sums = np.empty(
            (dataset.total, DEPTH, LOCAL_FEATURES - 1), dtype=np.float64
        )
        sums2 = np.empty_like(sums)
        for rank in range(dataset.total):
            row = dataset.load(rank)
            error_map[rank] = dataset.targets[rank] - row.host
            raw_tail = np.asarray(row.raw_locals[..., 1:], dtype=np.float64)
            sums[rank] = raw_tail.sum(axis=1, dtype=np.float64)
            sums2[rank] = np.square(raw_tail).sum(axis=1, dtype=np.float64)
            if (rank + 1) % 50 == 0 or rank + 1 == dataset.total:
                error_map.flush()
                print(f"prepare {rank + 1}/{dataset.total}", flush=True)
        error_map.flush()
        del error_map
        np.savez(sums_path, sum=sums, sum2=sums2)
        write_json_once(manifest_path, expected_manifest)
    error = np.load(error_path, mmap_mode="r", allow_pickle=False)
    payload = np.load(sums_path, allow_pickle=False)
    sums = payload["sum"]
    sums2 = payload["sum2"]
    if error.dtype != np.float32 or error.shape != (688, DEPTH, WIDTH):
        raise AssertionError("cached error schema changed")
    if sums.shape != (688, DEPTH, LOCAL_FEATURES - 1) or sums2.shape != sums.shape:
        raise AssertionError("cached raw-moment schema changed")
    folds = np.asarray([fold_of(int(value)) for value in dataset.mlp_ids], dtype=np.int64)
    return error, sums, sums2, folds


def moments_for(
    train: np.ndarray,
    error: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = float(len(train) * WIDTH)
    total = sums[train].sum(axis=0, dtype=np.float64)
    total2 = sums2[train].sum(axis=0, dtype=np.float64)
    centers = total / count
    variance = np.maximum(total2 / count - centers * centers, (2.0**-20) ** 2)
    layer_rms = np.sqrt(
        np.mean(np.square(error[train], dtype=np.float64), axis=(0, 2), dtype=np.float64)
    )
    return (
        centers.astype(np.float32),
        np.sqrt(variance).astype(np.float32),
        np.maximum(layer_rms, 2.0**-20).astype(np.float32),
    )


class DynamicMultiStateResidual(nn.Module):
    """Four exact physical transports plus a shared pointwise observer cell."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))
        inputs = LOCAL_FEATURES + 2 * CHANNELS
        self.trunk = nn.Sequential(
            nn.Linear(inputs, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.source_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN, CHANNELS))
        self.source_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.gate_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN, CHANNELS))
        self.gate_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.mix_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS, CHANNELS))
        nn.init.normal_(self.source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.gate_weight, mean=0.0, std=2.0e-3)

    def local_hidden(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        messages: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        rms = self.layer_rms[layer]
        message_rms = torch.clamp(messages / rms, -8.0, 8.0)
        message_sigma = torch.clamp(
            messages / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
            -8.0,
            8.0,
        )
        features = torch.cat(
            (raw_layer[..., :1], tail, message_rms, message_sigma), dim=-1
        )
        return self.trunk(features)

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        *,
        truth: torch.Tensor | None = None,
        exposure: float = 1.0,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= exposure <= 1.0:
            raise ValueError(exposure)
        if exposure < 1.0 and truth is None:
            raise ValueError("truth is required for scheduled exposure")
        batch = raw.shape[0]
        states = torch.zeros(
            batch, WIDTH, CHANNELS, dtype=raw.dtype, device=raw.device
        )
        outputs: list[torch.Tensor] = []
        state_history: list[torch.Tensor] = []
        identity = torch.eye(CHANNELS, dtype=raw.dtype, device=raw.device)
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                states = torch.zeros_like(states)
                outputs.append(states[..., 0])
                state_history.append(states)
                continue
            if layer == FIRST_CORRECTED_LAYER:
                messages = torch.zeros_like(states)
            else:
                prior = states
                if exposure < 1.0:
                    primary = (
                        exposure * states[..., 0]
                        + (1.0 - exposure) * truth[:, layer - 1]
                    )
                    prior = torch.cat((primary[..., None], states[..., 1:]), dim=-1)
                messages = torch.bmm(prior.transpose(1, 2), weights[:, layer]).transpose(1, 2)
                messages = raw[:, layer, :, 5, None] * messages
            hidden = self.local_hidden(raw[:, layer], sigma[:, layer], messages, layer)
            source_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.source_weight[layer]
            ) + self.source_bias[layer]
            physical_scale = sigma[:, layer] * self.layer_rms[layer]
            source = physical_scale[..., None] * source_standard
            gate_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.gate_weight[layer]
            ) + self.gate_bias[layer]
            dynamic_gain = 1.0 + 0.25 * torch.tanh(
                gate_standard + self.transport_delta[layer]
            )
            mixing = identity + 0.12 * torch.tanh(self.mix_delta[layer])
            transported = torch.einsum(
                "bwc,cd->bwd", dynamic_gain * messages, mixing
            )
            states = transported + source
            outputs.append(states[..., 0])
            state_history.append(states)
        prediction = torch.stack(outputs, dim=1)
        if return_states:
            return prediction, torch.stack(state_history, dim=1)
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def cost_record(model: nn.Module) -> dict[str, object]:
    corrected = DEPTH - FIRST_CORRECTED_LAYER
    transitions = corrected - 1
    dense_vector_matrix = 2 * WIDTH * WIDTH - WIDTH
    transport = transitions * CHANNELS * dense_vector_matrix
    raw20_materialization_reserve = DEPTH * 16 * WIDTH * WIDTH
    raw20_standardization = 2 * LOCAL_FEATURES * DEPTH * WIDTH
    inputs = LOCAL_FEATURES + 2 * CHANNELS
    per_unit_dense = (
        (2 * inputs - 1) * HIDDEN
        + (2 * HIDDEN - 1) * HIDDEN
        + (2 * HIDDEN - 1) * (2 * CHANNELS)
    )
    nonlinear_reserve = 64 * (2 * HIDDEN + CHANNELS)
    state_vector_and_mix_reserve = 128 + CHANNELS * (2 * CHANNELS - 1)
    local_cell = corrected * WIDTH * (
        per_unit_dense + nonlinear_reserve + state_vector_and_mix_reserve
    )
    output_addition = DEPTH * WIDTH
    correction = (
        transport
        + raw20_materialization_reserve
        + raw20_standardization
        + local_cell
        + output_addition
    )
    total = HOST_FLOPS + correction
    return {
        "parameter_count": parameter_count(model),
        "measured_host_flops": HOST_FLOPS,
        "four_channel_phi_w_transport": transport,
        "raw20_materialization_reserve": raw20_materialization_reserve,
        "raw20_standardization": raw20_standardization,
        "local_cell_dense_plus_special_reserve": local_cell,
        "output_addition": output_addition,
        "correction_source_structure_upper_estimate": correction,
        "host_plus_correction_upper_estimate": total,
        "phase2_budget": PHASE2_BUDGET,
        "headroom_under_upper_estimate": PHASE2_BUDGET - total,
        "within_phase2_budget_by_upper_estimate": total <= PHASE2_BUDGET,
        "authority": "frozen conservative source-structure admission estimate; literal flopscope meter required only after an accuracy pass",
    }


def load_batch(
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [dataset.load(int(rank)) for rank in ranks]
    raw = torch.from_numpy(
        np.ascontiguousarray(np.stack([row.raw_locals for row in rows]), dtype=np.float32)
    )
    sigma = torch.from_numpy(
        np.ascontiguousarray(np.stack([row.sigma for row in rows]), dtype=np.float32)
    )
    weights = torch.from_numpy(
        np.ascontiguousarray(
            np.stack(
                [
                    np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
                    for row in rows
                ]
            ),
            dtype=np.float32,
        )
    )
    truth = torch.from_numpy(np.ascontiguousarray(error[ranks], dtype=np.float32))
    return raw, sigma, weights, truth


def fit_fold(
    fold: int,
    model: DynamicMultiStateResidual,
    dataset: PublicResidualDataset,
    train: np.ndarray,
    error: np.ndarray,
) -> tuple[list[dict[str, object]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * math.ceil(len(train) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=MIN_LEARNING_RATE
    )
    layer_weights = torch.linspace(0.5, 1.5, DEPTH - FIRST_CORRECTED_LAYER)
    order_rng = np.random.default_rng(SEED ^ (fold * 0x9E3779B1))
    trace: list[dict[str, object]] = []
    started = time.perf_counter()
    max_rss = enforce_resources("training start")
    for epoch, exposure in enumerate(EXPOSURE_SCHEDULE):
        order = order_rng.permutation(train)
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            raw, sigma, weights, truth = load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states = model(
                raw,
                sigma,
                weights,
                truth=truth,
                exposure=exposure,
                return_states=True,
            )
            normalized = (
                prediction[:, FIRST_CORRECTED_LAYER:]
                - truth[:, FIRST_CORRECTED_LAYER:]
            ) / model.layer_rms[FIRST_CORRECTED_LAYER:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            rollout = torch.mean(layer_weights * layer_mse)
            final_focus = layer_mse[-1]
            auxiliary = states[:, FIRST_CORRECTED_LAYER:, :, 1:]
            aux_normalized = auxiliary / model.layer_rms[
                FIRST_CORRECTED_LAYER:
            ][None, :, None, None]
            aux_penalty = AUX_STATE_PENALTY * aux_normalized.square().mean()
            identity_penalty = IDENTITY_PENALTY * (
                model.transport_delta.square().mean()
                + model.mix_delta.square().mean()
            )
            loss = rollout + final_focus + aux_penalty + identity_penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"fold {fold} epoch {epoch + 1}: nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        max_rss = max(max_rss, enforce_resources(f"fold {fold} epoch {epoch + 1}"))
        record = {
            "fold": fold,
            "epoch": epoch + 1,
            "exposure": exposure,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "transport_delta_max_abs": float(model.transport_delta.detach().abs().max()),
            "mix_delta_max_abs": float(model.mix_delta.detach().abs().max()),
            "max_rss_bytes": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, max_rss


def score(
    model: DynamicMultiStateResidual,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
    *,
    exposure: float,
) -> dict[str, object]:
    layer_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[dict[str, float | int]] = []
    maximum_absolute = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            raw, sigma, weights, truth = load_batch(dataset, group, error)
            prediction = model(
                raw,
                sigma,
                weights,
                truth=truth if exposure < 1.0 else None,
                exposure=exposure,
            )
            assert isinstance(prediction, torch.Tensor)
            finite = finite and bool(torch.isfinite(prediction).all())
            maximum_absolute = max(maximum_absolute, float(prediction.abs().max()))
            squared = (prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            finals = squared[:, -1].mean(dim=1).cpu().numpy()
            per_network.extend(
                {
                    "role_rank": int(rank),
                    "mlp_id": int(dataset.mlp_ids[int(rank)]),
                    "final_mse": float(value),
                }
                for rank, value in zip(group, finals, strict=True)
            )
    layer_mse = layer_sse / (len(ranks) * WIDTH)
    return {
        "exposure": exposure,
        "networks": len(ranks),
        "finite_full_rollout": finite,
        "maximum_absolute_correction": maximum_absolute,
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "per_network": per_network,
    }


def fold_paths(fold: int) -> tuple[Path, Path]:
    return FOLD_DIR / f"fold_{fold}.json", FOLD_DIR / f"fold_{fold}.pt"


def run_fold(fold: int) -> dict[str, object]:
    if fold < 0 or fold >= FOLDS:
        raise ValueError(fold)
    seal = verify_seal()
    result_path, model_path = fold_paths(fold)
    if result_path.exists() or model_path.exists():
        if not result_path.is_file() or not model_path.is_file():
            raise FileExistsError("partial fold artifact collision")
        existing = read_json(result_path)
        if existing.get("program_sha256") != seal["program_sha256"]:
            raise AssertionError("completed fold belongs to another program")
        return existing
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = prepare_cache(dataset)
    valid = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    torch.manual_seed(SEED + fold)
    model = DynamicMultiStateResidual(centers, scales, layer_rms)
    cost = cost_record(model)
    if parameter_count(model) != 24_832:
        raise AssertionError(f"unexpected parameter count {parameter_count(model)}")
    if not cost["within_phase2_budget_by_upper_estimate"]:
        raise RuntimeError("frozen architecture misses the static cost gate")
    started = time.perf_counter()
    trace, max_rss = fit_fold(fold, model, dataset, train, error)
    autonomous = score(model, dataset, valid, error, exposure=1.0)
    teacher = score(model, dataset, valid, error, exposure=0.0)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-dynamic-multistate-fold-v1",
        "status": "complete",
        "fold": fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "architecture": {
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "parameters": parameter_count(model),
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
            "recurrence": "four Phi*(state@W) channels + dynamic local damping/mixing/source; channel zero is correction",
        },
        "training": {
            "epochs": EPOCHS,
            "exposure_schedule": list(EXPOSURE_SCHEDULE),
            "batch_size_networks": BATCH_SIZE,
            "trace": trace,
            "observed_max_rss_bytes": max_rss,
        },
        "host": {
            "final_mse": float(host_layer[-1]),
            "layer_mse": host_layer.tolist(),
        },
        "autonomous": autonomous,
        "teacher_forced": teacher,
        "cost": cost,
        "observer_selection_targets_opened": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "schema": "public-full688-dynamic-multistate-fold-model-v1",
        "fold": fold,
        "state_dict": model.state_dict(),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
    }
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.{os.getpid()}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(model_path)
    result["model_sha256"] = sha256(model_path)
    write_json_once(result_path, result)
    print(
        json.dumps(
            {
                "fold": fold,
                "autonomous_final_mse": autonomous["final_mse"],
                "teacher_final_mse": teacher["final_mse"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def aggregate() -> dict[str, object]:
    seal = verify_seal()
    if CV_RESULT.exists():
        return read_json(CV_RESULT)
    folds = []
    for fold in range(FOLDS):
        result_path, model_path = fold_paths(fold)
        if not result_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"fold {fold} is incomplete")
        record = read_json(result_path)
        if record.get("program_sha256") != seal["program_sha256"]:
            raise AssertionError(f"fold {fold}: program hash mismatch")
        if record.get("model_sha256") != sha256(model_path):
            raise AssertionError(f"fold {fold}: model hash mismatch")
        folds.append(record)
    total = sum(int(record["valid_networks"]) for record in folds)
    if total != 688:
        raise AssertionError(f"OOF coverage is {total}, expected 688")

    def weighted(section: str, key: str) -> float:
        return sum(
            int(record["valid_networks"]) * float(record[section][key])
            for record in folds
        ) / total

    def weighted_layers(section: str) -> list[float]:
        value = np.zeros(DEPTH, dtype=np.float64)
        for record in folds:
            value += int(record["valid_networks"]) * np.asarray(
                record[section]["layer_mse"], dtype=np.float64
            )
        return (value / total).tolist()

    autonomous_final = weighted("autonomous", "final_mse")
    teacher_final = weighted("teacher_forced", "final_mse")
    host_final = weighted("host", "final_mse")
    finite = all(bool(record["autonomous"]["finite_full_rollout"]) for record in folds)
    result: dict[str, object] = {
        "schema": "public-full688-dynamic-multistate-cv-v1",
        "status": "complete",
        "observer_fit_networks": total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "architecture": folds[0]["architecture"],
        "training": {
            "epochs": EPOCHS,
            "exposure_schedule": list(EXPOSURE_SCHEDULE),
            "fixed_final_checkpoint": True,
        },
        "oof": {
            "host_final_mse": host_final,
            "host_layer_mse": weighted_layers("host"),
            "autonomous_final_mse": autonomous_final,
            "autonomous_layer_mse": weighted_layers("autonomous"),
            "teacher_forced_final_mse": teacher_final,
            "teacher_forced_layer_mse": weighted_layers("teacher_forced"),
            "autonomous_ratio_to_host": autonomous_final / host_final,
            "teacher_ratio_to_host": teacher_final / host_final,
            "autonomous_teacher_gap": autonomous_final / teacher_final,
        },
        "decision_gates": {
            "finite_all_autonomous_rollouts": finite,
            "primary_below_3_123872816e_8": finite and autonomous_final < PRIMARY_GATE,
            "stretch_below_1e_8": finite and autonomous_final < STRETCH_GATE,
            "teacher_forced_below_1e_8": teacher_final < STRETCH_GATE,
        },
        "cost": folds[0]["cost"],
        "fold_results": [
            {
                "fold": record["fold"],
                "valid_networks": record["valid_networks"],
                "host_final_mse": record["host"]["final_mse"],
                "autonomous_final_mse": record["autonomous"]["final_mse"],
                "teacher_forced_final_mse": record["teacher_forced"]["final_mse"],
                "result_sha256": sha256(fold_paths(int(record["fold"]))[0]),
                "model_sha256": record["model_sha256"],
            }
            for record in folds
        ],
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
    }
    write_json_once(CV_RESULT, result)
    print(json.dumps(result["oof"], sort_keys=True), flush=True)
    return result


def smoke() -> None:
    seal = verify_seal()
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    centers = np.zeros((DEPTH, LOCAL_FEATURES - 1), dtype=np.float32)
    scales = np.ones_like(centers)
    rms = np.ones(DEPTH, dtype=np.float32)
    model = DynamicMultiStateResidual(centers, scales, rms)
    row = dataset.load(0)
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(np.asarray(row.raw_locals)[None].copy()),
            torch.from_numpy(np.asarray(row.sigma)[None].copy()),
            torch.from_numpy(
                np.asarray(
                    np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
                )[None].copy()
            ),
            exposure=1.0,
        )
    if tuple(prediction.shape) != (1, DEPTH, WIDTH):
        raise AssertionError("smoke output shape changed")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("smoke output is nonfinite")
    payload = {
        "readiness": readiness,
        "fold_sizes": [
            int(np.sum([fold_of(int(value)) == fold for value in dataset.mlp_ids]))
            for fold in range(FOLDS)
        ],
        "architecture": {
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "parameters": parameter_count(model),
            "output_shape": list(prediction.shape),
            "finite": True,
        },
        "cost": cost_record(model),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--fold", type=int)
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--cross-validate", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.prepare:
        verify_seal()
        dataset = PublicResidualDataset()
        error, sums, _, folds = prepare_cache(dataset)
        print(
            json.dumps(
                {
                    "ready": True,
                    "error_shape": list(error.shape),
                    "sums_shape": list(sums.shape),
                    "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif args.fold is not None:
        run_fold(args.fold)
    elif args.aggregate:
        aggregate()
    else:
        for fold in range(FOLDS):
            run_fold(fold)
        aggregate()


if __name__ == "__main__":
    main()
