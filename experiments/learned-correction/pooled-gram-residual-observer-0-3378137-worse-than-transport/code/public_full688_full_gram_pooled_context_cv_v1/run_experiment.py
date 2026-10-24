#!/Users/matt/anaconda3/bin/python3
"""P2R0F2K frozen full-688 full-Gram pooled-context CV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
J_PROGRAM = J_DIR / "run_experiment.py"
J_DECOMPOSITION = J_DIR / "OOF_DECOMPOSITION_AND_SUCCESSOR.json"
PARENT_CV_DIR = P2 / "public_full688_dynamic_multistate_residual_cv_v1"
LOADER_DIR = P2 / "public_full688_local_innovation_transport_v1"

_j_spec = importlib.util.spec_from_file_location("p2r0f2j_frozen_runner", J_PROGRAM)
if _j_spec is None or _j_spec.loader is None:
    raise ImportError(J_PROGRAM)
_j = importlib.util.module_from_spec(_j_spec)
_j_spec.loader.exec_module(_j)

DEPTH = _j.DEPTH
LOCAL_FEATURES = _j.LOCAL_FEATURES
WIDTH = _j.WIDTH
PublicResidualDataset = _j.PublicResidualDataset
fold_of = _j.fold_of
readiness_or_raise = _j.readiness_or_raise
load_shared_cache = _j.load_shared_cache
moments_for = _j.moments_for
load_batch = _j.load_batch


PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
FOLD_DIR = HERE / "folds"
CV_RESULT = HERE / "CV_RESULTS.json"
SHARED_CACHE_MANIFEST = PARENT_CV_DIR / "cache/MANIFEST.json"
SHARED_ERROR = PARENT_CV_DIR / "cache/patched_error.npy"
SHARED_STATS = PARENT_CV_DIR / "cache/raw_sufficient_stats.npz"

SEED = 1_753_358_438
FOLDS = 4
CHANNELS = 4
PAIR_INDEX = tuple(
    (left, right)
    for left in range(CHANNELS)
    for right in range(left + 1, CHANNELS)
)
PAIR_FEATURES = len(PAIR_INDEX)
LOCAL_INPUT_FEATURES = 42
LOCAL_HIDDEN = 96
CONTEXT_INPUT_FEATURES = 84
CONTEXT_HIDDEN = 32
FUSED_HIDDEN = LOCAL_HIDDEN + CONTEXT_HIDDEN
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
COHERENCE_FLOOR = 1.0e-4
PAIR_CORRELATION_FLOOR = 1.0e-8
POOL_RMS_FLOOR = 1.0e-8
MEAN_GATE = 2.9672528546945005e-8
MAX_NETWORK_GATE = 1.0e-7
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
HOST_FLOPS = 2_126_977_459_133
CORRECTION_FLOPS = 1_341_549_952
TOTAL_FLOPS = 2_128_319_009_085
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
    if sys.platform != "darwin":
        value *= 1024
    return value


def enforce_resources(where: str) -> int:
    rss = resident_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"RSS bound exceeded at {where}: {rss}")
    storage = sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())
    if storage > MAX_STORAGE_BYTES:
        raise OSError(f"storage bound exceeded at {where}: {storage}")
    return rss


def verify_seal() -> dict[str, object]:
    if not SEAL.is_file():
        raise FileNotFoundError("PREREGISTRATION_SEAL.json is required")
    seal = read_json(SEAL)
    if seal.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("seal status changed")
    bindings = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "loader_sha256": sha256(LOADER_DIR / "dataset.py"),
        "shared_cache_manifest_sha256": sha256(SHARED_CACHE_MANIFEST),
        "shared_error_sha256": sha256(SHARED_ERROR),
        "shared_stats_sha256": sha256(SHARED_STATS),
        "p2r0f2j_program_sha256": sha256(J_PROGRAM),
        "p2r0f2j_decomposition_sha256": sha256(J_DECOMPOSITION),
    }
    for key, value in bindings.items():
        if seal.get(key) != value:
            raise AssertionError(f"sealed binding changed: {key}")
    prereg = read_json(PREREGISTRATION)
    manifest = read_json(RUN_MANIFEST)
    if prereg.get("node") != "P2R0F2K":
        raise AssertionError("node binding changed")
    if prereg.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("preregistration status changed")
    if manifest.get("status") != "authorized-public-development-cv-held":
        raise AssertionError("run authorization changed")
    return seal


class FullGramPooledContextResidual(nn.Module):
    """Four signed transports, exact full Gram, and target-free width context."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))
        self.local_trunk = nn.Sequential(
            nn.Linear(LOCAL_INPUT_FEATURES, LOCAL_HIDDEN),
            nn.SiLU(),
            nn.Linear(LOCAL_HIDDEN, LOCAL_HIDDEN),
            nn.SiLU(),
        )
        self.context_trunk = nn.Sequential(
            nn.Linear(CONTEXT_INPUT_FEATURES, CONTEXT_HIDDEN),
            nn.SiLU(),
            nn.Linear(CONTEXT_HIDDEN, CONTEXT_HIDDEN),
            nn.SiLU(),
        )
        self.source_weight = nn.Parameter(
            torch.empty(DEPTH, FUSED_HIDDEN, CHANNELS)
        )
        self.source_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.gate_weight = nn.Parameter(
            torch.empty(DEPTH, FUSED_HIDDEN, CHANNELS)
        )
        self.gate_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.mix_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS, CHANNELS))
        nn.init.normal_(self.source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.gate_weight, mean=0.0, std=2.0e-3)

    def local_features(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        messages: torch.Tensor,
        energies: torch.Tensor,
        pair_correlation: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        rms = self.layer_rms[layer]
        rms2 = rms.square()
        message_rms = torch.clamp(messages / rms, -8.0, 8.0)
        message_sigma = torch.clamp(
            messages
            / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
            -8.0,
            8.0,
        )
        log_energy = torch.clamp(
            torch.log1p(torch.clamp(energies / rms2, min=0.0)), 0.0, 8.0
        )
        coherence = torch.clamp(
            messages / torch.sqrt(energies + COHERENCE_FLOOR * rms2),
            -8.0,
            8.0,
        )
        features = torch.cat(
            (
                raw_layer[..., :1],
                tail,
                message_rms,
                message_sigma,
                log_energy,
                coherence,
                pair_correlation,
            ),
            dim=-1,
        )
        if features.shape[-1] != LOCAL_INPUT_FEATURES:
            raise AssertionError(features.shape)
        return features

    def hidden_from_features(self, features: torch.Tensor) -> torch.Tensor:
        local = self.local_trunk(features)
        pooled_mean = features.mean(dim=1)
        pooled_rms = torch.sqrt(
            features.square().mean(dim=1) + POOL_RMS_FLOOR
        )
        pooled = torch.cat((pooled_mean, pooled_rms), dim=-1)
        context = self.context_trunk(pooled)
        broadcast = context[:, None, :].expand(-1, WIDTH, -1)
        return torch.cat((local, broadcast), dim=-1)

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        *,
        truth: torch.Tensor | None = None,
        exposure: float = 1.0,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        energy_history: list[torch.Tensor] = []
        pair_history: list[torch.Tensor] = []
        identity = torch.eye(CHANNELS, dtype=raw.dtype, device=raw.device)
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                states = torch.zeros_like(states)
                energies = torch.zeros_like(states)
                pair_correlation = torch.zeros(
                    batch,
                    WIDTH,
                    PAIR_FEATURES,
                    dtype=raw.dtype,
                    device=raw.device,
                )
                outputs.append(states[..., 0])
                state_history.append(states)
                energy_history.append(energies)
                pair_history.append(pair_correlation)
                continue
            if layer == FIRST_CORRECTED_LAYER:
                messages = torch.zeros_like(states)
                energies = torch.zeros_like(states)
                pair_correlation = torch.zeros(
                    batch,
                    WIDTH,
                    PAIR_FEATURES,
                    dtype=raw.dtype,
                    device=raw.device,
                )
            else:
                prior = states
                if exposure < 1.0:
                    primary = (
                        exposure * states[..., 0]
                        + (1.0 - exposure) * truth[:, layer - 1]
                    )
                    prior = torch.cat((primary[..., None], states[..., 1:]), dim=-1)
                layer_weights = weights[:, layer]
                weights_squared = layer_weights.square()
                messages = torch.bmm(
                    prior.transpose(1, 2), layer_weights
                ).transpose(1, 2)
                energies = torch.bmm(
                    prior.square().transpose(1, 2), weights_squared
                ).transpose(1, 2)
                pair_products = torch.stack(
                    [
                        prior[..., left] * prior[..., right]
                        for left, right in PAIR_INDEX
                    ],
                    dim=-1,
                )
                pair_gram = torch.bmm(
                    pair_products.transpose(1, 2), weights_squared
                ).transpose(1, 2)
                phi = raw[:, layer, :, 5, None]
                messages = phi * messages
                energies = phi.square() * energies
                pair_gram = phi.square() * pair_gram
                pair_denominator = torch.stack(
                    [
                        torch.sqrt(
                            energies[..., left] * energies[..., right]
                            + PAIR_CORRELATION_FLOOR
                            * self.layer_rms[layer].pow(4)
                        )
                        for left, right in PAIR_INDEX
                    ],
                    dim=-1,
                )
                pair_correlation = torch.clamp(
                    pair_gram / pair_denominator, -1.0, 1.0
                )
            features = self.local_features(
                raw[:, layer],
                sigma[:, layer],
                messages,
                energies,
                pair_correlation,
                layer,
            )
            hidden = self.hidden_from_features(features)
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
            energy_history.append(energies)
            pair_history.append(pair_correlation)
        prediction = torch.stack(outputs, dim=1)
        if return_states:
            return (
                prediction,
                torch.stack(state_history, dim=1),
                torch.stack(energy_history, dim=1),
                torch.stack(pair_history, dim=1),
            )
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def cost_record(model: nn.Module) -> dict[str, object]:
    corrected = DEPTH - FIRST_CORRECTED_LAYER
    transitions = corrected - 1
    dense_vector_matrix = 2 * WIDTH * WIDTH - WIDTH
    mean_energy_and_full_gram = (
        transitions * (CHANNELS + CHANNELS * (CHANNELS + 1) // 2) * dense_vector_matrix
    )
    weights_squared_materialization = transitions * WIDTH * WIDTH
    raw20_materialization_reserve = DEPTH * 16 * WIDTH * WIDTH
    raw20_standardization = 2 * LOCAL_FEATURES * DEPTH * WIDTH
    local_fused_special = corrected * WIDTH * 47_032
    pool_context_broadcast = corrected * 218_944
    output_addition = DEPTH * WIDTH
    correction = (
        mean_energy_and_full_gram
        + weights_squared_materialization
        + raw20_materialization_reserve
        + raw20_standardization
        + local_fused_special
        + pool_context_broadcast
        + output_addition
    )
    total = HOST_FLOPS + correction
    if mean_energy_and_full_gram != 381_495_296:
        raise AssertionError(mean_energy_and_full_gram)
    if weights_squared_materialization != 13_631_488:
        raise AssertionError(weights_squared_materialization)
    if local_fused_special != 674_250_752:
        raise AssertionError(local_fused_special)
    if pool_context_broadcast != 3_065_216:
        raise AssertionError(pool_context_broadcast)
    if correction != CORRECTION_FLOPS or total != TOTAL_FLOPS:
        raise AssertionError((correction, total))
    return {
        "parameter_count": parameter_count(model),
        "measured_host_flops": HOST_FLOPS,
        "mean_energy_and_full_gram_transport_upper_estimate": mean_energy_and_full_gram,
        "reused_w_squared_materialization_reserve": weights_squared_materialization,
        "raw20_materialization_reserve": raw20_materialization_reserve,
        "raw20_standardization": raw20_standardization,
        "local_fused_heads_and_special_reserve": local_fused_special,
        "width_pool_context_and_broadcast_reserve": pool_context_broadcast,
        "output_addition": output_addition,
        "correction_source_structure_upper_estimate": correction,
        "host_plus_correction_upper_estimate": total,
        "phase2_budget": PHASE2_BUDGET,
        "headroom_under_upper_estimate": PHASE2_BUDGET - total,
        "within_phase2_budget_by_upper_estimate": total <= PHASE2_BUDGET,
        "authority": "frozen conservative source-structure admission estimate; literal meter only after accuracy pass",
    }


def fit_fold(
    fold: int,
    model: FullGramPooledContextResidual,
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
    max_rss = enforce_resources("training start")
    started = time.perf_counter()
    for epoch, exposure in enumerate(EXPOSURE_SCHEDULE, start=1):
        order = order_rng.permutation(train)
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            raw, sigma, weights, truth = load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states, _, _ = model(
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
                raise FloatingPointError(f"fold {fold} epoch {epoch}: nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        max_rss = max(max_rss, enforce_resources(f"fold {fold} epoch {epoch}"))
        record = {
            "fold": fold,
            "epoch": epoch,
            "own_primary_state_exposure": exposure,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "transport_delta_max_abs": float(model.transport_delta.detach().abs().max()),
            "mix_delta_max_abs": float(model.mix_delta.detach().abs().max()),
            "context_weight_max_abs": max(
                float(value.detach().abs().max())
                for value in model.context_trunk.parameters()
            ),
            "max_rss_bytes": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, max_rss


def score(
    model: FullGramPooledContextResidual,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
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
            prediction = model(raw, sigma, weights, exposure=1.0)
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
    error, sums, sums2, folds = load_shared_cache(dataset)
    valid = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    torch.manual_seed(SEED + fold)
    model = FullGramPooledContextResidual(centers, scales, layer_rms)
    if parameter_count(model) != 34_048:
        raise AssertionError(f"unexpected parameter count {parameter_count(model)}")
    cost = cost_record(model)
    if not cost["within_phase2_budget_by_upper_estimate"]:
        raise RuntimeError("frozen full-Gram pooled-context model misses cost admission")
    started = time.perf_counter()
    trace, max_rss = fit_fold(fold, model, dataset, train, error)
    autonomous = score(model, dataset, valid, error)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-full-gram-pooled-context-fold-v1",
        "status": "complete",
        "node": "P2R0F2K",
        "fold": fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "architecture": {
            "channels": CHANNELS,
            "offdiagonal_pair_features": PAIR_FEATURES,
            "local_input_features": LOCAL_INPUT_FEATURES,
            "local_hidden": LOCAL_HIDDEN,
            "context_input_features": CONTEXT_INPUT_FEATURES,
            "context_hidden": CONTEXT_HIDDEN,
            "parameters": parameter_count(model),
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
            "signed_transport": "Phi*(s@W)",
            "full_gram_observable": "Phi^2*((s_c*s_d)@W^2)",
            "deployment_truth_inputs": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "own_primary_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "trace": trace,
            "observed_max_rss_bytes": max_rss,
        },
        "host": {
            "final_mse": float(host_layer[-1]),
            "layer_mse": host_layer.tolist(),
        },
        "autonomous": autonomous,
        "cost": cost,
        "observer_selection_targets_opened": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "schema": "public-full688-full-gram-pooled-context-fold-model-v1",
        "node": "P2R0F2K",
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
                "maximum_per_network_final_mse": max(
                    row["final_mse"] for row in autonomous["per_network"]
                ),
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
    records = []
    for fold in range(FOLDS):
        result_path, model_path = fold_paths(fold)
        if not result_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"fold {fold} is incomplete")
        record = read_json(result_path)
        if record.get("program_sha256") != seal["program_sha256"]:
            raise AssertionError(f"fold {fold}: program hash mismatch")
        if record.get("model_sha256") != sha256(model_path):
            raise AssertionError(f"fold {fold}: model hash mismatch")
        records.append(record)
    total = sum(int(record["valid_networks"]) for record in records)
    valid_union = [
        int(rank) for record in records for rank in record["valid_role_ranks"]
    ]
    if total != 688 or sorted(valid_union) != list(range(688)):
        raise AssertionError("OOF coverage is not exactly 688 disjoint networks")

    def weighted(section: str, key: str) -> float:
        return sum(
            int(record["valid_networks"]) * float(record[section][key])
            for record in records
        ) / total

    def weighted_layers(section: str) -> list[float]:
        value = np.zeros(DEPTH, dtype=np.float64)
        for record in records:
            value += int(record["valid_networks"]) * np.asarray(
                record[section]["layer_mse"], dtype=np.float64
            )
        return (value / total).tolist()

    autonomous_final = weighted("autonomous", "final_mse")
    host_final = weighted("host", "final_mse")
    per_network = [
        row for record in records for row in record["autonomous"]["per_network"]
    ]
    maximum_row = max(per_network, key=lambda row: float(row["final_mse"]))
    maximum_network = float(maximum_row["final_mse"])
    finite = all(record["autonomous"]["finite_full_rollout"] for record in records)
    mean_pass = autonomous_final < MEAN_GATE
    maximum_pass = maximum_network < MAX_NETWORK_GATE
    result: dict[str, object] = {
        "schema": "public-full688-full-gram-pooled-context-cv-v1",
        "status": "complete",
        "node": "P2R0F2K",
        "observer_fit_networks": total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "architecture": records[0]["architecture"],
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "own_primary_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "fixed_final_checkpoint": True,
        },
        "oof": {
            "host_final_mse": host_final,
            "host_layer_mse": weighted_layers("host"),
            "autonomous_final_mse": autonomous_final,
            "autonomous_layer_mse": weighted_layers("autonomous"),
            "autonomous_ratio_to_host": autonomous_final / host_final,
            "maximum_per_network_final_mse": maximum_network,
            "maximum_role_rank": int(maximum_row["role_rank"]),
            "maximum_mlp_id": int(maximum_row["mlp_id"]),
        },
        "decision_gate": {
            "finite_all_autonomous_rollouts": finite,
            "mean_below_2_9672528546945005e_8": mean_pass,
            "maximum_per_network_below_1e_7": maximum_pass,
            "conjunctive_pass": finite and mean_pass and maximum_pass,
        },
        "cost": records[0]["cost"],
        "fold_results": [
            {
                "fold": record["fold"],
                "valid_networks": record["valid_networks"],
                "host_final_mse": record["host"]["final_mse"],
                "autonomous_final_mse": record["autonomous"]["final_mse"],
                "maximum_per_network_final_mse": max(
                    row["final_mse"] for row in record["autonomous"]["per_network"]
                ),
                "result_sha256": sha256(fold_paths(int(record["fold"]))[0]),
                "model_sha256": record["model_sha256"],
            }
            for record in records
        ],
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
    }
    write_json_once(CV_RESULT, result)
    print(json.dumps({"oof": result["oof"], "gate": result["decision_gate"]}, sort_keys=True))
    return result


def smoke() -> None:
    seal = verify_seal()
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    train = np.flatnonzero(folds != 0)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    model = FullGramPooledContextResidual(centers, scales, layer_rms)
    row = dataset.load(0)
    with torch.no_grad():
        prediction, states, energies, pair_correlation = model(
            torch.from_numpy(np.asarray(row.raw_locals)[None].copy()),
            torch.from_numpy(np.asarray(row.sigma)[None].copy()),
            torch.from_numpy(
                np.asarray(
                    np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
                )[None].copy()
            ),
            exposure=1.0,
            return_states=True,
        )
    if tuple(prediction.shape) != (1, DEPTH, WIDTH):
        raise AssertionError("smoke output shape changed")
    if tuple(states.shape) != (1, DEPTH, WIDTH, CHANNELS):
        raise AssertionError("smoke state shape changed")
    if tuple(energies.shape) != (1, DEPTH, WIDTH, CHANNELS):
        raise AssertionError("smoke energy shape changed")
    if tuple(pair_correlation.shape) != (1, DEPTH, WIDTH, PAIR_FEATURES):
        raise AssertionError("smoke pair-correlation shape changed")
    tensors = (prediction, states, energies, pair_correlation)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("smoke output is nonfinite")
    if not bool((energies >= 0.0).all()):
        raise FloatingPointError("diagonal contribution energy became negative")
    if not bool((pair_correlation.abs() <= 1.0).all()):
        raise FloatingPointError("pair correlation escaped its bound")
    if model.local_trunk[0].in_features != LOCAL_INPUT_FEATURES:
        raise AssertionError("local input width changed")
    if model.context_trunk[0].in_features != CONTEXT_INPUT_FEATURES:
        raise AssertionError("context input width changed")
    if parameter_count(model) != 34_048:
        raise AssertionError("parameter count changed")
    payload = {
        "node": "P2R0F2K",
        "launch_status": "held_after_smoke_pending_d4_or_root_release",
        "readiness": readiness,
        "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        "architecture": {
            "channels": CHANNELS,
            "offdiagonal_pair_features": PAIR_FEATURES,
            "local_input_features": LOCAL_INPUT_FEATURES,
            "context_input_features": CONTEXT_INPUT_FEATURES,
            "parameters": parameter_count(model),
            "output_shape": list(prediction.shape),
            "state_shape": list(states.shape),
            "energy_shape": list(energies.shape),
            "pair_correlation_shape": list(pair_correlation.shape),
            "finite": True,
            "energy_nonnegative": True,
            "pair_correlation_bounded": True,
        },
        "cost": cost_record(model),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "observer_selection_targets_opened": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fold", type=int)
    mode.add_argument("--aggregate", action="store_true")
    mode.add_argument("--cross-validate", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
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
