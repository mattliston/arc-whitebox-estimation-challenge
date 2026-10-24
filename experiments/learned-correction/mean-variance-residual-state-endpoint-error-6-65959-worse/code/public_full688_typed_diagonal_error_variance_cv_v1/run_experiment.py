#!/Users/matt/anaconda3/bin/python3
"""P2R0F2I frozen full-688 typed diagonal-error-variance CV."""

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
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PARENT_CV_DIR = P2 / "public_full688_dynamic_multistate_residual_cv_v1"
LOADER_DIR = P2 / "public_full688_local_innovation_transport_v1"
sys.path.insert(0, str(LOADER_DIR))
from dataset import DEPTH, LOCAL_FEATURES, WIDTH, PublicResidualDataset  # noqa: E402


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
HIDDEN = 96
TYPED_FEATURES = 4
INPUT_FEATURES = LOCAL_FEATURES + TYPED_FEATURES
FIRST_CORRECTED_LAYER = 2
THREADS = 2
BATCH_SIZE = 3
EPOCHS = 6
EXPOSURE_SCHEDULE = (0.0, 0.25, 0.50, 0.75, 1.0, 1.0)
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
VARIANCE_LOSS_WEIGHT = 0.05
VARIANCE_SCORE_FLOOR = 1.0e-4
VARIANCE_STATE_FLOOR = 1.0e-8
VARIANCE_STATE_CAP = 64.0
VARIANCE_BIAS_INITIAL = -4.59511985013459
MEAN_GATE = 2.9672528546945005e-8
MAX_NETWORK_GATE = 1.0e-7
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
HOST_FLOPS = 2_126_977_459_133
CORRECTION_FLOPS = 842_887_168
TOTAL_FLOPS = 2_127_820_346_301
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
    }
    for key, value in bindings.items():
        if seal.get(key) != value:
            raise AssertionError(f"sealed binding changed: {key}")
    prereg = read_json(PREREGISTRATION)
    manifest = read_json(RUN_MANIFEST)
    if prereg.get("node") != "P2R0F2I":
        raise AssertionError("node binding changed")
    if prereg.get("status") != "frozen-before-cv-metrics":
        raise AssertionError("preregistration status changed")
    if manifest.get("status") != "authorized-public-development-cv":
        raise AssertionError("run authorization changed")
    return seal


def fold_of(mlp_id: int) -> int:
    payload = f"{SEED}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % FOLDS


def readiness_or_raise(dataset: PublicResidualDataset) -> dict[str, object]:
    readiness = dataset.readiness()
    expected_tails = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
    if not readiness["ready"]:
        raise RuntimeError(f"observer_fit host is not ready: {readiness}")
    if readiness["available"] != 688 or readiness["missing"]:
        raise AssertionError("base scan coverage changed")
    if readiness["data_derived_tail_ranks"] != expected_tails:
        raise AssertionError("tail roster changed")
    if readiness["repaired_tail_ranks"] != expected_tails:
        raise AssertionError("tail repair coverage changed")
    if readiness["unpatched_tail_ranks"]:
        raise AssertionError("an old-trigger tail is unpatched")
    return readiness


def load_shared_cache(
    dataset: PublicResidualDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    readiness_or_raise(dataset)
    error = np.load(SHARED_ERROR, mmap_mode="r", allow_pickle=False)
    payload = np.load(SHARED_STATS, allow_pickle=False)
    sums = payload["sum"]
    sums2 = payload["sum2"]
    if error.dtype != np.float32 or error.shape != (688, DEPTH, WIDTH):
        raise AssertionError("shared error schema changed")
    expected_stats = (688, DEPTH, LOCAL_FEATURES - 1)
    if sums.dtype != np.float64 or sums.shape != expected_stats:
        raise AssertionError("shared sum schema changed")
    if sums2.dtype != np.float64 or sums2.shape != expected_stats:
        raise AssertionError("shared sum2 schema changed")
    folds = np.asarray([fold_of(int(value)) for value in dataset.mlp_ids], dtype=np.int64)
    if [int(np.sum(folds == fold)) for fold in range(FOLDS)] != [147, 172, 171, 198]:
        raise AssertionError("fold assignment changed")
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


class TypedDiagonalVarianceObserver(nn.Module):
    """One signed mean transport and one nonnegative diagonal variance transport."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))
        self.trunk = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.source_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.source_bias = nn.Parameter(torch.zeros(DEPTH))
        self.gain_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.gain_bias = nn.Parameter(torch.zeros(DEPTH))
        self.variance_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN))
        self.variance_bias = nn.Parameter(
            torch.full((DEPTH,), VARIANCE_BIAS_INITIAL)
        )
        nn.init.normal_(self.source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.gain_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.variance_weight, mean=0.0, std=2.0e-3)

    def local_hidden(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        mean_message: torch.Tensor,
        variance_message: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        rms = self.layer_rms[layer]
        rms2 = rms.square()
        message_rms = torch.clamp(mean_message / rms, -8.0, 8.0)
        message_sigma = torch.clamp(
            mean_message
            / torch.clamp(sigma_layer * rms, min=2.0**-30),
            -8.0,
            8.0,
        )
        log_variance = torch.clamp(
            torch.log1p(torch.clamp(variance_message / rms2, min=0.0)),
            0.0,
            8.0,
        )
        coherence = torch.clamp(
            mean_message
            / torch.sqrt(variance_message + VARIANCE_SCORE_FLOOR * rms2),
            -8.0,
            8.0,
        )
        features = torch.cat(
            (
                raw_layer[..., :1],
                tail,
                message_rms[..., None],
                message_sigma[..., None],
                log_variance[..., None],
                coherence[..., None],
            ),
            dim=-1,
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
        return_variance: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= exposure <= 1.0:
            raise ValueError(exposure)
        if exposure < 1.0 and truth is None:
            raise ValueError("truth is required for scheduled mean-state exposure")
        batch = raw.shape[0]
        mean_state = torch.zeros(
            batch, WIDTH, dtype=raw.dtype, device=raw.device
        )
        variance_state = torch.zeros_like(mean_state)
        outputs: list[torch.Tensor] = []
        variance_history: list[torch.Tensor] = []
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                mean_state = torch.zeros_like(mean_state)
                variance_state = torch.zeros_like(variance_state)
                outputs.append(mean_state)
                variance_history.append(variance_state)
                continue
            if layer == FIRST_CORRECTED_LAYER:
                mean_message = torch.zeros_like(mean_state)
                variance_message = torch.zeros_like(variance_state)
            else:
                prior_mean = mean_state
                if exposure < 1.0:
                    prior_mean = (
                        exposure * mean_state
                        + (1.0 - exposure) * truth[:, layer - 1]
                    )
                layer_weights = weights[:, layer]
                mean_message = torch.bmm(
                    prior_mean[:, None, :], layer_weights
                ).squeeze(1)
                phi = raw[:, layer, :, 5]
                mean_message = phi * mean_message
                variance_message = torch.bmm(
                    variance_state[:, None, :], layer_weights.square()
                ).squeeze(1)
                variance_message = phi.square() * variance_message
            hidden = self.local_hidden(
                raw[:, layer],
                sigma[:, layer],
                mean_message,
                variance_message,
                layer,
            )
            source_standard = torch.einsum(
                "bwh,h->bw", hidden, self.source_weight[layer]
            ) + self.source_bias[layer]
            gain_standard = torch.einsum(
                "bwh,h->bw", hidden, self.gain_weight[layer]
            ) + self.gain_bias[layer]
            variance_standard = torch.einsum(
                "bwh,h->bw", hidden, self.variance_weight[layer]
            ) + self.variance_bias[layer]
            gain = 1.0 + 0.25 * torch.tanh(gain_standard)
            physical_scale = sigma[:, layer] * self.layer_rms[layer]
            mean_state = gain * mean_message + physical_scale * source_standard
            variance_state = (
                gain.square() * variance_message
                + physical_scale.square() * F.softplus(variance_standard)
            )
            rms2 = self.layer_rms[layer].square()
            variance_state = torch.minimum(
                torch.maximum(variance_state, VARIANCE_STATE_FLOOR * rms2),
                VARIANCE_STATE_CAP * rms2,
            )
            outputs.append(mean_state)
            variance_history.append(variance_state)
        prediction = torch.stack(outputs, dim=1)
        variance = torch.stack(variance_history, dim=1)
        if return_variance:
            return prediction, variance
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def cost_record(model: nn.Module) -> dict[str, object]:
    corrected = DEPTH - FIRST_CORRECTED_LAYER
    transitions = corrected - 1
    two_typed_transports = 2 * transitions * (2 * WIDTH * WIDTH - WIDTH)
    raw20_materialization_reserve = DEPTH * 16 * WIDTH * WIDTH
    raw20_standardization = 2 * LOCAL_FEATURES * DEPTH * WIDTH
    per_unit = (
        (2 * INPUT_FEATURES - 1) * HIDDEN
        + 64 * HIDDEN
        + (2 * HIDDEN - 1) * HIDDEN
        + 64 * HIDDEN
        + (2 * HIDDEN - 1) * 3
        + 512
    )
    local_cell = corrected * WIDTH * per_unit
    output_and_state_addition = 2 * DEPTH * WIDTH
    correction = (
        two_typed_transports
        + raw20_materialization_reserve
        + raw20_standardization
        + local_cell
        + output_and_state_addition
    )
    total = HOST_FLOPS + correction
    if correction != CORRECTION_FLOPS or total != TOTAL_FLOPS:
        raise AssertionError((correction, total))
    return {
        "parameter_count": parameter_count(model),
        "measured_host_flops": HOST_FLOPS,
        "mean_plus_variance_transport": two_typed_transports,
        "raw20_materialization_reserve": raw20_materialization_reserve,
        "raw20_standardization": raw20_standardization,
        "local_cell_dense_plus_special_reserve": local_cell,
        "output_and_state_addition": output_and_state_addition,
        "correction_source_structure_upper_estimate": correction,
        "host_plus_correction_upper_estimate": total,
        "phase2_budget": PHASE2_BUDGET,
        "headroom_under_upper_estimate": PHASE2_BUDGET - total,
        "within_phase2_budget_by_upper_estimate": total <= PHASE2_BUDGET,
        "authority": "frozen conservative source-structure admission estimate; literal meter only after accuracy pass",
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
    model: TypedDiagonalVarianceObserver,
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
        primary_sum = 0.0
        variance_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            raw, sigma, weights, truth = load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, variance = model(
                raw,
                sigma,
                weights,
                truth=truth,
                exposure=exposure,
                return_variance=True,
            )
            normalized = (
                prediction[:, FIRST_CORRECTED_LAYER:]
                - truth[:, FIRST_CORRECTED_LAYER:]
            ) / model.layer_rms[FIRST_CORRECTED_LAYER:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            rollout = torch.mean(layer_weights * layer_mse)
            primary = rollout + layer_mse[-1]
            standard_variance = (
                variance[:, FIRST_CORRECTED_LAYER:]
                / model.layer_rms[FIRST_CORRECTED_LAYER:].square()[None, :, None]
            )
            variance_score = torch.mean(
                torch.log(standard_variance + VARIANCE_SCORE_FLOOR)
                + normalized.square()
                / (standard_variance + VARIANCE_SCORE_FLOOR)
            )
            loss = primary + VARIANCE_LOSS_WEIGHT * variance_score
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"fold {fold} epoch {epoch}: nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            primary_sum += float(primary.detach()) * len(ranks)
            variance_sum += float(variance_score.detach()) * len(ranks)
            examples += len(ranks)
        max_rss = max(max_rss, enforce_resources(f"fold {fold} epoch {epoch}"))
        record = {
            "fold": fold,
            "epoch": epoch,
            "own_mean_state_exposure": exposure,
            "fit_objective": objective_sum / examples,
            "primary_objective": primary_sum / examples,
            "variance_score": variance_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "source_weight_max_abs": float(model.source_weight.detach().abs().max()),
            "gain_weight_max_abs": float(model.gain_weight.detach().abs().max()),
            "variance_weight_max_abs": float(model.variance_weight.detach().abs().max()),
            "max_rss_bytes": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, max_rss


def score(
    model: TypedDiagonalVarianceObserver,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> dict[str, object]:
    layer_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[dict[str, float | int]] = []
    maximum_absolute = 0.0
    maximum_variance = 0.0
    finite = True
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            raw, sigma, weights, truth = load_batch(dataset, group, error)
            prediction, variance = model(
                raw, sigma, weights, exposure=1.0, return_variance=True
            )
            finite = finite and bool(torch.isfinite(prediction).all())
            finite = finite and bool(torch.isfinite(variance).all())
            finite = finite and bool((variance >= 0.0).all())
            maximum_absolute = max(maximum_absolute, float(prediction.abs().max()))
            maximum_variance = max(maximum_variance, float(variance.max()))
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
        "maximum_variance_state": maximum_variance,
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
    model = TypedDiagonalVarianceObserver(centers, scales, layer_rms)
    if parameter_count(model) != 16_368:
        raise AssertionError(f"unexpected parameter count {parameter_count(model)}")
    cost = cost_record(model)
    if not cost["within_phase2_budget_by_upper_estimate"]:
        raise RuntimeError("frozen typed-state model misses cost admission")
    started = time.perf_counter()
    trace, max_rss = fit_fold(fold, model, dataset, train, error)
    autonomous = score(model, dataset, valid, error)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-typed-diagonal-error-variance-fold-v1",
        "status": "complete",
        "node": "P2R0F2I",
        "fold": fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "architecture": {
            "hidden": HIDDEN,
            "raw_features": LOCAL_FEATURES,
            "typed_state_features": TYPED_FEATURES,
            "input_features": INPUT_FEATURES,
            "parameters": parameter_count(model),
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
            "mean_transport": "Phi*(mu@W)",
            "variance_transport": "Phi^2*(v@W^2)",
            "deployment_truth_inputs": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "own_mean_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "variance_state_exposure": "fully autonomous",
            "variance_loss_weight": VARIANCE_LOSS_WEIGHT,
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
        "schema": "public-full688-typed-diagonal-error-variance-fold-model-v1",
        "node": "P2R0F2I",
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
        "schema": "public-full688-typed-diagonal-error-variance-cv-v1",
        "status": "complete",
        "node": "P2R0F2I",
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
            "own_mean_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "variance_state_exposure": "fully autonomous",
            "variance_loss_weight": VARIANCE_LOSS_WEIGHT,
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
    model = TypedDiagonalVarianceObserver(centers, scales, layer_rms)
    row = dataset.load(0)
    with torch.no_grad():
        prediction, variance = model(
            torch.from_numpy(np.asarray(row.raw_locals)[None].copy()),
            torch.from_numpy(np.asarray(row.sigma)[None].copy()),
            torch.from_numpy(
                np.asarray(
                    np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
                )[None].copy()
            ),
            exposure=1.0,
            return_variance=True,
        )
    expected_shape = (1, DEPTH, WIDTH)
    if tuple(prediction.shape) != expected_shape or tuple(variance.shape) != expected_shape:
        raise AssertionError("smoke output shape changed")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(variance).all()):
        raise FloatingPointError("smoke output is nonfinite")
    if not bool((variance >= 0.0).all()):
        raise FloatingPointError("variance state became negative")
    if model.trunk[0].in_features != INPUT_FEATURES:
        raise AssertionError("typed input width changed")
    if parameter_count(model) != 16_368:
        raise AssertionError("parameter count changed")
    payload = {
        "node": "P2R0F2I",
        "readiness": readiness,
        "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        "architecture": {
            "hidden": HIDDEN,
            "raw_features": LOCAL_FEATURES,
            "typed_state_features": TYPED_FEATURES,
            "input_features": INPUT_FEATURES,
            "parameters": parameter_count(model),
            "output_shape": list(prediction.shape),
            "variance_shape": list(variance.shape),
            "finite": True,
            "variance_nonnegative": True,
        },
        "cost": cost_record(model),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
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
