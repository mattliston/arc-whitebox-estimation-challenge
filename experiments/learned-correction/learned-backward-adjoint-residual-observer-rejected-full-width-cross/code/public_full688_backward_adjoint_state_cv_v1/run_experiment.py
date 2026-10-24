#!/Users/matt/anaconda3/bin/python3
"""P2R0F2M frozen full-688 learned backward-adjoint state CV."""

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
K_DIR = P2 / "public_full688_full_gram_pooled_context_cv_v1"
DECOMPOSITION = K_DIR / "K_VS_J_OOF_DECOMPOSITION_AND_SUCCESSOR.json"
DESIGN_DIR = P2 / "public_full688_bidirectional_adjoint_smoother_design_v1"
DESIGN_RECEIPT = DESIGN_DIR / "DESIGN_RECEIPT.json"
ADJOINT_DIAGNOSTIC = DESIGN_DIR / "DIAGNOSTIC_RESULT.json"
NATIVE_DIR = P2 / "public_q24_native_transport_energy_pilot_v1"
NATIVE_METER_RESULT = NATIVE_DIR / "runs/native_meter_rank0035_v1/RESULT.json"
NATIVE_PILOT_PREREGISTRATION = NATIVE_DIR / "PILOT_PREREGISTRATION.json"

_spec = importlib.util.spec_from_file_location("p2r0f2j_frozen_runner", J_PROGRAM)
if _spec is None or _spec.loader is None:
    raise ImportError(J_PROGRAM)
j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j)

DEPTH = j.DEPTH
LOCAL_FEATURES = j.LOCAL_FEATURES
WIDTH = j.WIDTH
PublicResidualDataset = j.PublicResidualDataset
fold_of = j.fold_of
readiness_or_raise = j.readiness_or_raise
load_shared_cache = j.load_shared_cache
moments_for = j.moments_for
load_batch = j.load_batch

PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
TRAINING_RELEASE = HERE / "TRAINING_RELEASE.json"
FOLD_DIR = HERE / "folds"
CV_RESULT = HERE / "CV_RESULTS.json"
SHARED_CACHE_MANIFEST = j.SHARED_CACHE_MANIFEST
SHARED_ERROR = j.SHARED_ERROR
SHARED_STATS = j.SHARED_STATS
LOADER_DIR = j.LOADER_DIR

SEED = 1_753_358_438
FOLDS = 4
CHANNELS = 4
FORWARD_INPUT_FEATURES = LOCAL_FEATURES + 6 * CHANNELS
FORWARD_HIDDEN = 96
BACKWARD_INPUT_FEATURES = LOCAL_FEATURES + 2 * CHANNELS
BACKWARD_HIDDEN = 64
FIRST_CORRECTED_LAYER = 2
CORRECTED_LAYERS = DEPTH - FIRST_CORRECTED_LAYER
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
MEAN_GATE = 2.9672528546945005e-8
MAX_NETWORK_GATE = 1.0e-7
STRETCH_MSE = 1.0e-8
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
HOST_FLOPS = 2_126_977_459_133
J_CORRECTION_FLOPS = 1_051_656_192
ADDED_BACKWARD_ADJOINT_FLOPS = 468_946_944
CORRECTION_FLOPS = 1_520_603_136
TOTAL_FLOPS = 2_128_498_062_269
PHASE2_BUDGET = 2_200_000_000_000
EXACT_SCORING_BUDGET = 2**41
NATIVE_HOST_TAPE_BASELINE_FLOPS = 2_131_692_137_721


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
    if seal.get("status") != "frozen-before-cv-metrics-native-transfer-hold":
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
        "p2r0f2j_cv_results_sha256": sha256(J_DIR / "CV_RESULTS.json"),
        "p2r0f2k_cv_results_sha256": sha256(K_DIR / "CV_RESULTS.json"),
        "decomposition_sha256": sha256(DECOMPOSITION),
        "design_receipt_sha256": sha256(DESIGN_RECEIPT),
        "adjoint_diagnostic_sha256": sha256(ADJOINT_DIAGNOSTIC),
        "native_meter_result_sha256": sha256(NATIVE_METER_RESULT),
        "native_pilot_preregistration_sha256": sha256(
            NATIVE_PILOT_PREREGISTRATION
        ),
    }
    for key, value in bindings.items():
        if seal.get(key) != value:
            raise AssertionError(f"sealed binding changed: {key}")
    prereg = read_json(PREREGISTRATION)
    manifest = read_json(RUN_MANIFEST)
    if prereg.get("node") != "P2R0F2M":
        raise AssertionError("node binding changed")
    if prereg.get("status") != "frozen-before-cv-metrics-native-transfer-hold":
        raise AssertionError("preregistration status changed")
    if manifest.get("status") != "authorized-smoke-only-native-transfer-hold":
        raise AssertionError("run authorization changed")
    return seal


def training_release_or_raise() -> dict[str, object]:
    """Require a root receipt bound to a terminal native-Q24 pilot result."""

    if not TRAINING_RELEASE.is_file():
        raise RuntimeError(
            "P2R0F2M training is held until a post-native-transfer root release"
        )
    release = read_json(TRAINING_RELEASE)
    if release.get("node") != "P2R0F2M":
        raise AssertionError("training release node changed")
    if release.get("status") != "authorized-by-root-after-native-transfer-terminal":
        raise AssertionError("training release status changed")
    if release.get("root_release") is not True:
        raise AssertionError("training release lacks root authority")
    relative = release.get("native_transfer_terminal_result")
    if not isinstance(relative, str):
        raise AssertionError("native transfer terminal result path missing")
    result_path = (NATIVE_DIR / relative).resolve()
    if NATIVE_DIR.resolve() not in result_path.parents:
        raise AssertionError("native transfer result escaped its experiment")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    if release.get("native_transfer_terminal_result_sha256") != sha256(result_path):
        raise AssertionError("native transfer terminal result binding changed")
    if release.get("capacity_only_firewall_acknowledged") is not True:
        raise AssertionError("capacity-only firewall was not acknowledged")
    return release


class BackwardAdjointStateResidual(nn.Module):
    """Target-free learned backward adjoint conditioning a physical forward state."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))

        # Define the forward J-descended blocks first, then the backward blocks.
        self.forward_trunk = nn.Sequential(
            nn.Linear(FORWARD_INPUT_FEATURES, FORWARD_HIDDEN),
            nn.SiLU(),
            nn.Linear(FORWARD_HIDDEN, FORWARD_HIDDEN),
            nn.SiLU(),
        )
        self.forward_source_weight = nn.Parameter(
            torch.empty(DEPTH, FORWARD_HIDDEN, CHANNELS)
        )
        self.forward_source_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.forward_gate_weight = nn.Parameter(
            torch.empty(DEPTH, FORWARD_HIDDEN, CHANNELS)
        )
        self.forward_gate_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.forward_transport_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.forward_mix_delta = nn.Parameter(
            torch.zeros(DEPTH, CHANNELS, CHANNELS)
        )
        nn.init.normal_(self.forward_source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.forward_gate_weight, mean=0.0, std=2.0e-3)

        self.backward_trunk = nn.Sequential(
            nn.Linear(BACKWARD_INPUT_FEATURES, BACKWARD_HIDDEN),
            nn.SiLU(),
            nn.Linear(BACKWARD_HIDDEN, BACKWARD_HIDDEN),
            nn.SiLU(),
        )
        self.backward_source_weight = nn.Parameter(
            torch.empty(CORRECTED_LAYERS, BACKWARD_HIDDEN, CHANNELS)
        )
        self.backward_source_bias = nn.Parameter(
            torch.zeros(CORRECTED_LAYERS, CHANNELS)
        )
        self.backward_gate_weight = nn.Parameter(
            torch.empty(CORRECTED_LAYERS, BACKWARD_HIDDEN, CHANNELS)
        )
        self.backward_gate_bias = nn.Parameter(
            torch.zeros(CORRECTED_LAYERS, CHANNELS)
        )
        self.backward_transport_delta = nn.Parameter(
            torch.zeros(CORRECTED_LAYERS, CHANNELS)
        )
        self.backward_mix_delta = nn.Parameter(
            torch.zeros(CORRECTED_LAYERS, CHANNELS, CHANNELS)
        )
        nn.init.normal_(self.backward_source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.backward_gate_weight, mean=0.0, std=2.0e-3)

    def standardized_raw(self, raw_layer: torch.Tensor, layer: int) -> torch.Tensor:
        tail = torch.clamp(
            (raw_layer[..., 1:] - self.centers[layer]) / self.scales[layer],
            -8.0,
            8.0,
        )
        return torch.cat((raw_layer[..., :1], tail), dim=-1)

    def backward_features(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        adjoint: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        rms = self.layer_rms[layer]
        features = torch.cat(
            (
                self.standardized_raw(raw_layer, layer),
                torch.clamp(adjoint / rms, -8.0, 8.0),
                torch.clamp(
                    adjoint
                    / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
                    -8.0,
                    8.0,
                ),
            ),
            dim=-1,
        )
        if features.shape[-1] != BACKWARD_INPUT_FEATURES:
            raise AssertionError(features.shape)
        return features

    def compute_backward(
        self, raw: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        batch = raw.shape[0]
        zero = torch.zeros(
            batch, WIDTH, CHANNELS, dtype=raw.dtype, device=raw.device
        )
        history: list[torch.Tensor] = [zero, zero] + [zero] * CORRECTED_LAYERS
        next_state = zero
        identity = torch.eye(CHANNELS, dtype=raw.dtype, device=raw.device)
        for layer in range(DEPTH - 1, FIRST_CORRECTED_LAYER - 1, -1):
            index = layer - FIRST_CORRECTED_LAYER
            if layer == DEPTH - 1:
                adjoint = zero
            else:
                phi_next = raw[:, layer + 1, :, 5, None]
                adjoint = torch.bmm(
                    weights[:, layer + 1], phi_next * next_state
                )
            hidden = self.backward_trunk(
                self.backward_features(
                    raw[:, layer], sigma[:, layer], adjoint, layer
                )
            )
            source_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.backward_source_weight[index]
            ) + self.backward_source_bias[index]
            source = (
                sigma[:, layer] * self.layer_rms[layer]
            )[..., None] * source_standard
            gate_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.backward_gate_weight[index]
            ) + self.backward_gate_bias[index]
            gain = 1.0 + 0.25 * torch.tanh(
                gate_standard + self.backward_transport_delta[index]
            )
            mixing = identity + 0.12 * torch.tanh(
                self.backward_mix_delta[index]
            )
            next_state = torch.einsum("bwc,cd->bwd", gain * adjoint, mixing) + source
            history[layer] = next_state
        return torch.stack(history, dim=1)

    def forward_features(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        messages: torch.Tensor,
        energies: torch.Tensor,
        backward: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        rms = self.layer_rms[layer]
        rms2 = rms.square()
        features = torch.cat(
            (
                self.standardized_raw(raw_layer, layer),
                torch.clamp(messages / rms, -8.0, 8.0),
                torch.clamp(
                    messages
                    / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
                    -8.0,
                    8.0,
                ),
                torch.clamp(
                    torch.log1p(torch.clamp(energies / rms2, min=0.0)),
                    0.0,
                    8.0,
                ),
                torch.clamp(
                    messages
                    / torch.sqrt(energies + COHERENCE_FLOOR * rms2),
                    -8.0,
                    8.0,
                ),
                torch.clamp(backward / rms, -8.0, 8.0),
                torch.clamp(
                    backward
                    / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
                    -8.0,
                    8.0,
                ),
            ),
            dim=-1,
        )
        if features.shape[-1] != FORWARD_INPUT_FEATURES:
            raise AssertionError(features.shape)
        return features

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
        backward_history = self.compute_backward(raw, sigma, weights)
        batch = raw.shape[0]
        states = torch.zeros(
            batch, WIDTH, CHANNELS, dtype=raw.dtype, device=raw.device
        )
        outputs: list[torch.Tensor] = []
        state_history: list[torch.Tensor] = []
        energy_history: list[torch.Tensor] = []
        identity = torch.eye(CHANNELS, dtype=raw.dtype, device=raw.device)
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                states = torch.zeros_like(states)
                energies = torch.zeros_like(states)
                outputs.append(states[..., 0])
                state_history.append(states)
                energy_history.append(energies)
                continue
            if layer == FIRST_CORRECTED_LAYER:
                messages = torch.zeros_like(states)
                energies = torch.zeros_like(states)
            else:
                prior = states
                if exposure < 1.0:
                    primary = (
                        exposure * states[..., 0]
                        + (1.0 - exposure) * truth[:, layer - 1]
                    )
                    prior = torch.cat((primary[..., None], states[..., 1:]), dim=-1)
                messages = torch.bmm(
                    prior.transpose(1, 2), weights[:, layer]
                ).transpose(1, 2)
                energies = torch.bmm(
                    prior.square().transpose(1, 2), weights[:, layer].square()
                ).transpose(1, 2)
                phi = raw[:, layer, :, 5, None]
                messages = phi * messages
                energies = phi.square() * energies
            hidden = self.forward_trunk(
                self.forward_features(
                    raw[:, layer],
                    sigma[:, layer],
                    messages,
                    energies,
                    backward_history[:, layer],
                    layer,
                )
            )
            source_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.forward_source_weight[layer]
            ) + self.forward_source_bias[layer]
            source = (
                sigma[:, layer] * self.layer_rms[layer]
            )[..., None] * source_standard
            gate_standard = torch.einsum(
                "bwh,hc->bwc", hidden, self.forward_gate_weight[layer]
            ) + self.forward_gate_bias[layer]
            gain = 1.0 + 0.25 * torch.tanh(
                gate_standard + self.forward_transport_delta[layer]
            )
            mixing = identity + 0.12 * torch.tanh(
                self.forward_mix_delta[layer]
            )
            states = torch.einsum("bwc,cd->bwd", gain * messages, mixing) + source
            outputs.append(states[..., 0])
            state_history.append(states)
            energy_history.append(energies)
        prediction = torch.stack(outputs, dim=1)
        if return_states:
            return (
                prediction,
                torch.stack(state_history, dim=1),
                torch.stack(energy_history, dim=1),
                backward_history,
            )
        return prediction


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def cost_record(model: nn.Module) -> dict[str, object]:
    transitions = CORRECTED_LAYERS - 1
    dense_vector_matrix = 2 * WIDTH * WIDTH - WIDTH
    j_mean_plus_energy_transport = transitions * 2 * CHANNELS * dense_vector_matrix
    backward_adjoint_transport = transitions * CHANNELS * dense_vector_matrix
    raw20_materialization_reserve = DEPTH * 16 * WIDTH * WIDTH
    raw20_standardization = 2 * LOCAL_FEATURES * DEPTH * WIDTH
    forward_per_unit = (
        (2 * FORWARD_INPUT_FEATURES - 1) * FORWARD_HIDDEN
        + (2 * FORWARD_HIDDEN - 1) * FORWARD_HIDDEN
        + (2 * FORWARD_HIDDEN - 1) * (2 * CHANNELS)
        + 64 * (2 * FORWARD_HIDDEN + CHANNELS)
        + 128
        + CHANNELS * (2 * CHANNELS - 1)
    )
    backward_per_unit = (
        (2 * BACKWARD_INPUT_FEATURES - 1) * BACKWARD_HIDDEN
        + (2 * BACKWARD_HIDDEN - 1) * BACKWARD_HIDDEN
        + (2 * BACKWARD_HIDDEN - 1) * (2 * CHANNELS)
        + 64 * (2 * BACKWARD_HIDDEN + 2 * CHANNELS)
        + 2_048
        + CHANNELS * (2 * CHANNELS - 1)
        + 128
    )
    forward_cell = CORRECTED_LAYERS * WIDTH * forward_per_unit
    backward_cell = CORRECTED_LAYERS * WIDTH * backward_per_unit
    output_addition = DEPTH * WIDTH
    correction = (
        j_mean_plus_energy_transport
        + backward_adjoint_transport
        + raw20_materialization_reserve
        + raw20_standardization
        + forward_cell
        + backward_cell
        + output_addition
    )
    total = HOST_FLOPS + correction
    if forward_per_unit != 40_916 or backward_per_unit != 23_572:
        raise AssertionError((forward_per_unit, backward_per_unit))
    if backward_adjoint_transport != 108_998_656:
        raise AssertionError(backward_adjoint_transport)
    if correction != CORRECTION_FLOPS or total != TOTAL_FLOPS:
        raise AssertionError((correction, total))
    if correction - J_CORRECTION_FLOPS != ADDED_BACKWARD_ADJOINT_FLOPS:
        raise AssertionError("added correction arithmetic changed")
    native_total = NATIVE_HOST_TAPE_BASELINE_FLOPS + correction
    return {
        "parameter_count": parameter_count(model),
        "measured_old_q24_host_flops": HOST_FLOPS,
        "j_mean_plus_energy_transport_upper_estimate": j_mean_plus_energy_transport,
        "backward_exact_adjoint_transport_upper_estimate": backward_adjoint_transport,
        "raw20_materialization_reserve": raw20_materialization_reserve,
        "raw20_standardization": raw20_standardization,
        "forward_typed_cell_reserve": forward_cell,
        "backward_typed_cell_reserve": backward_cell,
        "output_addition": output_addition,
        "added_over_j_correction": correction - J_CORRECTION_FLOPS,
        "correction_source_structure_upper_estimate": correction,
        "old_q24_host_plus_correction_upper_estimate": total,
        "native_q24_host_tape_baseline_flops": NATIVE_HOST_TAPE_BASELINE_FLOPS,
        "native_q24_host_tape_plus_correction_upper_estimate": native_total,
        "headroom_below_exact_2^41_native_total": EXACT_SCORING_BUDGET - native_total,
        "phase2_budget": PHASE2_BUDGET,
        "within_phase2_budget_by_upper_estimate": total <= PHASE2_BUDGET,
        "authority": "frozen conservative source-structure estimate; literal meter only after accuracy and native-transfer gates pass",
    }


def fit_fold(
    fold: int,
    model: BackwardAdjointStateResidual,
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
    layer_weights = torch.linspace(0.5, 1.5, CORRECTED_LAYERS)
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
            prediction, forward_states, _, backward_states = model(
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
            aux = torch.cat(
                (
                    forward_states[:, FIRST_CORRECTED_LAYER:, :, 1:],
                    backward_states[:, FIRST_CORRECTED_LAYER:],
                ),
                dim=-1,
            )
            aux_normalized = aux / model.layer_rms[
                FIRST_CORRECTED_LAYER:
            ][None, :, None, None]
            aux_penalty = AUX_STATE_PENALTY * aux_normalized.square().mean()
            identity_penalty = IDENTITY_PENALTY * (
                model.forward_transport_delta.square().mean()
                + model.forward_mix_delta.square().mean()
                + model.backward_transport_delta.square().mean()
                + model.backward_mix_delta.square().mean()
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
            "own_primary_forward_state_exposure": exposure,
            "backward_state_exposure": 1.0,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "forward_transport_delta_max_abs": float(
                model.forward_transport_delta.detach().abs().max()
            ),
            "backward_transport_delta_max_abs": float(
                model.backward_transport_delta.detach().abs().max()
            ),
            "max_rss_bytes": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, max_rss


def score(
    model: BackwardAdjointStateResidual,
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
            if not isinstance(prediction, torch.Tensor):
                raise AssertionError(type(prediction))
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
    release = training_release_or_raise()
    result_path, model_path = fold_paths(fold)
    if result_path.exists() or model_path.exists():
        if not result_path.is_file() or not model_path.is_file():
            raise FileExistsError("partial fold artifact collision")
        existing = read_json(result_path)
        if existing.get("program_sha256") != seal["program_sha256"]:
            raise AssertionError("completed fold belongs to another program")
        return existing
    dataset = PublicResidualDataset(role="observer_fit")
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    valid = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    torch.manual_seed(SEED + fold)
    model = BackwardAdjointStateResidual(centers, scales, layer_rms)
    if parameter_count(model) != 39_944:
        raise AssertionError(f"unexpected parameter count {parameter_count(model)}")
    cost = cost_record(model)
    started = time.perf_counter()
    trace, max_rss = fit_fold(fold, model, dataset, train, error)
    autonomous = score(model, dataset, valid, error)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-backward-adjoint-state-fold-v1",
        "status": "complete",
        "node": "P2R0F2M",
        "fold": fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "architecture": {
            "forward_channels": CHANNELS,
            "backward_channels": CHANNELS,
            "forward_input_features": FORWARD_INPUT_FEATURES,
            "backward_input_features": BACKWARD_INPUT_FEATURES,
            "parameters": parameter_count(model),
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
            "forward_transport": "Phi*(s@W)",
            "backward_transport": "W@(Phi*b), exact transpose of signed forward transport",
            "deployment_truth_inputs": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "own_primary_forward_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "backward_state_exposure": 1.0,
            "trace": trace,
            "observed_max_rss_bytes": max_rss,
        },
        "host": {"final_mse": float(host_layer[-1]), "layer_mse": host_layer.tolist()},
        "autonomous": autonomous,
        "cost": cost,
        "capacity_only_firewall": True,
        "training_release_sha256": sha256(TRAINING_RELEASE),
        "native_transfer_terminal_result_sha256": release[
            "native_transfer_terminal_result_sha256"
        ],
        "observer_selection_targets_opened": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "schema": "public-full688-backward-adjoint-state-fold-model-v1",
        "node": "P2R0F2M",
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
    training_release_or_raise()
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
        "schema": "public-full688-backward-adjoint-state-cv-v1",
        "status": "complete",
        "node": "P2R0F2M",
        "observer_fit_networks": total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "own_primary_forward_state_exposure_schedule": list(EXPOSURE_SCHEDULE),
            "backward_state_exposure": 1.0,
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
            "stretch_mean_below_1e_8": autonomous_final < STRETCH_MSE,
        },
        "cost": records[0]["cost"],
        "capacity_only_pending_native_q24_deployment_verification": True,
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
    dataset = PublicResidualDataset(role="observer_fit")
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    train = np.flatnonzero(folds != 0)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    torch.manual_seed(SEED)
    model = BackwardAdjointStateResidual(centers, scales, layer_rms)
    row = dataset.load(0)
    raw = torch.from_numpy(np.asarray(row.raw_locals)[None].copy())
    sigma = torch.from_numpy(np.asarray(row.sigma)[None].copy())
    weights = torch.from_numpy(
        np.asarray(np.load(row.weights_path, mmap_mode="r", allow_pickle=False))[None].copy()
    )
    truth = torch.from_numpy(np.asarray(row.target)[None].copy())
    with torch.no_grad():
        prediction, forward_states, energies, backward_states = model(
            raw,
            sigma,
            weights,
            truth=truth,
            exposure=0.0,
            return_states=True,
        )
        _, _, _, backward_control = model(
            raw,
            sigma,
            weights,
            truth=torch.zeros_like(truth),
            exposure=0.0,
            return_states=True,
        )
    expected_state = (1, DEPTH, WIDTH, CHANNELS)
    if tuple(prediction.shape) != (1, DEPTH, WIDTH):
        raise AssertionError("smoke output shape changed")
    if tuple(forward_states.shape) != expected_state:
        raise AssertionError("smoke forward-state shape changed")
    if tuple(energies.shape) != expected_state:
        raise AssertionError("smoke energy shape changed")
    if tuple(backward_states.shape) != expected_state:
        raise AssertionError("smoke backward-state shape changed")
    tensors = (prediction, forward_states, energies, backward_states)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("smoke output is nonfinite")
    if not bool((energies >= 0.0).all()):
        raise FloatingPointError("forward contribution energy became negative")
    if not torch.equal(backward_states, backward_control):
        raise AssertionError("backward state consumed teacher truth")
    if not bool((backward_states[:, :FIRST_CORRECTED_LAYER] == 0.0).all()):
        raise AssertionError("backward state changed protected early layers")
    if model.forward_trunk[0].in_features != FORWARD_INPUT_FEATURES:
        raise AssertionError("forward input width changed")
    if model.backward_trunk[0].in_features != BACKWARD_INPUT_FEATURES:
        raise AssertionError("backward input width changed")
    if parameter_count(model) != 39_944:
        raise AssertionError("parameter count changed")
    payload = {
        "node": "P2R0F2M",
        "launch_status": "held_after_smoke_pending_native_transfer_terminal_and_root_release",
        "readiness": readiness,
        "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        "architecture": {
            "forward_channels": CHANNELS,
            "backward_channels": CHANNELS,
            "forward_input_features": FORWARD_INPUT_FEATURES,
            "backward_input_features": BACKWARD_INPUT_FEATURES,
            "parameters": parameter_count(model),
            "output_shape": list(prediction.shape),
            "forward_state_shape": list(forward_states.shape),
            "backward_state_shape": list(backward_states.shape),
            "finite": True,
            "energy_nonnegative": True,
            "backward_truth_independent_byte_equal": True,
        },
        "cost": cost_record(model),
        "training_release_present": TRAINING_RELEASE.is_file(),
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
