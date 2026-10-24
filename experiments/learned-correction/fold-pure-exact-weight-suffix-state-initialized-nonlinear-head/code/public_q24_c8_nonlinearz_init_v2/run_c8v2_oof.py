#!/usr/bin/env python3
"""Fold-pure exact-initialized C8 suffix-state OOF discriminator.

Each fold first deterministically reconstructs the frozen nonlinear-Z model
trained without that fold and verifies its held predictions against the
persisted OOF authority.  C8 then starts at exactly that scalar function:
the inherited 42-input path and scalar head are copied, the eight message
columns and seven extra channels are zero, and the epoch-zero audit bypasses
the zero adapter.  Only after parity is recorded are the extra heads seeded
and trained in two fixed stages.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
V1_PROGRAM = P2 / "public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py"
V1_TERMINAL = P2 / "public_q24_c8_autonomous_correction_state_v1/PROCEDURAL_TERMINAL_RECEIPT.json"
BASE_PROGRAM = P2 / "public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py"
BASE_RESULT = P2 / "public_q24_nonlinear_transported_raw20_head_v1/RESULT.json"
BASE_MODEL = P2 / "public_q24_nonlinear_transported_raw20_head_v1/NONLINEAR_Z_HEAD.pt"
NATIVE_RESULT = P2 / "public_q24_native_transport_energy_final_v1/FINAL_FIT_RESULT.json"
NATIVE_MODEL = P2 / "public_q24_native_transport_energy_final_v1/FINAL_MODEL.pt"
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
D3_GATE = P2 / "public_q24_d3_alpha075_full688_qhead_v1/FULL_GATE_RESULT.json"
D4_GATE = P2 / "public_q24_d4_alpha075_full688_qhead_v1/FULL_GATE_RESULT.json"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_STATE = HERE / "RUN_STATE.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "C8V2_AUTONOMOUS_STATE.pt"

V1_PROGRAM_SHA256 = "82df4b70b2d9262edea50cb4e659e6538e24fd28a93b6ec0224975f6bed98519"
V1_TERMINAL_SHA256 = "0d633e07cd6f6c2008b73f6bfeeeeab314b03480ec1982c60ceb091971726391"
BASE_PROGRAM_SHA256 = "032d994d45d0255d46ec09b5dca4fdff9f9baa3273972082e75014e0501dee96"
BASE_RESULT_SHA256 = "c1ef8105ac29b6f4ff75dc7aa5974b1d8f93668437ed402d32ebbad393f2dd35"
BASE_MODEL_SHA256 = "1b542406201a3f8757c8423d1f9b2c540784f7e6c98da687e8fca4e735388676"
NATIVE_RESULT_SHA256 = "e8261895c5d8f5f1feb1e15f257612d1d1b36a90e8bde1ab05e2844100a8ba08"
NATIVE_MODEL_SHA256 = "21ad1dede3373c90d2694158b6599c4c172d7181c72634a6ec8e362449b7d1f8"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
Q_SOURCE_SHA256 = "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3"
D3_GATE_SHA256 = "3e84fae43460a409a4212f90b912630c98589d1c2538ce0553a34a636b239eb4"
D4_GATE_SHA256 = "742bef9476c84c5c6afb2d6e3dced0e3d94e2b4a1ea3aae6c5a2c71d75f5eca2"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ALL_RANKS = np.arange(688, dtype=np.int64)
SUFFIX = tuple(range(10, 16))
FOLDS = 4
CHANNELS = 8
EXTRAS = 7
BASE_INPUTS = 42
MESSAGE_INPUTS = 8
HIDDEN = 64
PARAMETERS = 10_544
LOCAL_ANCHOR_WEIGHT = 0.05
PHASE1_EPOCHS = 4
PHASE1_LR = 3.0e-4
PHASE1_MIN_LR = 1.0e-4
PHASE2_EPOCHS = 8
PHASE2_LR = 2.0e-4
PHASE2_MIN_LR = 5.0e-5
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
MODEL_SEED = 0x43385A12
EXTRA_SEED = 0xC8E7A513
EXTRA_INIT_STD = 2.0**-16
MIN_RELATIVE_GAIN = 0.0025
MIN_IMPROVING_FOLDS = 3
MIN_PAIRED_Z = 2.0
MIN_NETWORK_WIN_FRACTION = 0.55
MIN_FOLD_WIN_FRACTION = 0.50
MEASURED_NATIVE_AUTHORITY_MSE = 2.9736792457285063e-8
LEGACY_SCIENTIFIC_ADMISSION_MSE = 2.9672528546945005e-8
D4_BOARD_RAW_MSE = 2.9409071835573285e-8
D4_BOARD_ADJUSTED_SCORE = 1.783621478716186e-8
D3_BOARD_MULTIPLIER_328620 = 0.6377160962460696
C8_TOTAL_INCREMENTAL_VS_D3_UPPER = 758_625_259
C8_D3_STRUCTURAL_MULTIPLIER = (
    D3_BOARD_MULTIPLIER_328620
    + C8_TOTAL_INCREMENTAL_VS_D3_UPPER / 2_199_023_255_552
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    specification = importlib.util.spec_from_file_location(tag, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[tag] = module
    specification.loader.exec_module(module)
    return module


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_static_bindings() -> dict[str, object]:
    observed = {
        "program_sha256": sha256(Path(__file__)),
        "v1_program_sha256": sha256(V1_PROGRAM),
        "v1_terminal_sha256": sha256(V1_TERMINAL),
        "nonlinear_z_program_sha256": sha256(BASE_PROGRAM),
        "nonlinear_z_result_sha256": sha256(BASE_RESULT),
        "nonlinear_z_model_sha256": sha256(BASE_MODEL),
        "native_result_sha256": sha256(NATIVE_RESULT),
        "native_model_sha256": sha256(NATIVE_MODEL),
        "j_source_sha256": sha256(J_SOURCE),
        "q_source_sha256": sha256(Q_SOURCE),
        "d3_gate_sha256": sha256(D3_GATE),
        "d4_gate_sha256": sha256(D4_GATE),
    }
    literal = {
        "v1_program_sha256": V1_PROGRAM_SHA256,
        "v1_terminal_sha256": V1_TERMINAL_SHA256,
        "nonlinear_z_program_sha256": BASE_PROGRAM_SHA256,
        "nonlinear_z_result_sha256": BASE_RESULT_SHA256,
        "nonlinear_z_model_sha256": BASE_MODEL_SHA256,
        "native_result_sha256": NATIVE_RESULT_SHA256,
        "native_model_sha256": NATIVE_MODEL_SHA256,
        "j_source_sha256": J_SOURCE_SHA256,
        "q_source_sha256": Q_SOURCE_SHA256,
        "d3_gate_sha256": D3_GATE_SHA256,
        "d4_gate_sha256": D4_GATE_SHA256,
    }
    for key, wanted in literal.items():
        if observed[key] != wanted:
            raise AssertionError(f"literal authority changed: {key}")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_ANY_C8V2_TARGET_OPEN":
        raise AssertionError("c8v2 preregistration is not frozen")
    for key, value in observed.items():
        if prereg.get("bindings", {}).get(key) != value:
            raise AssertionError(f"preregistered binding changed: {key}")
    return prereg


class ExactInitializedC8(nn.Module):
    def __init__(self, nonlinear_model):
        super().__init__()
        self.register_buffer("centers", nonlinear_model.centers.detach().clone())
        self.register_buffer("scales", nonlinear_model.scales.detach().clone())
        self.register_buffer("layer_rms", nonlinear_model.layer_rms.detach().clone())
        self.register_buffer("z_center", nonlinear_model.z_center.detach().clone())
        self.register_buffer("z_scale", nonlinear_model.z_scale.detach().clone())
        self.base_linear = nn.Linear(BASE_INPUTS, HIDDEN)
        self.message_linear = nn.Linear(MESSAGE_INPUTS, HIDDEN, bias=False)
        self.trunk2 = nn.Linear(HIDDEN, HIDDEN)
        self.scalar_weight = nn.Parameter(torch.empty(len(SUFFIX), HIDDEN))
        self.scalar_bias = nn.Parameter(torch.empty(len(SUFFIX)))
        self.extra_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN, EXTRAS))
        self.extra_bias = nn.Parameter(torch.zeros(len(SUFFIX), EXTRAS))
        with torch.no_grad():
            self.base_linear.weight.copy_(nonlinear_model.trunk[0].weight)
            self.base_linear.bias.copy_(nonlinear_model.trunk[0].bias)
            self.message_linear.weight.zero_()
            self.trunk2.weight.copy_(nonlinear_model.trunk[2].weight)
            self.trunk2.bias.copy_(nonlinear_model.trunk[2].bias)
            self.scalar_weight.copy_(nonlinear_model.head_weight)
            self.scalar_bias.copy_(nonlinear_model.head_bias)
        if sum(value.numel() for value in self.parameters()) != PARAMETERS:
            raise AssertionError("c8v2 parameter count changed")

    def base_features(self, raw, sigma, base_state, base_q, z_state, layer: int):
        slot = SUFFIX.index(layer)
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        z_standard = torch.clamp(
            (z_state - self.z_center[slot]) / self.z_scale[slot], -8.0, 8.0
        )
        return torch.cat(
            (
                raw[..., :1],
                tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(
                    base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0
                )[..., None],
                z_standard,
            ),
            dim=-1,
        )

    def source(
        self, raw, sigma, base_state, base_q, z_state, message, layer: int,
        *, adapter_enabled: bool,
    ):
        features = self.base_features(raw, sigma, base_state, base_q, z_state, layer)
        preactivation = self.base_linear(features)
        if adapter_enabled:
            rms = self.layer_rms[layer]
            preactivation = preactivation + self.message_linear(
                torch.clamp(message / rms, -8.0, 8.0)
            )
        hidden = torch.nn.functional.silu(preactivation)
        hidden = torch.nn.functional.silu(self.trunk2(hidden))
        slot = SUFFIX.index(layer)
        scalar_standard = torch.einsum(
            "...h,h->...", hidden, self.scalar_weight[slot]
        ) + self.scalar_bias[slot]
        extra_standard = hidden @ self.extra_weight[slot] + self.extra_bias[slot]
        physical = sigma * self.layer_rms[layer]
        return physical * scalar_standard, physical[..., None] * extra_standard

    def seed_extras(self, stream_id: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((EXTRA_SEED ^ (int(stream_id) * 0x9E3779B1)) & 0xFFFF_FFFF)
        with torch.no_grad():
            self.extra_weight.normal_(0.0, EXTRA_INIT_STD, generator=generator)
            self.extra_bias.zero_()

    def phase_parameters(self, joint: bool):
        for parameter in self.parameters():
            parameter.requires_grad_(joint)
        if not joint:
            self.message_linear.weight.requires_grad_(True)
            self.extra_weight.requires_grad_(True)
            self.extra_bias.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def load_final_nonlinear(base, alpha_model):
    artifact = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "q24-nonlinear-transported-raw20-head-checkpoint-v1":
        raise AssertionError("final nonlinear-Z schema changed")
    model = base.NonlinearZHead(
        alpha_model,
        artifact["state_dict"]["z_center"].cpu().numpy(),
        artifact["state_dict"]["z_scale"].cpu().numpy(),
    )
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model


def static_audit() -> dict[str, object]:
    verify_static_bindings()
    base = load_module("c8v2_static_base", BASE_PROGRAM)
    base_runner = load_module("c8v2_static_full688", base.BASE_PROGRAM)
    j = load_module("c8v2_static_j", J_SOURCE)
    alpha = base_runner.load_alpha_j(j)
    nonlinear = load_final_nonlinear(base, alpha)
    candidate = ExactInitializedC8(nonlinear)
    generator = torch.Generator().manual_seed(0xC8A017)
    raw = torch.randn((64, 20), generator=generator)
    raw[:, 5] = torch.sigmoid(raw[:, 5])
    sigma = torch.rand(64, generator=generator) + 0.25
    state = torch.randn(64, generator=generator) * 1e-4
    local_q = torch.randn(64, generator=generator) * 1e-4
    zstate = torch.randn((64, 20), generator=generator)
    zero_message = torch.zeros((64, CHANNELS))
    with torch.no_grad():
        inherited = nonlinear.source(raw, sigma, state, local_q, zstate, 10)
        scalar, extras = candidate.source(
            raw, sigma, state, local_q, zstate, zero_message, 10,
            adapter_enabled=False,
        )
    checks = {
        "parameters_exactly_10544": sum(v.numel() for v in candidate.parameters()) == PARAMETERS,
        "synthetic_scalar_torch_equal": bool(torch.equal(inherited, scalar)),
        "synthetic_extras_exact_zero": bool(torch.equal(extras, torch.zeros_like(extras))),
        "message_adapter_exact_zero": bool(torch.count_nonzero(candidate.message_linear.weight) == 0),
        "extra_heads_exact_zero": bool(
            torch.count_nonzero(candidate.extra_weight) == 0
            and torch.count_nonzero(candidate.extra_bias) == 0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "status": "STATIC_GO",
        "program_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "checks": checks,
        "targets_opened": [],
        "training_started": False,
    }


def nonlinear_oof_authority() -> tuple[dict[str, object], np.ndarray]:
    result = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PROMOTED_FIXED_FULL688_MODEL"
        or result.get("bindings", {}).get("promoted_model_sha256") != BASE_MODEL_SHA256
    ):
        raise AssertionError("nonlinear-Z result authority changed")
    values = np.empty(688, dtype=np.float64)
    rows = result.get("crossfit", {}).get("per_network", [])
    if len(rows) != 688:
        raise AssertionError("nonlinear-Z OOF coverage changed")
    for row in rows:
        values[int(row["role_rank"])] = float(row["candidate_final_mse"])
    return result, values


def fold_base_reconstruction(base, q, alpha, prepared, dataset, train, held, fold, authority):
    model, q_rms, trace = base.train_fixed(q, alpha, prepared, train, fold)
    scored = base.score_fold(model, prepared, dataset, held)
    rows = scored["per_network"]
    observed = np.asarray([float(row["candidate_final_mse"]) for row in rows])
    expected = authority[held]
    maximum = float(np.max(np.abs(observed - expected)))
    if maximum > 5e-20:
        raise AssertionError(("fold nonlinear reconstruction mismatch", fold, maximum))
    return model, q_rms, trace, maximum


def all_row_source_parity(candidate, nonlinear, prepared) -> dict[str, object]:
    maximum = 0.0
    exact = True
    extras_zero = True
    zero_message = torch.zeros((1024, CHANNELS), dtype=torch.float32)
    nonlinear.eval()
    candidate.eval()
    with torch.no_grad():
        for rank in range(688):
            for slot, layer in enumerate(SUFFIX):
                raw = torch.from_numpy(prepared.raw_suffix[rank, slot])
                sigma = torch.from_numpy(prepared.sigma_suffix[rank, slot])
                state = torch.from_numpy(prepared.base_state[rank, slot])
                local_q = torch.from_numpy(prepared.base_q[rank, slot])
                zstate = torch.from_numpy(prepared.z_suffix[rank, slot])
                inherited = nonlinear.source(raw, sigma, state, local_q, zstate, layer)
                scalar, extras = candidate.source(
                    raw, sigma, state, local_q, zstate, zero_message, layer,
                    adapter_enabled=False,
                )
                exact = exact and bool(torch.equal(inherited, scalar))
                extras_zero = extras_zero and bool(torch.equal(extras, torch.zeros_like(extras)))
                maximum = max(maximum, float(torch.max(torch.abs(inherited - scalar))))
    if not exact or not extras_zero or maximum != 0.0:
        raise AssertionError(("all-row epoch0 source parity failed", exact, extras_zero, maximum))
    return {
        "rows": 688,
        "layers_per_row": 6,
        "scalar_torch_equal_everywhere": exact,
        "extra_channels_exact_zero_everywhere": extras_zero,
        "maximum_absolute_source_difference": maximum,
    }


def epoch0_numpy_score(candidate, prepared, dataset, truth_suffix, held):
    rows = []
    candidate.eval()
    zero_message = torch.zeros((1024, CHANNELS), dtype=torch.float32)
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(1024, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                if slot == 0:
                    message = np.zeros(1024, dtype=np.float64)
                else:
                    message = (
                        prepared.raw_suffix[rank, slot, :, 5].astype(np.float64)
                        * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                    )
                scalar, extras = candidate.source(
                    torch.from_numpy(prepared.raw_suffix[rank, slot]),
                    torch.from_numpy(prepared.sigma_suffix[rank, slot]),
                    torch.from_numpy(prepared.base_state[rank, slot]),
                    torch.from_numpy(prepared.base_q[rank, slot]),
                    torch.from_numpy(prepared.z_suffix[rank, slot]),
                    zero_message,
                    layer,
                    adapter_enabled=False,
                )
                if torch.count_nonzero(extras) != 0:
                    raise AssertionError("epoch0 extras became nonzero")
                state = (message + scalar.cpu().numpy().astype(np.float64)).astype(np.float32)
            residual = truth_suffix[rank, -1].astype(np.float64) - state.astype(np.float64)
            rows.append(float(np.mean(residual * residual)))
    return np.asarray(rows, dtype=np.float64)


def rollout_torch(candidate, prepared, dataset, rank: int):
    weights_np = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
    scalar_state = torch.zeros(1024, dtype=torch.float32)
    extra_state = torch.zeros((1024, EXTRAS), dtype=torch.float32)
    scalar_states = []
    for slot, layer in enumerate(SUFFIX):
        raw = torch.from_numpy(prepared.raw_suffix[rank, slot])
        sigma = torch.from_numpy(prepared.sigma_suffix[rank, slot])
        if slot == 0:
            scalar_message = torch.zeros_like(scalar_state)
            extra_message = torch.zeros_like(extra_state)
        else:
            weights = torch.from_numpy(np.asarray(weights_np[layer], dtype=np.float32)).double()
            phi = raw[:, 5].double()
            scalar_message = (phi * (scalar_state.double() @ weights)).float()
            extra_message = (phi[:, None] * (weights.T @ extra_state.double())).float()
        message = torch.cat((scalar_message[:, None], extra_message), dim=1)
        scalar_source, extra_source = candidate.source(
            raw,
            sigma,
            torch.from_numpy(prepared.base_state[rank, slot]),
            torch.from_numpy(prepared.base_q[rank, slot]),
            torch.from_numpy(prepared.z_suffix[rank, slot]),
            message,
            layer,
            adapter_enabled=True,
        )
        scalar_state = scalar_message + scalar_source
        extra_state = extra_message + extra_source
        scalar_states.append(scalar_state)
    return scalar_states


def training_rms(truth_suffix: np.ndarray, train: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.mean(truth_suffix[train].astype(np.float64) ** 2, axis=(0, 2)))
    return np.maximum(values, 2.0**-20).astype(np.float32)


def train_phase(
    candidate, prepared, dataset, truth_suffix, train, rms, stream_id,
    *, phase: str, epochs: int, learning_rate: float, minimum_lr: float, joint: bool,
):
    parameters = candidate.phase_parameters(joint)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train), eta_min=minimum_lr
    )
    rng = np.random.default_rng(
        (MODEL_SEED ^ (int(stream_id) * 0x9E3779B1) ^ (0x51515151 if joint else 0xA0A0A0A0))
        & 0xFFFF_FFFF
    )
    trace = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        objective = 0.0
        endpoint_sum = 0.0
        candidate.train()
        for rank_value in rng.permutation(train):
            rank = int(rank_value)
            states = rollout_torch(candidate, prepared, dataset, rank)
            truth = torch.from_numpy(truth_suffix[rank])
            scale = torch.from_numpy(rms)[:, None]
            endpoint = torch.mean(((states[-1] - truth[-1]) / scale[-1]).square())
            local = torch.stack([
                torch.mean(((states[slot] - truth[slot]) / scale[slot]).square())
                for slot in range(len(SUFFIX))
            ]).mean()
            loss = endpoint + LOCAL_ANCHOR_WEIGHT * local
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError((stream_id, phase, epoch, rank))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective += float(loss.detach())
            endpoint_sum += float(endpoint.detach())
        row = {
            "stream_id": int(stream_id),
            "phase": phase,
            "epoch": epoch,
            "objective": objective / len(train),
            "endpoint_objective": endpoint_sum / len(train),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    candidate.eval()
    return trace


def adapt(candidate, prepared, dataset, truth_suffix, train, stream_id):
    rms = training_rms(truth_suffix, train)
    candidate.seed_extras(stream_id)
    phase1 = train_phase(
        candidate, prepared, dataset, truth_suffix, train, rms, stream_id,
        phase="extra_heads_and_message_adapter",
        epochs=PHASE1_EPOCHS, learning_rate=PHASE1_LR,
        minimum_lr=PHASE1_MIN_LR, joint=False,
    )
    phase2 = train_phase(
        candidate, prepared, dataset, truth_suffix, train, rms, stream_id,
        phase="joint_lower_lr",
        epochs=PHASE2_EPOCHS, learning_rate=PHASE2_LR,
        minimum_lr=PHASE2_MIN_LR, joint=True,
    )
    return rms, phase1 + phase2


def score_candidate(candidate, prepared, dataset, truth_suffix, held):
    rows = []
    candidate.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            final = rollout_torch(candidate, prepared, dataset, rank)[-1].cpu().numpy().astype(np.float64)
            residual = truth_suffix[rank, -1].astype(np.float64) - final
            rows.append({
                "role_rank": rank,
                "mlp_id": int(dataset.mlp_ids[rank]),
                "candidate_final_mse": float(np.mean(residual * residual)),
            })
    return rows


def main_run() -> None:
    if RUN_STATE.exists() or RESULT.exists() or MODEL.exists():
        raise FileExistsError("c8v2 one-shot has already started")
    verify_static_bindings()
    nonlinear_result, nonlinear_oof = nonlinear_oof_authority()
    v1 = load_module("c8v2_v1_helpers", V1_PROGRAM)
    base = load_module("c8v2_nonlinear", BASE_PROGRAM)
    base_runner = load_module("c8v2_full688", base.BASE_PROGRAM)
    q = load_module("c8v2_q_sampling", Q_SOURCE)
    j = load_module("c8v2_exact_j", J_SOURCE)
    records, native_storage = base_runner.verify_native_rows()
    folds = base_runner.balanced_folds(records)
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("fold authority changed")
    write_json_once(RUN_STATE, {
        "schema": "q24-c8v2-run-state-v1", "status": "RUNNING_ONE_SHOT",
        "pid": os.getpid(), "program_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION), "target_columns_opened": [],
    })

    targets = base_runner.load_targets(records)
    hosts = np.ascontiguousarray(np.stack([
        np.load(base_runner.row_dir(rank) / "host.npy", allow_pickle=False)
        for rank in range(688)
    ]), dtype=np.float32)
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = base_runner.All688Dataset(records)
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    alpha = base_runner.load_alpha_j(j)
    prepared = base.prepare_all_with_z(base_runner, j, alpha, dataset, error)
    truth_suffix = np.ascontiguousarray(error[:, SUFFIX] - prepared.base_state, dtype=np.float32)
    native_values = v1.recompute_native_bank(j, v1.load_native_model(j), dataset, error)
    del targets, hosts, error
    gc.collect()

    per_network: list[dict[str, object] | None] = [None] * 688
    fold_results = []
    base_traces = []
    adaptation_traces = []
    parity_receipts = []
    rms_records = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train = ALL_RANKS[folds != fold]
        held = ALL_RANKS[folds == fold]
        nonlinear, q_rms, base_trace, reconstruction_max = fold_base_reconstruction(
            base, q, alpha, prepared, dataset, train, held, fold, nonlinear_oof
        )
        candidate = ExactInitializedC8(nonlinear)
        source_parity = all_row_source_parity(candidate, nonlinear, prepared)
        epoch0 = epoch0_numpy_score(candidate, prepared, dataset, truth_suffix, held)
        epoch0_max = float(np.max(np.abs(epoch0 - nonlinear_oof[held])))
        if epoch0_max > 5e-20:
            raise AssertionError(("epoch0 held parity failed", fold, epoch0_max))
        parity = {
            "fold": fold,
            "fold_base_reconstruction_maximum_mse_difference": reconstruction_max,
            "epoch0_candidate_maximum_mse_difference": epoch0_max,
            "all_row_source_parity": source_parity,
            "extra_seed_applied_only_after_parity": True,
        }
        parity_receipts.append(parity)
        print(json.dumps({"stage": "epoch0_parity", **parity}, sort_keys=True), flush=True)
        rms, adaptation_trace = adapt(
            candidate, prepared, dataset, truth_suffix, train, fold
        )
        rows = score_candidate(candidate, prepared, dataset, truth_suffix, held)
        values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
        for row in rows:
            per_network[int(row["role_rank"])] = row
        summary = {
            "fold": fold,
            "training_rows": len(train),
            "held_rows": held.tolist(),
            "candidate_final_mse": float(np.mean(values)),
            "nonlinear_z_oof_final_mse": float(np.mean(nonlinear_oof[held])),
            "native_bank_same688_final_mse": float(np.mean(native_values[held])),
            "win_fraction_vs_nonlinear_z": float(np.mean(values < nonlinear_oof[held])),
            "win_fraction_vs_native_bank": float(np.mean(values < native_values[held])),
        }
        fold_results.append(summary)
        base_traces.extend({"fold": fold, **row} for row in base_trace)
        adaptation_traces.extend({"fold": fold, **row} for row in adaptation_trace)
        rms_records.append(rms.tolist())
        print(json.dumps({"stage": "fold_score", **summary}, sort_keys=True), flush=True)
        del nonlinear, candidate
        gc.collect()

    if any(row is None for row in per_network):
        raise AssertionError("c8v2 OOF coverage incomplete")
    rows = [row for row in per_network if row is not None]
    candidate_values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
    gain = nonlinear_oof - candidate_values
    candidate_mse = float(np.mean(candidate_values))
    nonlinear_mse = float(np.mean(nonlinear_oof))
    native_mse = float(np.mean(native_values))
    relative_gain = (nonlinear_mse - candidate_mse) / nonlinear_mse
    gain_se = float(np.std(gain, ddof=1) / math.sqrt(688))
    paired_z = float(np.mean(gain) / gain_se) if gain_se > 0 else float("inf")
    fold_candidate = [float(np.mean(candidate_values[folds == fold])) for fold in range(FOLDS)]
    fold_nonlinear = [float(np.mean(nonlinear_oof[folds == fold])) for fold in range(FOLDS)]
    fold_wins = [float(np.mean(gain[folds == fold] > 0)) for fold in range(FOLDS)]
    improving_folds = sum(a < b for a, b in zip(fold_candidate, fold_nonlinear, strict=True))
    protected = np.asarray(
        nonlinear_result["crossfit"]["legacy_protected40_oof"]["role_ranks"], dtype=np.int64
    )
    projected_adjusted = candidate_mse * C8_D3_STRUCTURAL_MULTIPLIER
    scientific_gates = {
        "finite_688_oof": bool(np.isfinite(candidate_values).all()),
        "relative_gain_vs_nonlinear_z_at_least_0_25pct": bool(relative_gain >= MIN_RELATIVE_GAIN),
        "at_least_three_folds_improve": bool(improving_folds >= MIN_IMPROVING_FOLDS),
        "paired_network_z_at_least_2": bool(paired_z >= MIN_PAIRED_Z),
        "network_wins_vs_nonlinear_z_at_least_55pct": bool(np.mean(gain > 0) >= MIN_NETWORK_WIN_FRACTION),
        "every_fold_wins_at_least_50pct": bool(min(fold_wins) >= MIN_FOLD_WIN_FRACTION),
        "protected40_does_not_worsen": bool(np.mean(candidate_values[protected]) <= np.mean(nonlinear_oof[protected])),
        "all_fold_reconstructions_and_epoch0_parity_pass": bool(len(parity_receipts) == 4),
    }
    scientific_pass = all(scientific_gates.values())
    absolute_gates = {
        "candidate_below_recomputed_native_bank": bool(candidate_mse < native_mse),
        "native_recompute_matches_authority": bool(np.isclose(native_mse, MEASURED_NATIVE_AUTHORITY_MSE, rtol=0.0, atol=2e-15)),
        "candidate_below_legacy_scientific_admission": bool(candidate_mse < LEGACY_SCIENTIFIC_ADMISSION_MSE),
        "network_wins_vs_native_at_least_55pct": bool(np.mean(candidate_values < native_values) >= MIN_NETWORK_WIN_FRACTION),
        "structural_adjusted_beats_d4_328622": bool(projected_adjusted < D4_BOARD_ADJUSTED_SCORE),
    }
    submission_worthy = scientific_pass and all(absolute_gates.values())

    model_hash = None
    final_trace = []
    final_parity = None
    if scientific_pass:
        nonlinear = load_final_nonlinear(base, alpha)
        final_candidate = ExactInitializedC8(nonlinear)
        final_parity = all_row_source_parity(final_candidate, nonlinear, prepared)
        final_rms, final_trace = adapt(
            final_candidate, prepared, dataset, truth_suffix, ALL_RANKS, FOLDS
        )
        artifact = {
            "schema": "q24-c8v2-exact-nonlinearz-initialized-state-v1",
            "state_dict": final_candidate.state_dict(),
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "channels": CHANNELS,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "nonlinear_z_result_sha256": BASE_RESULT_SHA256,
            "training_rms": final_rms,
        }
        temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary)
        temporary.replace(MODEL)
        model_hash = sha256(MODEL)

    result = {
        "schema": "q24-c8v2-exact-nonlinearz-initialized-oof-v1",
        "status": (
            "PROMOTED_SUBMISSION_WORTHY_PENDING_LITERAL_METER" if submission_worthy
            else "PROMOTED_SCIENTIFIC_ONLY" if scientific_pass
            else "CLOSED_NO_BROAD_OOF_GAIN"
        ),
        "fixed_rule": {
            "architecture": "inherited 42->64->64 scalar path plus 8->64 message adapter and seven extra source channels",
            "parameters": PARAMETERS,
            "fold_pure_base_reconstruction": True,
            "epoch0_scalar_source_torch_equal_all688_each_fold": True,
            "phase1": "4 epochs adapters/extras only, 3e-4 to1e-4",
            "phase2": "8 epochs joint, 2e-4 to5e-5",
            "extra_seed_std": EXTRA_INIT_STD,
            "loss": f"endpoint + {LOCAL_ANCHOR_WEIGHT} local channel0 anchor",
        },
        "crossfit": {
            "candidate_oof_final_mse": candidate_mse,
            "nonlinear_z_oof_final_mse": nonlinear_mse,
            "recomputed_native_bank_same688_final_mse": native_mse,
            "relative_gain_vs_nonlinear_z": relative_gain,
            "paired_network_z": paired_z,
            "network_win_fraction_vs_nonlinear_z": float(np.mean(gain > 0)),
            "network_win_fraction_vs_native": float(np.mean(candidate_values < native_values)),
            "fold_candidate_mse": fold_candidate,
            "fold_nonlinear_z_mse": fold_nonlinear,
            "fold_network_win_fractions": fold_wins,
            "fold_results": fold_results,
            "per_network": rows,
        },
        "parity": {"folds": parity_receipts, "final_all688": final_parity},
        "gates": {
            "scientific": scientific_gates,
            "scientific_pass": scientific_pass,
            "absolute": absolute_gates,
            "submission_worthy_pending_literal_meter": submission_worthy,
        },
        "economics": {
            "d3_board_multiplier_328620": D3_BOARD_MULTIPLIER_328620,
            "c8_incremental_upper_flops": C8_TOTAL_INCREMENTAL_VS_D3_UPPER,
            "c8_d3_structural_multiplier": C8_D3_STRUCTURAL_MULTIPLIER,
            "raw_threshold_to_beat_d4": D4_BOARD_ADJUSTED_SCORE / C8_D3_STRUCTURAL_MULTIPLIER,
            "projected_adjusted_score": projected_adjusted,
            "d4_board_adjusted_score": D4_BOARD_ADJUSTED_SCORE,
            "literal_integrated_meter_required": True,
        },
        "training": {
            "fold_base_reconstruction_traces": base_traces,
            "fold_adaptation_traces": adaptation_traces,
            "fold_training_rms": rms_records,
            "final_trace": final_trace,
        },
        "deployability": {
            "incremental_vs_nonlinear_z_upper_flops": 100_159_488,
            "incremental_vs_compute_only_d3_upper_flops": C8_TOTAL_INCREMENTAL_VS_D3_UPPER,
            "live_state_bytes_float32": 1024 * CHANNELS * 4,
            "parameter_bytes_float32": PARAMETERS * 4,
            "native_row_storage_bytes": native_storage,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "v1_terminal_sha256": V1_TERMINAL_SHA256,
            "nonlinear_z_program_sha256": BASE_PROGRAM_SHA256,
            "nonlinear_z_result_sha256": BASE_RESULT_SHA256,
            "nonlinear_z_model_sha256": BASE_MODEL_SHA256,
            "native_result_sha256": NATIVE_RESULT_SHA256,
            "native_model_sha256": NATIVE_MODEL_SHA256,
            "j_source_sha256": J_SOURCE_SHA256,
            "q_source_sha256": Q_SOURCE_SHA256,
            "d3_gate_sha256": D3_GATE_SHA256,
            "d4_gate_sha256": D4_GATE_SHA256,
            "promoted_model_sha256": model_hash,
        },
        "access_declarations": {
            "role": "observer_fit", "observer_selection_targets_opened": [],
            "board_opened": False, "submission": False, "names_opened": [],
        },
    }
    write_json_once(RESULT, result)
    replace_json(RUN_STATE, {
        "schema": "q24-c8v2-run-state-v1", "status": "COMPLETE",
        "result_sha256": sha256(RESULT), "model_sha256": model_hash,
    })
    print(json.dumps({"status": result["status"], "crossfit": result["crossfit"], "economics": result["economics"]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-audit", action="store_true")
    arguments = parser.parse_args()
    if arguments.static_audit:
        print(json.dumps(static_audit(), indent=2, sort_keys=True))
    else:
        main_run()


if __name__ == "__main__":
    main()
