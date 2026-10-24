#!/usr/bin/env python3
"""Fold-pure suffix-global-context discriminator over frozen C8v2.

The only new information is a target-blind 112-vector computed once per
network/layer and broadcast into the existing suffix source cell.  Every fold
first reconstructs its nonlinear-Z model and C8v2 parent exactly, then checks
the held predictions against the persisted C8v2 OOF authority.  The parent is
frozen throughout context training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
C8_PROGRAM = P2 / "public_q24_c8_nonlinearz_init_v2/run_c8v2_oof.py"
C8_RESULT = P2 / "public_q24_c8_nonlinearz_init_v2/RESULT.json"
C8_MODEL = P2 / "public_q24_c8_nonlinearz_init_v2/C8V2_AUTONOMOUS_STATE.pt"
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
STATIC_AUDIT = HERE / "STATIC_AUDIT.json"
RUN_STATE = HERE / "RUN_STATE.json"
RESULT = HERE / "RESULT.json"
MODEL = HERE / "SUFFIX_GLOBAL_CONTEXT.pt"

C8_PROGRAM_SHA256 = "17e54c14d41eed964f2c1ef0c1513bbf788fd414e7438ca56b465151e442f283"
C8_RESULT_SHA256 = "3fa6b70c9b2f0cc799470a7443b488b3f9cfff48ad652c134c429282827f472a"
C8_MODEL_SHA256 = "60c6efb2238586f03ede999b2870630f33ac17621f7bfa219acbd63e9c14f719"
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
WIDTH = 1024
C8_CHANNELS = 8
C8_EXTRAS = 7
CONTEXT_INPUTS = 112
CONTEXT_HIDDEN = 32
CONTEXT_LATENT = 16
LOCAL_HIDDEN = 64
PARENT_PARAMETERS = 10_544
CONTEXT_PARAMETERS = 10_288
TOTAL_PARAMETERS = PARENT_PARAMETERS + CONTEXT_PARAMETERS
LOCAL_ANCHOR_WEIGHT = 0.05
PHASE1_EPOCHS = 4
PHASE1_LR = 3.0e-4
PHASE1_MIN_LR = 1.0e-4
PHASE2_EPOCHS = 8
PHASE2_LR = 2.0e-4
PHASE2_MIN_LR = 5.0e-5
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
MODEL_SEED = 0x610BA17C
MIN_RELATIVE_GAIN = 0.0025
MIN_IMPROVING_FOLDS = 3
MIN_PAIRED_Z = 2.0
MIN_NETWORK_WIN_FRACTION = 0.55
MIN_FOLD_WIN_FRACTION = 0.50
MAX_RSS_BYTES = 8 * (1 << 30)
MEASURED_NATIVE_AUTHORITY_MSE = 2.9736792457285063e-8
LEGACY_SCIENTIFIC_ADMISSION_MSE = 2.9672528546945005e-8
D4_BOARD_ADJUSTED_SCORE = 1.783621478716186e-8
D3_BOARD_MULTIPLIER_328620 = 0.6377160962460696
C8_INCREMENTAL_VS_D3_UPPER = 758_625_259
CONTEXT_INCREMENTAL_UPPER_FLOPS = 2_500_000
TOTAL_INCREMENTAL_VS_D3_UPPER = (
    C8_INCREMENTAL_VS_D3_UPPER + CONTEXT_INCREMENTAL_UPPER_FLOPS
)
STRUCTURAL_MULTIPLIER = (
    D3_BOARD_MULTIPLIER_328620
    + TOTAL_INCREMENTAL_VS_D3_UPPER / 2_199_023_255_552
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


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def enforce_rss(where: str) -> int:
    value = resident_bytes()
    if value >= MAX_RSS_BYTES:
        raise MemoryError(f"RSS gate failed at {where}: {value}")
    return value


def verify_static_bindings() -> dict[str, object]:
    observed = {
        "program_sha256": sha256(Path(__file__)),
        "c8v2_program_sha256": sha256(C8_PROGRAM),
        "c8v2_result_sha256": sha256(C8_RESULT),
        "c8v2_model_sha256": sha256(C8_MODEL),
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
        "c8v2_program_sha256": C8_PROGRAM_SHA256,
        "c8v2_result_sha256": C8_RESULT_SHA256,
        "c8v2_model_sha256": C8_MODEL_SHA256,
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
    if prereg.get("status") != "FROZEN_BEFORE_ANY_SUFFIX_GLOBAL_CONTEXT_TARGET_OPEN":
        raise AssertionError("suffix-global-context preregistration is not frozen")
    for key, value in observed.items():
        if prereg.get("bindings", {}).get(key) != value:
            raise AssertionError(f"preregistered binding changed: {key}")
    return prereg


def context112_from_live(
    parent,
    raw: torch.Tensor,
    sigma: torch.Tensor,
    base_state: torch.Tensor,
    base_q: torch.Tensor,
    z_state: torch.Tensor,
    j_messages: torch.Tensor,
    j_energies: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Build the fixed target-blind 112-vector for one live suffix layer."""
    slot = SUFFIX.index(layer)
    tail = torch.clamp(
        (raw[..., 1:] - parent.centers[layer]) / parent.scales[layer], -8.0, 8.0
    )
    raw_standard = torch.cat((raw[..., :1], tail), dim=-1)
    z_standard = torch.clamp(
        (z_state - parent.z_center[slot]) / parent.z_scale[slot], -8.0, 8.0
    )

    raw_mean = torch.mean(raw_standard, dim=0)
    raw_rms = torch.sqrt(torch.mean(raw_standard.square(), dim=0) + 2.0**-30)
    z_mean = torch.mean(z_standard, dim=0)
    z_rms = torch.sqrt(torch.mean(z_standard.square(), dim=0) + 2.0**-30)
    raw_z = torch.mean(raw_standard * z_standard, dim=0)

    rms = parent.layer_rms[layer]
    a_standard = torch.clamp(base_state / rms, -8.0, 8.0)
    q_standard = torch.clamp(
        base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0
    )
    phi = torch.clamp(raw[..., 5], -8.0, 8.0)
    log_sigma = torch.clamp(
        torch.log2(torch.clamp(sigma, min=2.0**-30)), -8.0, 8.0
    )

    def mean_rms(field: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (torch.mean(field), torch.sqrt(torch.mean(field.square()) + 2.0**-30))
        )

    coherence = torch.clamp(
        j_messages
        / torch.sqrt(j_energies + 1.0e-4 * rms.square()),
        -8.0,
        8.0,
    ).mean(dim=0)
    action = torch.cat(
        (
            mean_rms(a_standard),
            mean_rms(q_standard),
            mean_rms(phi),
            mean_rms(log_sigma),
            coherence,
        )
    )
    context = torch.cat((raw_mean, raw_rms, z_mean, z_rms, raw_z, action))
    if tuple(context.shape) != (CONTEXT_INPUTS,):
        raise AssertionError(f"context shape changed: {tuple(context.shape)}")
    return torch.clamp(context, -8.0, 8.0)


class GlobalContextC8(nn.Module):
    """Frozen C8v2 plus a pooled-context preactivation shift."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.context1 = nn.Linear(CONTEXT_INPUTS, CONTEXT_HIDDEN)
        self.context2 = nn.Linear(CONTEXT_HIDDEN, CONTEXT_LATENT)
        self.context_adapter = nn.Parameter(
            torch.zeros(len(SUFFIX), CONTEXT_LATENT, LOCAL_HIDDEN)
        )
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        if sum(value.numel() for value in self.parameters()) != TOTAL_PARAMETERS:
            raise AssertionError("global-context parameter count changed")

    def context_latent(self, context: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.context1(context))
        return torch.nn.functional.silu(self.context2(hidden))

    def source(
        self,
        raw,
        sigma,
        base_state,
        base_q,
        z_state,
        message,
        context,
        layer: int,
        *,
        context_enabled: bool,
    ):
        if not context_enabled:
            return self.parent.source(
                raw,
                sigma,
                base_state,
                base_q,
                z_state,
                message,
                layer,
                adapter_enabled=True,
            )
        features = self.parent.base_features(
            raw, sigma, base_state, base_q, z_state, layer
        )
        preactivation = self.parent.base_linear(features)
        rms = self.parent.layer_rms[layer]
        preactivation = preactivation + self.parent.message_linear(
            torch.clamp(message / rms, -8.0, 8.0)
        )
        slot = SUFFIX.index(layer)
        shift = self.context_latent(context) @ self.context_adapter[slot]
        preactivation = preactivation + shift
        hidden = torch.nn.functional.silu(preactivation)
        hidden = torch.nn.functional.silu(self.parent.trunk2(hidden))
        scalar_standard = torch.einsum(
            "...h,h->...", hidden, self.parent.scalar_weight[slot]
        ) + self.parent.scalar_bias[slot]
        extra_standard = (
            hidden @ self.parent.extra_weight[slot] + self.parent.extra_bias[slot]
        )
        physical = sigma * rms
        return physical * scalar_standard, physical[..., None] * extra_standard

    def phase_parameters(self, *, encoder_enabled: bool):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.context_adapter.requires_grad_(True)
        if encoder_enabled:
            for parameter in self.context1.parameters():
                parameter.requires_grad_(True)
            for parameter in self.context2.parameters():
                parameter.requires_grad_(True)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def c8_oof_authority() -> tuple[dict[str, object], np.ndarray]:
    result = json.loads(C8_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "q24-c8v2-exact-nonlinearz-initialized-oof-v1"
        or result.get("status") != "PROMOTED_SCIENTIFIC_ONLY"
        or result.get("bindings", {}).get("program_sha256") != C8_PROGRAM_SHA256
        or result.get("bindings", {}).get("promoted_model_sha256") != C8_MODEL_SHA256
    ):
        raise AssertionError("C8v2 OOF authority changed")
    values = np.empty(688, dtype=np.float64)
    rows = result.get("crossfit", {}).get("per_network", [])
    if len(rows) != 688:
        raise AssertionError("C8v2 OOF coverage changed")
    for row in rows:
        values[int(row["role_rank"])] = float(row["candidate_final_mse"])
    if not np.isclose(
        float(np.mean(values)),
        float(result["crossfit"]["candidate_oof_final_mse"]),
        rtol=0.0,
        atol=1.0e-20,
    ):
        raise AssertionError("C8v2 OOF mean changed")
    return result, values


def load_final_parent(v2, base, alpha):
    nonlinear = v2.load_final_nonlinear(base, alpha)
    parent = v2.ExactInitializedC8(nonlinear)
    artifact = torch.load(C8_MODEL, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "q24-c8v2-exact-nonlinearz-initialized-state-v1":
        raise AssertionError("C8v2 model schema changed")
    parent.load_state_dict(artifact["state_dict"], strict=True)
    parent.eval()
    return parent


def prepare_context_bank(parent, alpha, dataset, prepared) -> tuple[np.ndarray, dict[str, object]]:
    """Materialize target-blind context only; no target/error argument is accepted."""
    bank = np.empty((688, len(SUFFIX), CONTEXT_INPUTS), dtype=np.float32)
    maxima = np.zeros(CONTEXT_INPUTS, dtype=np.float64)
    alpha.eval()
    with torch.no_grad():
        for rank in range(688):
            row = dataset.load(rank)
            raw = torch.from_numpy(
                np.ascontiguousarray(row.raw_locals, dtype=np.float32)
            )[None]
            sigma = torch.from_numpy(
                np.ascontiguousarray(row.sigma, dtype=np.float32)
            )[None]
            weights = torch.from_numpy(
                np.ascontiguousarray(
                    np.load(row.weights_path, mmap_mode="r", allow_pickle=False),
                    dtype=np.float32,
                )
            )[None]
            prediction, states, energies = alpha(
                raw, sigma, weights, exposure=1.0, return_states=True
            )
            for slot, layer in enumerate(SUFFIX):
                prior = states[0, layer - 1]
                message = weights[0, layer].T @ prior
                message = raw[0, layer, :, 5, None] * message
                if not torch.equal(prediction[0, layer], states[0, layer, :, 0]):
                    raise AssertionError("alpha-J primary/state mismatch")
                context = context112_from_live(
                    parent,
                    raw[0, layer],
                    sigma[0, layer],
                    torch.from_numpy(prepared.base_state[rank, slot]),
                    torch.from_numpy(prepared.base_q[rank, slot]),
                    torch.from_numpy(prepared.z_suffix[rank, slot]),
                    message,
                    energies[0, layer],
                    layer,
                )
                bank[rank, slot] = context.cpu().numpy()
                maxima = np.maximum(maxima, np.abs(bank[rank, slot]))
            if (rank + 1) % 32 == 0:
                print(json.dumps({"stage": "prepare_context112", "rows": rank + 1}), flush=True)
            enforce_rss(f"context preparation rank {rank}")
    if not np.isfinite(bank).all():
        raise FloatingPointError("context bank is nonfinite")
    return bank, {
        "depends_on_targets": False,
        "shape": list(bank.shape),
        "bytes": bank.nbytes,
        "maximum_absolute_by_feature": maxima.tolist(),
        "feature_order": [
            "mean standardized raw20[20]",
            "RMS standardized raw20[20]",
            "mean standardized transported Z20[20]",
            "RMS standardized transported Z20[20]",
            "mean matched raw20*Z20[20]",
            "mean/RMS normalized A, qJ, phi, log2(sigma)[8]",
            "mean live alpha-J message/energy coherence by four channels[4]",
        ],
    }


def target_blind_context_normalization(alpha, prepared):
    """Expose only inherited normalization buffers, never final fitted weights."""
    return SimpleNamespace(
        centers=alpha.centers.detach(),
        scales=alpha.scales.detach(),
        layer_rms=alpha.layer_rms.detach(),
        z_center=torch.from_numpy(prepared.z_center.copy()),
        z_scale=torch.from_numpy(prepared.z_scale.copy()),
    )


def reconstruct_fold_parent(
    v2, base, q, alpha, prepared, dataset, truth_suffix, train, held, fold, nonlinear_oof, c8_oof
):
    nonlinear, q_rms, nonlinear_trace, nonlinear_max = v2.fold_base_reconstruction(
        base, q, alpha, prepared, dataset, train, held, fold, nonlinear_oof
    )
    parent = v2.ExactInitializedC8(nonlinear)
    _, c8_trace = v2.adapt(parent, prepared, dataset, truth_suffix, train, fold)
    scored = v2.score_candidate(parent, prepared, dataset, truth_suffix, held)
    observed = np.asarray([float(row["candidate_final_mse"]) for row in scored])
    maximum = float(np.max(np.abs(observed - c8_oof[held])))
    if maximum > 5.0e-20:
        raise AssertionError(("fold C8v2 reconstruction mismatch", fold, maximum))
    return parent, nonlinear_trace, c8_trace, nonlinear_max, maximum


def rollout(candidate, prepared, dataset, context_bank, rank: int, *, context_enabled: bool):
    weights_np = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
    scalar_state = torch.zeros(WIDTH, dtype=torch.float32)
    extra_state = torch.zeros((WIDTH, C8_EXTRAS), dtype=torch.float32)
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
            torch.from_numpy(context_bank[rank, slot]),
            layer,
            context_enabled=context_enabled,
        )
        scalar_state = scalar_message + scalar_source
        extra_state = extra_message + extra_source
        scalar_states.append(scalar_state)
    return scalar_states


def training_rms(truth_suffix: np.ndarray, train: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.mean(truth_suffix[train].astype(np.float64) ** 2, axis=(0, 2)))
    return np.maximum(values, 2.0**-20).astype(np.float32)


def train_phase(
    candidate,
    prepared,
    dataset,
    context_bank,
    truth_suffix,
    train,
    rms,
    stream_id,
    *,
    phase: str,
    epochs: int,
    learning_rate: float,
    minimum_lr: float,
    encoder_enabled: bool,
):
    parameters = candidate.phase_parameters(encoder_enabled=encoder_enabled)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train), eta_min=minimum_lr
    )
    rng = np.random.default_rng(
        (
            MODEL_SEED
            ^ (int(stream_id) * 0x9E3779B1)
            ^ (0x51515151 if encoder_enabled else 0xA0A0A0A0)
        )
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
            states = rollout(
                candidate, prepared, dataset, context_bank, rank, context_enabled=True
            )
            truth = torch.from_numpy(truth_suffix[rank])
            scale = torch.from_numpy(rms)[:, None]
            endpoint = torch.mean(((states[-1] - truth[-1]) / scale[-1]).square())
            local = torch.stack(
                [
                    torch.mean(((states[slot] - truth[slot]) / scale[slot]).square())
                    for slot in range(len(SUFFIX))
                ]
            ).mean()
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


def adapt_context(candidate, prepared, dataset, context_bank, truth_suffix, train, stream_id):
    rms = training_rms(truth_suffix, train)
    phase1 = train_phase(
        candidate,
        prepared,
        dataset,
        context_bank,
        truth_suffix,
        train,
        rms,
        stream_id,
        phase="layer_adapter_only",
        epochs=PHASE1_EPOCHS,
        learning_rate=PHASE1_LR,
        minimum_lr=PHASE1_MIN_LR,
        encoder_enabled=False,
    )
    phase2 = train_phase(
        candidate,
        prepared,
        dataset,
        context_bank,
        truth_suffix,
        train,
        rms,
        stream_id,
        phase="context_encoder_and_adapter",
        epochs=PHASE2_EPOCHS,
        learning_rate=PHASE2_LR,
        minimum_lr=PHASE2_MIN_LR,
        encoder_enabled=True,
    )
    return rms, phase1 + phase2


def score_candidate(candidate, prepared, dataset, context_bank, truth_suffix, held, *, enabled=True):
    rows = []
    candidate.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            final = (
                rollout(
                    candidate,
                    prepared,
                    dataset,
                    context_bank,
                    rank,
                    context_enabled=enabled,
                )[-1]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            residual = truth_suffix[rank, -1].astype(np.float64) - final
            rows.append(
                {
                    "role_rank": rank,
                    "mlp_id": int(dataset.mlp_ids[rank]),
                    "candidate_final_mse": float(np.mean(residual * residual)),
                }
            )
    return rows


def static_audit() -> dict[str, object]:
    verify_static_bindings()
    v2 = load_module("global_context_static_c8v2", C8_PROGRAM)
    base = load_module("global_context_static_nonlinear", BASE_PROGRAM)
    base_runner = load_module("global_context_static_full688", base.BASE_PROGRAM)
    j = load_module("global_context_static_j", J_SOURCE)
    alpha = base_runner.load_alpha_j(j)
    parent = load_final_parent(v2, base, alpha)
    torch.manual_seed(MODEL_SEED)
    candidate = GlobalContextC8(parent)
    generator = torch.Generator().manual_seed(0x610BA017)
    raw = torch.randn((64, 20), generator=generator)
    raw[:, 5] = torch.sigmoid(raw[:, 5])
    sigma = torch.rand(64, generator=generator) + 0.25
    state = torch.randn(64, generator=generator) * 1.0e-4
    local_q = torch.randn(64, generator=generator) * 1.0e-4
    zstate = torch.randn((64, 20), generator=generator)
    message = torch.randn((64, C8_CHANNELS), generator=generator) * 1.0e-4
    j_message = torch.randn((64, 4), generator=generator) * 1.0e-4
    j_energy = torch.rand((64, 4), generator=generator) * 1.0e-8
    context = context112_from_live(
        parent, raw, sigma, state, local_q, zstate, j_message, j_energy, 10
    )
    with torch.no_grad():
        inherited = parent.source(
            raw, sigma, state, local_q, zstate, message, 10, adapter_enabled=True
        )
        disabled = candidate.source(
            raw,
            sigma,
            state,
            local_q,
            zstate,
            message,
            context,
            10,
            context_enabled=False,
        )
        enabled = candidate.source(
            raw,
            sigma,
            state,
            local_q,
            zstate,
            message,
            context,
            10,
            context_enabled=True,
        )
    phase1_count = sum(value.numel() for value in candidate.phase_parameters(encoder_enabled=False))
    phase2_count = sum(value.numel() for value in candidate.phase_parameters(encoder_enabled=True))
    signature = inspect.signature(prepare_context_bank)
    forbidden = {"target", "targets", "truth", "error"}
    checks = {
        "context_shape_exactly_112": tuple(context.shape) == (CONTEXT_INPUTS,),
        "context_finite": bool(torch.isfinite(context).all()),
        "parameters_exactly_20832": sum(v.numel() for v in candidate.parameters()) == TOTAL_PARAMETERS,
        "context_parameters_exactly_10288": CONTEXT_PARAMETERS == 10_288,
        "zero_adapter_exact": bool(torch.count_nonzero(candidate.context_adapter) == 0),
        "disabled_scalar_torch_equal": bool(torch.equal(inherited[0], disabled[0])),
        "disabled_extras_torch_equal": bool(torch.equal(inherited[1], disabled[1])),
        "zero_adapter_enabled_scalar_torch_equal": bool(torch.equal(inherited[0], enabled[0])),
        "zero_adapter_enabled_extras_torch_equal": bool(torch.equal(inherited[1], enabled[1])),
        "phase1_only_6144_adapter_parameters": phase1_count == 6_144,
        "phase2_exactly_10288_context_parameters": phase2_count == CONTEXT_PARAMETERS,
        "parent_frozen_in_phase2": all(not p.requires_grad for p in candidate.parent.parameters()),
        "context_preparation_signature_has_no_target": not (forbidden & set(signature.parameters)),
        "structural_raw_threshold_below_native": (
            D4_BOARD_ADJUSTED_SCORE / STRUCTURAL_MULTIPLIER
            < MEASURED_NATIVE_AUTHORITY_MSE
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "schema": "q24-suffix-global-context-static-audit-v1",
        "status": "STATIC_GO",
        "program_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "checks": checks,
        "economics": {
            "context_incremental_upper_flops": CONTEXT_INCREMENTAL_UPPER_FLOPS,
            "total_incremental_vs_d3_upper_flops": TOTAL_INCREMENTAL_VS_D3_UPPER,
            "structural_multiplier": STRUCTURAL_MULTIPLIER,
            "raw_threshold_to_beat_d4": D4_BOARD_ADJUSTED_SCORE / STRUCTURAL_MULTIPLIER,
        },
        "targets_opened": [],
        "training_started": False,
    }


def main_run() -> None:
    if RUN_STATE.exists() or RESULT.exists() or MODEL.exists():
        raise FileExistsError("suffix-global-context one-shot has already started")
    verify_static_bindings()
    c8_result, c8_oof = c8_oof_authority()
    v2 = load_module("global_context_c8v2", C8_PROGRAM)
    base = load_module("global_context_nonlinear", BASE_PROGRAM)
    base_runner = load_module("global_context_full688", base.BASE_PROGRAM)
    q = load_module("global_context_q", Q_SOURCE)
    j = load_module("global_context_j", J_SOURCE)
    v1 = load_module("global_context_native_recompute", v2.V1_PROGRAM)
    records, native_storage = base_runner.verify_native_rows()
    folds = base_runner.balanced_folds(records)
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("fold authority changed")
    write_json_once(
        RUN_STATE,
        {
            "schema": "q24-suffix-global-context-run-state-v1",
            "status": "RUNNING_ONE_SHOT",
            "pid": os.getpid(),
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "target_columns_opened": [],
        },
    )

    targets = base_runner.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(base_runner.row_dir(rank) / "host.npy", allow_pickle=False)
                for rank in range(688)
            ]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    dataset = base_runner.All688Dataset(records)
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    alpha = base_runner.load_alpha_j(j)
    prepared = base.prepare_all_with_z(base_runner, j, alpha, dataset, error)
    truth_suffix = np.ascontiguousarray(
        error[:, SUFFIX] - prepared.base_state, dtype=np.float32
    )
    nonlinear_result, nonlinear_oof = v2.nonlinear_oof_authority()
    native_values = v1.recompute_native_bank(
        j, v1.load_native_model(j), dataset, error
    )
    context_normalization = target_blind_context_normalization(alpha, prepared)
    context_bank, context_diagnostic = prepare_context_bank(
        context_normalization, alpha, dataset, prepared
    )
    del targets, hosts, error, context_normalization
    gc.collect()
    enforce_rss("post preparation")

    per_network: list[dict[str, object] | None] = [None] * 688
    fold_results = []
    reconstruction_receipts = []
    nonlinear_traces = []
    parent_traces = []
    context_traces = []
    rms_records = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train = ALL_RANKS[folds != fold]
        held = ALL_RANKS[folds == fold]
        parent, nonlinear_trace, parent_trace, nonlinear_max, parent_max = reconstruct_fold_parent(
            v2,
            base,
            q,
            alpha,
            prepared,
            dataset,
            truth_suffix,
            train,
            held,
            fold,
            nonlinear_oof,
            c8_oof,
        )
        torch.manual_seed((MODEL_SEED ^ (fold * 0x9E3779B1)) & 0xFFFF_FFFF)
        candidate = GlobalContextC8(parent)
        epoch0_rows = score_candidate(
            candidate,
            prepared,
            dataset,
            context_bank,
            truth_suffix,
            held,
            enabled=True,
        )
        epoch0_values = np.asarray(
            [float(row["candidate_final_mse"]) for row in epoch0_rows]
        )
        epoch0_max = float(np.max(np.abs(epoch0_values - c8_oof[held])))
        if epoch0_max > 5.0e-20:
            raise AssertionError(("epoch0 context parity failed", fold, epoch0_max))
        receipt = {
            "fold": fold,
            "nonlinear_z_reconstruction_maximum_mse_difference": nonlinear_max,
            "c8v2_reconstruction_maximum_mse_difference": parent_max,
            "zero_context_epoch0_maximum_mse_difference": epoch0_max,
            "parent_frozen": True,
        }
        reconstruction_receipts.append(receipt)
        print(json.dumps({"stage": "epoch0_parity", **receipt}, sort_keys=True), flush=True)
        rms, trace = adapt_context(
            candidate,
            prepared,
            dataset,
            context_bank,
            truth_suffix,
            train,
            fold,
        )
        rows = score_candidate(
            candidate,
            prepared,
            dataset,
            context_bank,
            truth_suffix,
            held,
        )
        values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
        for row in rows:
            per_network[int(row["role_rank"])] = row
        summary = {
            "fold": fold,
            "training_rows": len(train),
            "held_rows": held.tolist(),
            "candidate_final_mse": float(np.mean(values)),
            "c8v2_oof_final_mse": float(np.mean(c8_oof[held])),
            "nonlinear_z_oof_final_mse": float(np.mean(nonlinear_oof[held])),
            "native_bank_same688_final_mse": float(np.mean(native_values[held])),
            "win_fraction_vs_c8v2": float(np.mean(values < c8_oof[held])),
            "win_fraction_vs_nonlinear_z": float(np.mean(values < nonlinear_oof[held])),
            "win_fraction_vs_native_bank": float(np.mean(values < native_values[held])),
        }
        fold_results.append(summary)
        nonlinear_traces.extend({"fold": fold, **row} for row in nonlinear_trace)
        parent_traces.extend({"fold": fold, **row} for row in parent_trace)
        context_traces.extend({"fold": fold, **row} for row in trace)
        rms_records.append(rms.tolist())
        print(json.dumps({"stage": "fold_score", **summary}, sort_keys=True), flush=True)
        del parent, candidate
        gc.collect()

    if any(row is None for row in per_network):
        raise AssertionError("global-context OOF coverage incomplete")
    rows = [row for row in per_network if row is not None]
    values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
    gain = c8_oof - values
    candidate_mse = float(np.mean(values))
    c8_mse = float(np.mean(c8_oof))
    nonlinear_mse = float(np.mean(nonlinear_oof))
    native_mse = float(np.mean(native_values))
    relative_gain = (c8_mse - candidate_mse) / c8_mse
    gain_se = float(np.std(gain, ddof=1) / math.sqrt(688))
    paired_z = float(np.mean(gain) / gain_se) if gain_se > 0 else float("inf")
    fold_candidate = [float(np.mean(values[folds == fold])) for fold in range(FOLDS)]
    fold_c8 = [float(np.mean(c8_oof[folds == fold])) for fold in range(FOLDS)]
    fold_wins = [float(np.mean(gain[folds == fold] > 0)) for fold in range(FOLDS)]
    improving_folds = sum(a < b for a, b in zip(fold_candidate, fold_c8, strict=True))
    protected = np.asarray(
        c8_result["crossfit"].get(
            "legacy_protected40_role_ranks",
            nonlinear_result["crossfit"]["legacy_protected40_oof"]["role_ranks"],
        ),
        dtype=np.int64,
    )
    projected_adjusted = candidate_mse * STRUCTURAL_MULTIPLIER
    scientific_gates = {
        "finite_688_oof": bool(np.isfinite(values).all()),
        "relative_gain_vs_c8v2_at_least_0_25pct": bool(relative_gain >= MIN_RELATIVE_GAIN),
        "at_least_three_folds_improve": bool(improving_folds >= MIN_IMPROVING_FOLDS),
        "paired_network_z_at_least_2": bool(paired_z >= MIN_PAIRED_Z),
        "network_wins_vs_c8v2_at_least_55pct": bool(
            np.mean(gain > 0) >= MIN_NETWORK_WIN_FRACTION
        ),
        "every_fold_wins_at_least_50pct": bool(min(fold_wins) >= MIN_FOLD_WIN_FRACTION),
        "candidate_below_nonlinear_z": bool(candidate_mse < nonlinear_mse),
        "legacy_protected40_does_not_worsen_vs_c8v2": bool(
            np.mean(values[protected]) <= np.mean(c8_oof[protected])
        ),
        "all_fold_parent_reconstructions_and_epoch0_parity_pass": bool(
            len(reconstruction_receipts) == FOLDS
        ),
    }
    scientific_pass = all(scientific_gates.values())
    absolute_gates = {
        "candidate_below_recomputed_native_bank": bool(candidate_mse < native_mse),
        "native_recompute_matches_authority": bool(
            np.isclose(native_mse, MEASURED_NATIVE_AUTHORITY_MSE, rtol=0.0, atol=2.0e-15)
        ),
        "candidate_below_legacy_scientific_admission": bool(
            candidate_mse < LEGACY_SCIENTIFIC_ADMISSION_MSE
        ),
        "network_wins_vs_native_at_least_55pct": bool(
            np.mean(values < native_values) >= MIN_NETWORK_WIN_FRACTION
        ),
        "structural_adjusted_beats_d4_328622": bool(
            projected_adjusted < D4_BOARD_ADJUSTED_SCORE
        ),
    }
    submission_worthy = scientific_pass and all(absolute_gates.values())

    model_hash = None
    final_trace = []
    if scientific_pass:
        parent = load_final_parent(v2, base, alpha)
        torch.manual_seed((MODEL_SEED ^ (FOLDS * 0x9E3779B1)) & 0xFFFF_FFFF)
        candidate = GlobalContextC8(parent)
        final_rms, final_trace = adapt_context(
            candidate,
            prepared,
            dataset,
            context_bank,
            truth_suffix,
            ALL_RANKS,
            FOLDS,
        )
        artifact = {
            "schema": "q24-suffix-global-context-over-c8v2-state-v1",
            "state_dict": candidate.state_dict(),
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "context_inputs": CONTEXT_INPUTS,
            "context_latent": CONTEXT_LATENT,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "c8v2_result_sha256": C8_RESULT_SHA256,
            "training_rms": final_rms,
        }
        temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary)
        temporary.replace(MODEL)
        model_hash = sha256(MODEL)

    result = {
        "schema": "q24-suffix-global-context-over-c8v2-oof-v1",
        "status": (
            "PROMOTED_SUBMISSION_WORTHY_PENDING_LITERAL_METER"
            if submission_worthy
            else "PROMOTED_SCIENTIFIC_ONLY"
            if scientific_pass
            else "CLOSED_NO_BROAD_OOF_GAIN"
        ),
        "fixed_rule": {
            "architecture": "frozen fold-pure C8v2 plus shared112->32->16 context and six layer-specific16->64 broadcast shifts",
            "parameters_total": TOTAL_PARAMETERS,
            "parameters_context": CONTEXT_PARAMETERS,
            "features": context_diagnostic["feature_order"],
            "phase1": "4 epochs layer adapters only,3e-4 to1e-4",
            "phase2": "8 epochs context encoder plus adapters,2e-4 to5e-5",
            "parent_training": "frozen after exact fold-pure reconstruction",
            "loss": f"endpoint + {LOCAL_ANCHOR_WEIGHT} local channel0 anchor",
        },
        "crossfit": {
            "candidate_oof_final_mse": candidate_mse,
            "c8v2_oof_final_mse": c8_mse,
            "nonlinear_z_oof_final_mse": nonlinear_mse,
            "recomputed_native_bank_same688_final_mse": native_mse,
            "relative_gain_vs_c8v2": relative_gain,
            "paired_network_z": paired_z,
            "network_win_fraction_vs_c8v2": float(np.mean(gain > 0)),
            "network_win_fraction_vs_native": float(np.mean(values < native_values)),
            "fold_candidate_mse": fold_candidate,
            "fold_c8v2_mse": fold_c8,
            "fold_network_win_fractions": fold_wins,
            "fold_results": fold_results,
            "per_network": rows,
        },
        "parity": reconstruction_receipts,
        "gates": {
            "scientific": scientific_gates,
            "scientific_pass": scientific_pass,
            "absolute": absolute_gates,
            "submission_worthy_pending_literal_meter": submission_worthy,
        },
        "economics": {
            "d3_board_multiplier_328620": D3_BOARD_MULTIPLIER_328620,
            "c8_incremental_vs_d3_upper_flops": C8_INCREMENTAL_VS_D3_UPPER,
            "context_incremental_upper_flops": CONTEXT_INCREMENTAL_UPPER_FLOPS,
            "total_incremental_vs_d3_upper_flops": TOTAL_INCREMENTAL_VS_D3_UPPER,
            "structural_multiplier": STRUCTURAL_MULTIPLIER,
            "raw_threshold_to_beat_d4": D4_BOARD_ADJUSTED_SCORE / STRUCTURAL_MULTIPLIER,
            "projected_adjusted_score": projected_adjusted,
            "d4_board_adjusted_score": D4_BOARD_ADJUSTED_SCORE,
            "literal_integrated_meter_required": True,
        },
        "target_blind_context_diagnostic": context_diagnostic,
        "training": {
            "fold_nonlinear_reconstruction_traces": nonlinear_traces,
            "fold_c8v2_reconstruction_traces": parent_traces,
            "fold_context_traces": context_traces,
            "fold_training_rms": rms_records,
            "final_trace": final_trace,
        },
        "deployability": {
            "incremental_context_upper_flops": CONTEXT_INCREMENTAL_UPPER_FLOPS,
            "live_context112_bytes_float32": CONTEXT_INPUTS * 4,
            "live_context_bytes_float32": CONTEXT_LATENT * 4,
            "live_context_shift_bytes_float32": LOCAL_HIDDEN * 4,
            "incremental_live_context_total_bytes_float32": (
                CONTEXT_INPUTS + CONTEXT_LATENT + LOCAL_HIDDEN
            ) * 4,
            "context_parameter_bytes_float32": CONTEXT_PARAMETERS * 4,
            "training_context_cache_bytes": context_bank.nbytes,
            "native_row_storage_bytes": native_storage,
            "maximum_process_rss_bytes": enforce_rss("result"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "c8v2_program_sha256": C8_PROGRAM_SHA256,
            "c8v2_result_sha256": C8_RESULT_SHA256,
            "c8v2_model_sha256": C8_MODEL_SHA256,
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
            "role": "observer_fit",
            "observer_fit_targets_opened": ["mlp_id", "all_layer_means"],
            "observer_selection_targets_opened": [],
            "board_opened": False,
            "submission": False,
            "names_opened": [],
        },
    }
    write_json_once(RESULT, result)
    replace_json(
        RUN_STATE,
        {
            "schema": "q24-suffix-global-context-run-state-v1",
            "status": "COMPLETE",
            "result_sha256": sha256(RESULT),
            "model_sha256": model_hash,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "crossfit": result["crossfit"],
                "economics": result["economics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


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
