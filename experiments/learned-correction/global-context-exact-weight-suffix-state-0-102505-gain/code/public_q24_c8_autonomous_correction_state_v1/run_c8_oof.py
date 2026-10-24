#!/usr/bin/env python3
"""Frozen four-fold OOF test of an eight-channel autonomous q10:q15 state.

The candidate replaces the scalar nonlinear-Z suffix head.  It retains the
same alpha=.75 J state and exact target-blind transported raw20 bank, but adds
an eight-channel coordinate field C.  Each channel is transported through the
literal network weights before a shared pointwise cell supplies the next
state.  Channel zero is the reported correction.  Training is endpoint-led;
a fixed small local-state anchor prevents unidentifiable latent drift.
"""

from __future__ import annotations

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
MODEL = HERE / "C8_AUTONOMOUS_STATE.pt"

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
INPUTS = 50  # raw20 + A + local qJ + transported-Z20 + C-message8
HIDDEN = 64
PARAMETERS = 10_544
EPOCHS = 6
NETWORKS_PER_STEP = 1
LEARNING_RATE = 8.0e-4
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
LOCAL_ANCHOR_WEIGHT = 0.05
MODEL_SEED = 0x43385A11
MIN_RELATIVE_GAIN_VS_NONLINEAR_Z = 0.0025
MIN_IMPROVING_FOLDS = 3
MIN_PAIRED_Z = 2.0
MIN_NETWORK_WIN_FRACTION = 0.55
MIN_FOLD_WIN_FRACTION = 0.50
MAX_RSS_BYTES = 8 * (1 << 30)

# These are deliberately different authorities.  The first is the measured
# fixed-fit native-bank value on these exact 688 rows.  The second is a stricter
# legacy scientific admission threshold (0.9 * P2R0F2D OOF), not a measurement.
MEASURED_NATIVE_AUTHORITY_MSE = 2.9736792457285063e-8
LEGACY_SCIENTIFIC_ADMISSION_MSE = 2.9672528546945005e-8
D4_BOARD_RAW_MSE = 2.9409071835573285e-8
D4_BOARD_ADJUSTED_SCORE = 1.783621478716186e-8
D3_BOARD_MULTIPLIER_328620 = 0.6377160962460696
D3_22_BILLS = (1_387_818_002_786, 1_425_446_354_419, 1_386_800_025_067)
D1_22_BILLS = (1_574_258_714_978, 1_612_699_516_403, 1_574_053_187_051)
INCREMENTAL_C8_VS_22_UPPER = 661_925_888
C8_TOTAL_INCREMENTAL_VS_COMPUTE_ONLY_D3_UPPER = 758_625_259
C8_D3_MAX_COMPUTE_RATIO = max(
    (numerator + INCREMENTAL_C8_VS_22_UPPER) / denominator
    for numerator, denominator in zip(D3_22_BILLS, D1_22_BILLS, strict=True)
)
C8_D3_STRUCTURAL_BOARD_MULTIPLIER = (
    D3_BOARD_MULTIPLIER_328620
    + C8_TOTAL_INCREMENTAL_VS_COMPUTE_ONLY_D3_UPPER / 2_199_023_255_552
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def replace_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_static() -> dict[str, object]:
    expected = {
        "program_sha256": sha256(Path(__file__)),
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
    literals = {
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
    for key, wanted in literals.items():
        if expected[key] != wanted:
            raise AssertionError(f"literal authority changed: {key}")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_ANY_C8_TARGET_OPEN":
        raise AssertionError("c8 preregistration is not frozen")
    for key, observed in expected.items():
        if prereg.get("bindings", {}).get(key) != observed:
            raise AssertionError(f"preregistered binding changed: {key}")
    return prereg


def load_nonlinear_oof() -> tuple[dict[str, object], np.ndarray]:
    result = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PROMOTED_FIXED_FULL688_MODEL"
        or result.get("promotion_gate", {}).get("conjunctive_pass") is not True
        or result.get("bindings", {}).get("promoted_model_sha256") != BASE_MODEL_SHA256
    ):
        raise AssertionError("nonlinear-Z authority changed")
    rows = result.get("crossfit", {}).get("per_network", [])
    if len(rows) != 688:
        raise AssertionError("nonlinear-Z OOF coverage changed")
    values = np.empty(688, dtype=np.float64)
    for row in rows:
        values[int(row["role_rank"])] = float(row["candidate_final_mse"])
    return result, values


def load_native_model(j):
    result = json.loads(NATIVE_RESULT.read_text(encoding="utf-8"))
    measured = float(result["postfit_in_sample_diagnostic_not_used_for_selection"]["autonomous_final_mse"])
    if measured != MEASURED_NATIVE_AUTHORITY_MSE or result["bindings"]["model_sha256"] != NATIVE_MODEL_SHA256:
        raise AssertionError("measured native authority changed")
    artifact = torch.load(NATIVE_MODEL, map_location="cpu", weights_only=False)
    state = artifact["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].cpu().numpy(),
        state["scales"].cpu().numpy(),
        state["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def recompute_native_bank(j, native_model, dataset, error: np.ndarray) -> np.ndarray:
    values = np.empty(688, dtype=np.float64)
    with torch.no_grad():
        for rank in range(688):
            raw, sigma, weights, _ = j.load_batch(dataset, np.asarray([rank]), error)
            correction = native_model(raw, sigma, weights, exposure=1.0)
            residual = error[rank, -1].astype(np.float64) - correction[0, -1].cpu().numpy().astype(np.float64)
            values[rank] = float(np.mean(residual * residual))
            if (rank + 1) % 64 == 0:
                print(json.dumps({"stage": "native_bank_recompute", "rows": rank + 1}), flush=True)
    observed = float(np.mean(values))
    if not np.isclose(observed, MEASURED_NATIVE_AUTHORITY_MSE, rtol=0.0, atol=2e-15):
        raise AssertionError(("same-688 native-bank recomputation drift", observed))
    return values


class C8State(nn.Module):
    def __init__(self, alpha_model, nonlinear_state: dict[str, torch.Tensor]):
        super().__init__()
        self.register_buffer("centers", alpha_model.centers.detach().clone())
        self.register_buffer("scales", alpha_model.scales.detach().clone())
        self.register_buffer("layer_rms", alpha_model.layer_rms.detach().clone())
        self.register_buffer("z_center", nonlinear_state["z_center"].detach().clone())
        self.register_buffer("z_scale", nonlinear_state["z_scale"].detach().clone())
        self.trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        )
        self.source_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN, CHANNELS))
        self.source_bias = nn.Parameter(torch.zeros(len(SUFFIX), CHANNELS))

    def source(self, raw, sigma, base_state, base_q, z_state, message, layer: int):
        slot = SUFFIX.index(layer)
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        z_standard = torch.clamp(
            (z_state - self.z_center[slot]) / self.z_scale[slot], -8.0, 8.0
        )
        features = torch.cat(
            (
                raw[..., :1], tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0)[..., None],
                z_standard,
                torch.clamp(message / rms, -8.0, 8.0),
            ),
            dim=-1,
        )
        hidden = self.trunk(features)
        standard = hidden @ self.source_weight[slot] + self.source_bias[slot]
        return (sigma * rms)[..., None] * standard


def load_nonlinear_state() -> dict[str, torch.Tensor]:
    artifact = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    if (
        artifact.get("schema") != "q24-nonlinear-transported-raw20-head-checkpoint-v1"
        or int(artifact.get("selected_epoch", -1)) != 13
        or artifact.get("base_oof_result_sha256")
        != "87d22be8d48d7c73821c02bb3562b3e8ad67955022df0e2185c2f94838e87a25"
    ):
        raise AssertionError("nonlinear-Z model contract changed")
    return artifact["state_dict"]


def rollout(model: C8State, prepared, dataset, rank: int) -> list[torch.Tensor]:
    weights_np = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
    state = torch.zeros((1024, CHANNELS), dtype=torch.float32)
    states: list[torch.Tensor] = []
    for slot, layer in enumerate(SUFFIX):
        raw = torch.from_numpy(prepared.raw_suffix[rank, slot])
        sigma = torch.from_numpy(prepared.sigma_suffix[rank, slot])
        if slot == 0:
            message = torch.zeros_like(state)
        else:
            weights = torch.from_numpy(np.asarray(weights_np[layer], dtype=np.float32))
            message = raw[:, 5, None] * (weights.T @ state)
        source = model.source(
            raw,
            sigma,
            torch.from_numpy(prepared.base_state[rank, slot]),
            torch.from_numpy(prepared.base_q[rank, slot]),
            torch.from_numpy(prepared.z_suffix[rank, slot]),
            message,
            layer,
        )
        state = message + source
        states.append(state)
    return states


def training_rms(truth_suffix: np.ndarray, train: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.mean(truth_suffix[train].astype(np.float64) ** 2, axis=(0, 2)))
    return np.maximum(values, 2.0**-20).astype(np.float32)


def train_fixed(prepared, dataset, alpha_model, nonlinear_state, truth_suffix, train, stream_id):
    torch.manual_seed(MODEL_SEED)
    model = C8State(alpha_model, nonlinear_state)
    if sum(value.numel() for value in model.parameters()) != PARAMETERS:
        raise AssertionError("c8 parameter count changed")
    rms = training_rms(truth_suffix, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train), eta_min=MIN_LEARNING_RATE
    )
    rng = np.random.default_rng((MODEL_SEED ^ (int(stream_id) * 0x9E3779B1)) & 0xFFFF_FFFF)
    trace = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        objective = 0.0
        endpoint_objective = 0.0
        for rank_value in rng.permutation(train):
            rank = int(rank_value)
            states = rollout(model, prepared, dataset, rank)
            truth = torch.from_numpy(truth_suffix[rank])
            scale = torch.from_numpy(rms)[:, None]
            endpoint = torch.mean(((states[-1][:, 0] - truth[-1]) / scale[-1]).square())
            local = torch.stack(
                [torch.mean(((states[slot][:, 0] - truth[slot]) / scale[slot]).square())
                 for slot in range(len(SUFFIX))]
            ).mean()
            loss = endpoint + LOCAL_ANCHOR_WEIGHT * local
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"stream {stream_id} epoch {epoch}: nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective += float(loss.detach())
            endpoint_objective += float(endpoint.detach())
        row = {
            "stream_id": int(stream_id),
            "epoch": epoch,
            "objective": objective / len(train),
            "endpoint_objective": endpoint_objective / len(train),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_process_rss_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss),
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.eval()
    return model, rms, trace


def score(model, prepared, dataset, truth_suffix, held) -> list[dict[str, object]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            final = rollout(model, prepared, dataset, rank)[-1][:, 0].cpu().numpy().astype(np.float64)
            residual = truth_suffix[rank, -1].astype(np.float64) - final
            rows.append({
                "role_rank": rank,
                "mlp_id": int(dataset.mlp_ids[rank]),
                "candidate_final_mse": float(np.mean(residual * residual)),
            })
    return rows


def main() -> None:
    if RUN_STATE.exists() or RESULT.exists() or MODEL.exists():
        raise FileExistsError("c8 one-shot has already started")
    verify_static()
    nonlinear_result, nonlinear_oof = load_nonlinear_oof()
    base = load_module("c8_nonlinear_authority", BASE_PROGRAM)
    base_runner = load_module("c8_full688_authority", base.BASE_PROGRAM)
    records, native_storage = base_runner.verify_native_rows()
    folds = base_runner.balanced_folds(records)
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("target-blind folds changed")
    write_json_once(RUN_STATE, {
        "schema": "q24-c8-run-state-v1", "status": "RUNNING_ONE_SHOT",
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
    j = load_module("c8_exact_j", J_SOURCE)
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    alpha_model = base_runner.load_alpha_j(j)
    prepared = base.prepare_all_with_z(base_runner, j, alpha_model, dataset, error)
    nonlinear_state = load_nonlinear_state()
    for name in ("z_center", "z_scale"):
        left = getattr(prepared, name)
        right = nonlinear_state[name].cpu().numpy()
        if not np.array_equal(left.view(np.uint32), right.view(np.uint32)):
            raise AssertionError(f"prepared/frozen nonlinear normalization mismatch: {name}")
    truth_suffix = np.ascontiguousarray(
        error[:, SUFFIX] - prepared.base_state, dtype=np.float32
    )
    native_values = recompute_native_bank(j, load_native_model(j), dataset, error)
    del targets, hosts, error
    gc.collect()

    all_rows: list[dict[str, object] | None] = [None] * 688
    fold_rows = []
    traces = []
    rms_records = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train = ALL_RANKS[folds != fold]
        held = ALL_RANKS[folds == fold]
        candidate, rms, trace = train_fixed(
            prepared, dataset, alpha_model, nonlinear_state, truth_suffix, train, fold
        )
        rows = score(candidate, prepared, dataset, truth_suffix, held)
        for row in rows:
            all_rows[int(row["role_rank"])] = row
        values = np.asarray([float(row["candidate_final_mse"]) for row in rows])
        summary = {
            "fold": fold, "training_rows": len(train), "held_rows": held.tolist(),
            "candidate_final_mse": float(np.mean(values)),
            "nonlinear_z_oof_final_mse": float(np.mean(nonlinear_oof[held])),
            "native_bank_same688_final_mse": float(np.mean(native_values[held])),
            "win_fraction_vs_nonlinear_z": float(np.mean(values < nonlinear_oof[held])),
            "win_fraction_vs_native_bank": float(np.mean(values < native_values[held])),
        }
        fold_rows.append(summary)
        traces.extend(trace)
        rms_records.append(rms.tolist())
        print(json.dumps({"stage": "fold_score", **summary}, sort_keys=True), flush=True)
        del candidate
        gc.collect()

    if any(row is None for row in all_rows):
        raise AssertionError("c8 OOF coverage incomplete")
    rows = [row for row in all_rows if row is not None]
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
    projected_board_raw = candidate_mse
    projected_adjusted = candidate_mse * C8_D3_STRUCTURAL_BOARD_MULTIPLIER

    scientific_gates = {
        "finite_688_oof": bool(np.isfinite(candidate_values).all()),
        "relative_gain_vs_nonlinear_z_at_least_0_25pct": relative_gain >= MIN_RELATIVE_GAIN_VS_NONLINEAR_Z,
        "at_least_three_folds_improve_nonlinear_z": improving_folds >= MIN_IMPROVING_FOLDS,
        "paired_network_z_at_least_2": paired_z >= MIN_PAIRED_Z,
        "network_wins_vs_nonlinear_z_at_least_55pct": float(np.mean(gain > 0)) >= MIN_NETWORK_WIN_FRACTION,
        "every_fold_wins_vs_nonlinear_z_at_least_50pct": min(fold_wins) >= MIN_FOLD_WIN_FRACTION,
        "protected40_does_not_worsen_nonlinear_z": float(np.mean(candidate_values[protected])) <= float(np.mean(nonlinear_oof[protected])),
    }
    scientific_pass = all(scientific_gates.values())
    absolute_gates = {
        "candidate_strictly_below_recomputed_same688_native_bank": candidate_mse < native_mse,
        "recomputed_native_matches_measured_authority": np.isclose(native_mse, MEASURED_NATIVE_AUTHORITY_MSE, rtol=0.0, atol=2e-15),
        "candidate_strictly_below_legacy_scientific_admission": candidate_mse < LEGACY_SCIENTIFIC_ADMISSION_MSE,
        "network_wins_vs_native_bank_at_least_55pct": float(np.mean(candidate_values < native_values)) >= MIN_NETWORK_WIN_FRACTION,
    }
    deployment_gates = {
        **absolute_gates,
        "candidate_raw_times_structural_c8_d3_multiplier_beats_compute_only_d3": projected_adjusted < D4_BOARD_RAW_MSE * D3_BOARD_MULTIPLIER_328620,
        "structural_projected_adjusted_beats_d4_board_328622": projected_adjusted < D4_BOARD_ADJUSTED_SCORE,
        "literal_integrated_flops_required_before_submission": False,
    }
    submission_worthy = scientific_pass and all(
        value for key, value in deployment_gates.items()
        if key != "literal_integrated_flops_required_before_submission"
    )

    model_hash = None
    final_trace = []
    if scientific_pass:
        final_model, final_rms, final_trace = train_fixed(
            prepared, dataset, alpha_model, nonlinear_state, truth_suffix, ALL_RANKS, FOLDS
        )
        artifact = {
            "schema": "q24-c8-autonomous-correction-state-v1",
            "state_dict": final_model.state_dict(),
            "epochs": EPOCHS,
            "channels": CHANNELS,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "nonlinear_z_oof_result_sha256": BASE_RESULT_SHA256,
            "training_rms": final_rms,
        }
        temporary = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary)
        temporary.replace(MODEL)
        model_hash = sha256(MODEL)

    result = {
        "schema": "q24-c8-autonomous-correction-state-oof-v1",
        "status": (
            "PROMOTED_SUBMISSION_WORTHY_PENDING_LITERAL_METER" if submission_worthy
            else "PROMOTED_SCIENTIFIC_ONLY" if scientific_pass
            else "CLOSED_NO_BROAD_OOF_GAIN"
        ),
        "fixed_rule": {
            "architecture": "50->64->64->six layer-specific 8-channel sources",
            "parameters": PARAMETERS, "channels": CHANNELS, "epochs": EPOCHS,
            "recurrence": "M_l=Phi_l*(W_l.T@C_(l-1)); C_l=M_l+sigma_l*rms_l*f(raw20,A,qJ,Z,M_l/rms_l)",
            "reported_correction": "C_l channel0; final score uses C_15 channel0",
            "loss": f"normalized endpoint MSE + {LOCAL_ANCHOR_WEIGHT} * normalized six-layer channel0 anchor",
            "base_nonlinear_z_arm_retrained": False,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        },
        "crossfit": {
            "candidate_oof_final_mse": candidate_mse,
            "nonlinear_z_oof_final_mse": nonlinear_mse,
            "recomputed_native_bank_same688_final_mse": native_mse,
            "measured_native_authority_final_mse": MEASURED_NATIVE_AUTHORITY_MSE,
            "legacy_scientific_admission_threshold_not_measurement": LEGACY_SCIENTIFIC_ADMISSION_MSE,
            "relative_gain_vs_nonlinear_z": relative_gain,
            "paired_network_z_vs_nonlinear_z": paired_z,
            "network_win_fraction_vs_nonlinear_z": float(np.mean(gain > 0)),
            "network_win_fraction_vs_native_bank": float(np.mean(candidate_values < native_values)),
            "fold_candidate_mse": fold_candidate,
            "fold_nonlinear_z_mse": fold_nonlinear,
            "fold_network_win_fractions_vs_nonlinear_z": fold_wins,
            "fold_results": fold_rows, "per_network": rows,
        },
        "gates": {
            "scientific": scientific_gates, "scientific_pass": scientific_pass,
            "absolute": absolute_gates, "deployment": deployment_gates,
            "submission_worthy_pending_literal_meter": submission_worthy,
        },
        "projection": {
            "law": "c8 OOF raw * (exact #328620 D3 board multiplier + frozen c8 incremental structural bill / 2^41)",
            "d4_board_raw_mse": D4_BOARD_RAW_MSE,
            "d4_board_adjusted_score": D4_BOARD_ADJUSTED_SCORE,
            "d3_board_multiplier_328620": D3_BOARD_MULTIPLIER_328620,
            "c8_total_incremental_vs_compute_only_d3_upper": C8_TOTAL_INCREMENTAL_VS_COMPUTE_ONLY_D3_UPPER,
            "c8_d3_structural_board_multiplier": C8_D3_STRUCTURAL_BOARD_MULTIPLIER,
            "maximum_raw_to_beat_d4_under_structural_bill": D4_BOARD_ADJUSTED_SCORE / C8_D3_STRUCTURAL_BOARD_MULTIPLIER,
            "c8_d3_structural_max_compute_ratio": C8_D3_MAX_COMPUTE_RATIO,
            "projected_board_raw_mse": projected_board_raw,
            "projected_adjusted_score": projected_adjusted,
            "literal_meter_required_before_submission": True,
        },
        "training": {"fold_rms": rms_records, "crossfit_trace": traces, "final_trace": final_trace},
        "deployability": {
            "incremental_flops_vs_22input_head_upper": INCREMENTAL_C8_VS_22_UPPER,
            "incremental_flops_vs_nonlinear_z_upper": 100_159_488,
            "live_c8_state_bytes_float32": 1024 * CHANNELS * 4,
            "parameters_bytes_float32": PARAMETERS * 4,
            "native_row_storage_bytes": native_storage,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREGISTRATION),
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
        "schema": "q24-c8-run-state-v1", "status": "COMPLETE",
        "result_sha256": sha256(RESULT), "model_sha256": model_hash,
    })
    print(json.dumps({"status": result["status"], "crossfit": result["crossfit"], "projection": result["projection"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
