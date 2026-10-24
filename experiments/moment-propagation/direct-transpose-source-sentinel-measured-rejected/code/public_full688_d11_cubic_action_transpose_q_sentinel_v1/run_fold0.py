#!/Users/matt/anaconda3/bin/python3
"""One-shot balanced-fold-0 D11 cubic-action transpose q sentinel.

Preflight is target-free.  Execution verifies all target-blind rows, acquires an
exclusive durable no-retry claim, and only then decodes the observer-fit target.
The derived c/b fields are formed in memory and are never cached.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

import feature_generator


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREGISTRATION = HERE / "PREREGISTRATION.json"
DESIGN_FREEZE = HERE / "FREEZE_RECEIPT.json"
FEATURE_SOURCE = HERE / "feature_generator.py"
STATIC_VERIFICATION = HERE / "STATIC_VERIFICATION.json"
RELEASE_DEFAULT = HERE / "FOLD0_EXECUTION_RELEASE.json"
CLAIM = HERE / "FOLD0_RUN_CLAIM.json"
RESULT = HERE / "FOLD0_RESULT.json"
MODELS = HERE / "FOLD0_MODELS.pt"
FAILURE = HERE / "FOLD0_TERMINAL_FAILURE.json"

J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
NATIVE22_MODELS = (
    P2
    / "public_q24_signed_pair_qinnovation_distillation_v1"
    / "NATIVE22_CAPACITY_OUTPUT_V1"
    / "MODELS.pt"
)
NATIVE22_RESULT = NATIVE22_MODELS.with_name("RESULT.json")
TERMINAL_82_RESULT = P2 / "public_q24_abs_sq_message_qhead_oof_v1/RESULT.json"
TERMINAL_82_INTEGRITY = (
    P2 / "public_q24_abs_sq_message_qhead_oof_v1/INTEGRITY_SUMMARY.json"
)
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FEATURE_MANIFEST = (
    P2
    / "learned_residual_observer_design"
    / "v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
    / "manifest.json"
)

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
QREF_SOURCE_SHA256 = "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2"
NATIVE22_MODELS_SHA256 = "cff45c489c743f5de31454875e7dedfeb7d5a0fb9102663bd756b91dea482ee4"
NATIVE22_RESULT_SHA256 = "5c940c501c383fe65e9e4b5dda103ba287797d0cb18ed22bc6314588f37245fd"
TERMINAL_82_RESULT_SHA256 = "87624c36e90e3db7e5592ff8eb56acf5696a37d9ad0f2dd7ed656394301f98b4"
TERMINAL_82_INTEGRITY_SHA256 = "73a73b492469c861f68fbc7170c30c75225a72651ec0d45c913ee21ce73f1a5e"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ROWS = 688
DEPTH = 16
WIDTH = 1024
LAYERS = tuple(range(5, 16))
FOLD = 0
FIT = 516
HELD = 172
INPUTS = 25
HIDDEN = 64
MODEL_SEED = 1_362_178_069
EPOCHS = 13
SCHEDULER_HORIZON_EPOCHS = 14
SAMPLES_PER_LAYER = 256
NETWORK_BATCH = 4
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
ORACLE_GATE = 1.0e-8
ENDPOINT_GATE = 2.79555972111794e-8
FROZEN_INDEPENDENT256_ENDPOINT = 3.0220317427127064e-8
MIN_WIN_FRACTION = 0.60
MIN_Z = 3.0
EXPECTED_FOLD0_BASE = 3.0923227278552953e-8
MAX_RSS_BYTES = 8 * (1 << 30)

ACTIVE_NONCE: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_exclusive_durable(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_torch_exclusive_durable(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def verify_file(path: Path, digest: str) -> None:
    if not path.is_file() or sha256(path) != digest:
        raise AssertionError(f"frozen input changed: {path}")


def verify_fixed_inputs(*, include_target_bytes: bool) -> None:
    expected = {
        J_SOURCE: J_SOURCE_SHA256,
        QREF_SOURCE: QREF_SOURCE_SHA256,
        NATIVE22_MODELS: NATIVE22_MODELS_SHA256,
        NATIVE22_RESULT: NATIVE22_RESULT_SHA256,
        TERMINAL_82_RESULT: TERMINAL_82_RESULT_SHA256,
        TERMINAL_82_INTEGRITY: TERMINAL_82_INTEGRITY_SHA256,
        FEATURE_MANIFEST: FEATURE_MANIFEST_SHA256,
    }
    if include_target_bytes:
        expected[TARGETS] = TARGETS_SHA256
    for path, digest in expected.items():
        verify_file(path, digest)
    terminal = read_json(TERMINAL_82_RESULT)
    integrity = read_json(TERMINAL_82_INTEGRITY)
    if (
        terminal.get("status") != "FAIL_MATERIAL_OOF__STOP_NO_REFIT"
        or terminal.get("decision") != "STOP_NO_REFIT"
        or integrity.get("status") != "PASS_FROZEN_BYTES_COMPLETE_OOF_TERMINAL_NO_REFIT"
    ):
        raise AssertionError("82-arm interlock is not terminal FAIL")


def validate_release(path: Path) -> dict[str, Any]:
    release = read_json(path)
    if (
        release.get("schema")
        != "public-full688-d11-cubic-action-transpose-fold0-execution-release-v1"
        or release.get("status") != "RELEASE_EXACTLY_ONE_FOLD0_SENTINEL__NO_RETRY"
        or release.get("sentinel_fold") != 0
        or release.get("one_shot") is not True
        or release.get("target_decode") is not True
        or release.get("training") is not True
        or release.get("sentinel_execution") is not True
        or release.get("retry") is not False
        or release.get("role") != "observer_fit"
        or release.get("target_columns") != ["mlp_id", "all_layer_means"]
        or release.get("observer_selection") is not False
        or release.get("names") is not False
        or release.get("scores") is not False
        or release.get("network") is not False
        or release.get("package") is not False
        or release.get("remaining_folds") is not False
        or release.get("all_data_fit") is not False
        or release.get("cache") is not False
        or release.get("integration") is not False
        or release.get("meter") is not False
        or release.get("board") is not False
        or release.get("submission") is not False
    ):
        raise PermissionError("release is not the narrow fold-0 authority")
    expected = {
        "runner_sha256": sha256(Path(__file__)),
        "feature_generator_sha256": sha256(FEATURE_SOURCE),
        "static_verification_sha256": sha256(STATIC_VERIFICATION),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "design_freeze_sha256": sha256(DESIGN_FREEZE),
        "native22_models_sha256": NATIVE22_MODELS_SHA256,
        "native22_result_sha256": NATIVE22_RESULT_SHA256,
        "terminal_82_result_sha256": TERMINAL_82_RESULT_SHA256,
        "terminal_82_integrity_sha256": TERMINAL_82_INTEGRITY_SHA256,
        "target_bundle_sha256": TARGETS_SHA256,
        "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
        "fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
    }
    bindings = release.get("bindings", {})
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise PermissionError(f"release binding changed: {key}")
    return release


def verify_runtime_and_launch_contract() -> None:
    if (
        sys.version_info[:3] != (3, 11, 5)
        or np.__version__ != "1.26.4"
        or torch.__version__ != "2.7.0"
    ):
        raise RuntimeError(
            f"runtime tuple changed: {sys.version_info[:3]}, {np.__version__}, {torch.__version__}"
        )
    expected_threads = {
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
    }
    for name, expected in expected_threads.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"launch thread contract changed: {name}")
    if os.nice(0) < 10:
        raise RuntimeError("launch must use nice -n 10 or lower priority")


def claim_owned() -> bool:
    if ACTIVE_NONCE is None or not CLAIM.is_file():
        return False
    try:
        value = read_json(CLAIM)
    except Exception:
        return False
    return (
        value.get("owner_pid") == os.getpid()
        and value.get("owner_nonce") == ACTIVE_NONCE
    )


def record_failure_if_owned(error: BaseException) -> None:
    if not claim_owned() or RESULT.exists() or FAILURE.exists():
        return
    write_exclusive_durable(
        FAILURE,
        {
            "schema": "public-full688-d11-cubic-action-transpose-fold0-terminal-failure-v1",
            "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
            "failed_utc": utc_now(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "claim_sha256": sha256(CLAIM),
            "result_present": RESULT.exists(),
            "model_present": MODELS.exists(),
            "decision": "STOP_NO_RETRY_NO_FOLDS_1_TO_3",
        },
    )


@dataclass
class Prepared:
    base_state: np.ndarray
    base_q: np.ndarray
    delta_q: np.ndarray
    residual_final: np.ndarray
    raw_selected: np.ndarray
    sigma_selected: np.ndarray
    appended: np.ndarray
    base_mse: np.ndarray


class DirectQHead(nn.Module):
    def __init__(
        self,
        centers: np.ndarray | torch.Tensor,
        scales: np.ndarray | torch.Tensor,
        layer_rms: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers).detach().clone())
        self.register_buffer("scales", torch.as_tensor(scales).detach().clone())
        self.register_buffer("layer_rms", torch.as_tensor(layer_rms).detach().clone())
        self.trunk = nn.Sequential(
            nn.Linear(INPUTS, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.zeros(len(LAYERS), HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(len(LAYERS)))

    def source(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        base_state: torch.Tensor,
        base_q: torch.Tensor,
        appended: torch.Tensor,
        layer: int,
        *,
        source_on: bool,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        if not source_on:
            appended = torch.zeros_like(appended)
        features = torch.cat(
            (
                raw[..., :1],
                tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(
                    base_q / torch.clamp(sigma * rms, min=2.0**-30),
                    -8.0,
                    8.0,
                )[..., None],
                appended,
            ),
            dim=-1,
        )
        slot = LAYERS.index(layer)
        hidden = self.trunk(features)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[slot])
        return sigma * rms * (standard + self.head_bias[slot])


def canonical_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        value = tensor.detach().cpu().numpy()
        if value.dtype != np.float32:
            raise AssertionError("model state is not float32")
        digest.update(np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def make_model(base_payload: dict[str, Any]) -> tuple[DirectQHead, str]:
    torch.manual_seed(MODEL_SEED)
    model = DirectQHead(
        base_payload["j_centers"],
        base_payload["j_scales"],
        base_payload["j_layer_rms"],
    )
    with torch.no_grad():
        model.head_weight.zero_()
        model.head_bias.zero_()
    count = sum(value.numel() for value in model.parameters())
    if count != 6_539:
        raise AssertionError(f"parameter count changed: {count}")
    return model, canonical_model_hash(model)


def sample_indices(rank: int, layer: int, epoch: int) -> np.ndarray:
    value = (
        MODEL_SEED
        ^ (int(rank) * 0x9E3779B1)
        ^ (int(layer) * 0x85EBCA77)
        ^ (int(epoch) * 0xC2B2AE3D)
    )
    return np.random.default_rng(value & 0xFFFF_FFFF).choice(
        WIDTH, size=SAMPLES_PER_LAYER, replace=False
    )


def q_rms_for(prepared: Prepared, fit: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sqrt(
            np.mean(
                prepared.delta_q[fit].astype(np.float64) ** 2,
                axis=(0, 2),
            )
        ),
        2.0**-20,
    ).astype(np.float32)


def train_arm(
    name: str,
    source_on: bool,
    base_payload: dict[str, Any],
    prepared: Prepared,
    fit: np.ndarray,
) -> tuple[DirectQHead, dict[str, Any]]:
    model, initial_hash = make_model(base_payload)
    q_rms = q_rms_for(prepared, fit)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
    )
    steps_per_epoch = math.ceil(len(fit) / NETWORK_BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_HORIZON_EPOCHS * steps_per_epoch,
        eta_min=MIN_LEARNING_RATE,
    )
    order_seed = (MODEL_SEED ^ 0xA5A5A5A5 ^ (FOLD * 0x9E3779B1)) & 0xFFFF_FFFF
    order_rng = np.random.Generator(np.random.PCG64(order_seed))
    order_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = order_rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        objective_sum = 0.0
        examples = 0
        for offset in range(0, len(order), NETWORK_BATCH):
            ranks = order[offset : offset + NETWORK_BATCH]
            if len(ranks) != NETWORK_BATCH:
                raise AssertionError("fold size no longer divides into batches of four")
            losses = []
            for slot, layer in enumerate(LAYERS):
                indices = [sample_indices(int(rank), layer, epoch) for rank in ranks]
                for index in indices:
                    sample_digest.update(
                        np.ascontiguousarray(index.astype("<i8", copy=False)).tobytes()
                    )
                raw = np.concatenate(
                    [
                        prepared.raw_selected[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                sigma = np.concatenate(
                    [
                        prepared.sigma_selected[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                state = np.concatenate(
                    [
                        prepared.base_state[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                base_q = np.concatenate(
                    [
                        prepared.base_q[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                appended = np.concatenate(
                    [
                        prepared.appended[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                truth_q = np.concatenate(
                    [
                        prepared.delta_q[int(rank), slot, index]
                        for rank, index in zip(ranks, indices, strict=True)
                    ]
                )
                source = model.source(
                    torch.from_numpy(np.ascontiguousarray(raw)),
                    torch.from_numpy(np.ascontiguousarray(sigma)),
                    torch.from_numpy(np.ascontiguousarray(state)),
                    torch.from_numpy(np.ascontiguousarray(base_q)),
                    torch.from_numpy(np.ascontiguousarray(appended)),
                    layer,
                    source_on=source_on,
                )
                truth = torch.from_numpy(np.ascontiguousarray(truth_q))
                losses.append(
                    torch.mean(((source - truth) / float(q_rms[slot])).square())
                )
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {name} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        record = {
            "arm": name,
            "epoch": epoch,
            "sampled_teacher_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_process_rss_bytes": enforce_rss(f"{name} epoch {epoch}"),
        }
        trace.append(record)
        print(json.dumps({"stage": "training", **record}, sort_keys=True), flush=True)
    model.eval()
    return model, {
        "initial_state_sha256": initial_hash,
        "final_state_sha256": canonical_model_hash(model),
        "row_order_sha256": order_digest.hexdigest(),
        "node_sample_sha256": sample_digest.hexdigest(),
        "q_rms": q_rms.tolist(),
        "trace": trace,
    }


def prepare_fold0(
    j: Any,
    qref: Any,
    records: list[dict[str, Any]],
    folds: np.ndarray,
    targets: np.ndarray,
    base_payload: dict[str, Any],
    held_predictions: dict[int, np.ndarray],
) -> tuple[Prepared, Any]:
    dataset = qref.All688Dataset(records)
    centers = np.asarray(base_payload["j_centers"], dtype=np.float32)
    scales = np.asarray(base_payload["j_scales"], dtype=np.float32)
    layer_rms = np.asarray(base_payload["j_layer_rms"], dtype=np.float32)
    base = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    base.load_state_dict(base_payload["j_state_dict"], strict=True)
    base.eval()

    shape = (ROWS, len(LAYERS), WIDTH)
    base_state = np.empty(shape, dtype=np.float32)
    base_q = np.empty(shape, dtype=np.float32)
    delta_q = np.empty(shape, dtype=np.float32)
    residual_final = np.empty((ROWS, WIDTH), dtype=np.float32)
    raw_selected = np.empty((*shape, 20), dtype=np.float32)
    sigma_selected = np.empty(shape, dtype=np.float32)
    appended = np.empty((*shape, 3), dtype=np.float32)
    base_mse = np.full(ROWS, np.nan, dtype=np.float64)

    with torch.no_grad():
        for rank in range(ROWS):
            row = dataset.load(rank)
            raw = row.raw_locals
            sigma = row.sigma
            weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
            if (
                raw.dtype != np.float32
                or raw.shape != (DEPTH, WIDTH, 20)
                or sigma.dtype != np.float32
                or sigma.shape != (DEPTH, WIDTH)
                or weights.dtype != np.float32
                or weights.shape != (DEPTH, WIDTH, WIDTH)
            ):
                raise AssertionError(f"native tensor schema changed at rank {rank}")
            if rank in held_predictions:
                a = held_predictions[rank]
                raw_t = sigma_t = weights_t = prediction = None
            else:
                raw_t = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None]
                sigma_t = torch.from_numpy(np.array(sigma, dtype=np.float32, copy=True))[None]
                weights_t = torch.from_numpy(np.array(weights, dtype=np.float32, copy=True))[None]
                prediction = base(raw_t, sigma_t, weights_t, exposure=1.0)
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("fold J returned non-tensor")
                a = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            host = np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False)
            if host.dtype != np.float32 or host.shape != (DEPTH, WIDTH):
                raise AssertionError(f"host schema changed at rank {rank}")
            error = targets[rank].astype(np.float32, copy=False) - host
            residual = error - a
            residual_final[rank] = residual[-1]
            raw_selected[rank] = raw[list(LAYERS)]
            sigma_selected[rank] = sigma[list(LAYERS)]
            built = feature_generator.build_features(
                raw,
                sigma,
                weights,
                layer_rms,
                first_layer=LAYERS[0],
                last_layer=LAYERS[-1],
            )
            if built.layers != LAYERS:
                raise AssertionError("feature layer ordering changed")
            appended[rank] = built.appended_features
            for slot, layer in enumerate(LAYERS):
                phi = raw[layer, :, feature_generator.RAW_PHI].astype(np.float64)
                prior = np.stack((a[layer - 1], residual[layer - 1])).astype(np.float64)
                message = phi[None, :] * (
                    prior @ weights[layer].astype(np.float64)
                )
                base_state[rank, slot] = a[layer]
                base_q[rank, slot] = (
                    a[layer].astype(np.float64) - message[0]
                ).astype(np.float32)
                delta_q[rank, slot] = (
                    residual[layer].astype(np.float64) - message[1]
                ).astype(np.float32)
            if int(folds[rank]) == FOLD:
                final = residual_final[rank].astype(np.float64)
                base_mse[rank] = float(np.mean(final * final))
            del raw_t, sigma_t, weights_t, prediction, weights, built
            if (rank + 1) % 16 == 0 or rank + 1 == ROWS:
                print(
                    json.dumps(
                        {
                            "stage": "prepare_in_memory",
                            "rows": rank + 1,
                            "maximum_process_rss_bytes": enforce_rss(
                                f"prepare rank {rank}"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                gc.collect()
    prepared = Prepared(
        base_state,
        base_q,
        delta_q,
        residual_final,
        raw_selected,
        sigma_selected,
        appended,
        base_mse,
    )
    for name in (
        "base_state",
        "base_q",
        "delta_q",
        "residual_final",
        "raw_selected",
        "sigma_selected",
        "appended",
    ):
        if not bool(np.all(np.isfinite(getattr(prepared, name)))):
            raise FloatingPointError(f"nonfinite prepared array: {name}")
    return prepared, dataset


def load_target_subset(
    qref: Any,
    records: list[dict[str, Any]],
    ranks: np.ndarray,
) -> np.ndarray:
    """Materialize and return only the requested observer-fit target rows.

    The parquet file has one row group, so this makes no claim that its physical
    pages for unreturned rows were not decoded internally by Arrow.
    """

    ranks = np.asarray(ranks, dtype=np.int64)
    expected_ids = np.asarray(
        [int(records[int(rank)]["mlp_id"]) for rank in ranks], dtype=np.int32
    )
    table = qref.pq.read_table(
        TARGETS,
        columns=["mlp_id", "all_layer_means"],
        filters=[("mlp_id", "in", expected_ids.tolist())],
    )
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not qref.pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    decoded = storage.to_numpy(zero_copy_only=False).reshape(len(ids), DEPTH, WIDTH)
    if ids.dtype != np.int32 or decoded.dtype != np.float32 or len(ids) != len(ranks):
        raise AssertionError("held target subset schema or size changed")
    positions = {int(value): position for position, value in enumerate(ids)}
    if len(positions) != len(ranks) or set(positions) != set(map(int, expected_ids)):
        raise AssertionError("held target subset identity changed")
    order = np.asarray([positions[int(value)] for value in expected_ids], dtype=np.int64)
    return np.ascontiguousarray(decoded[order], dtype=np.float32)


def held_oracle_gate(
    j: Any,
    qref: Any,
    records: list[dict[str, Any]],
    held: np.ndarray,
    base_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Evaluate the exact start-5 oracle before materializing/using fit targets.

    Algebraically, e_15-d_15=T_15...T_5 e_4.  The frozen numerical spelling
    starts from float32 e_4, applies each tangent in float64, and rounds the
    state back to float32 after every layer.
    """

    targets = load_target_subset(qref, records, held)
    dataset = qref.All688Dataset(records)
    centers = np.asarray(base_payload["j_centers"], dtype=np.float32)
    scales = np.asarray(base_payload["j_scales"], dtype=np.float32)
    layer_rms = np.asarray(base_payload["j_layer_rms"], dtype=np.float32)
    base = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
    base.load_state_dict(base_payload["j_state_dict"], strict=True)
    base.eval()
    base_mse = np.empty(len(held), dtype=np.float64)
    oracle_mse = np.empty(len(held), dtype=np.float64)
    predictions: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for position, rank_value in enumerate(held):
            rank = int(rank_value)
            row = dataset.load(rank)
            raw = row.raw_locals
            sigma = row.sigma
            weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
            prediction = base(
                torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None],
                torch.from_numpy(np.array(sigma, dtype=np.float32, copy=True))[None],
                torch.from_numpy(np.array(weights, dtype=np.float32, copy=True))[None],
                exposure=1.0,
            )
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("fold J returned non-tensor")
            a = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            predictions[rank] = np.array(a, dtype=np.float32, copy=True)
            host = np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False)
            error = targets[position] - host
            residual = error - a
            state = np.array(residual[4], dtype=np.float32, copy=True)
            for layer in LAYERS:
                phi = raw[layer, :, feature_generator.RAW_PHI].astype(np.float64)
                state = (
                    phi
                    * (
                        state.astype(np.float64)
                        @ weights[layer].astype(np.float64)
                    )
                ).astype(np.float32)
            final = residual[-1].astype(np.float64)
            base_mse[position] = float(np.mean(final * final))
            oracle_mse[position] = float(np.mean(state.astype(np.float64) ** 2))
            if (position + 1) % 16 == 0 or position + 1 == len(held):
                print(
                    json.dumps(
                        {
                            "stage": "held_only_pretraining_oracle",
                            "rows": position + 1,
                            "fit_target_rows_materialized_returned_or_used": 0,
                            "maximum_process_rss_bytes": enforce_rss(
                                f"held oracle position {position}"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return base_mse, oracle_mse, predictions


def paired_z(improvement: np.ndarray) -> float:
    standard_error = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    if standard_error == 0.0:
        return float("inf") if float(np.mean(improvement)) > 0.0 else 0.0
    return float(np.mean(improvement) / standard_error)


def score_pair(
    source_on_model: DirectQHead,
    source_zero_model: DirectQHead,
    prepared: Prepared,
    dataset: Any,
    held: np.ndarray,
) -> dict[str, Any]:
    local_sse = {"source_on": 0.0, "source_zero": 0.0}
    local_count = len(held) * len(LAYERS) * WIDTH
    rows: list[dict[str, Any]] = []
    source_on_model.eval()
    source_zero_model.eval()
    with torch.no_grad():
        for position, rank_value in enumerate(held):
            rank = int(rank_value)
            weights = np.load(
                dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False
            )
            states = np.zeros((3, WIDTH), dtype=np.float32)
            for slot, layer in enumerate(LAYERS):
                raw = torch.from_numpy(
                    np.ascontiguousarray(prepared.raw_selected[rank, slot])
                )
                sigma = torch.from_numpy(
                    np.ascontiguousarray(prepared.sigma_selected[rank, slot])
                )
                base_state = torch.from_numpy(
                    np.ascontiguousarray(prepared.base_state[rank, slot])
                )
                base_q = torch.from_numpy(
                    np.ascontiguousarray(prepared.base_q[rank, slot])
                )
                appended = torch.from_numpy(
                    np.ascontiguousarray(prepared.appended[rank, slot])
                )
                source_on = source_on_model.source(
                    raw,
                    sigma,
                    base_state,
                    base_q,
                    appended,
                    layer,
                    source_on=True,
                ).cpu().numpy()
                source_zero = source_zero_model.source(
                    raw,
                    sigma,
                    base_state,
                    base_q,
                    appended,
                    layer,
                    source_on=False,
                ).cpu().numpy()
                truth_q = prepared.delta_q[rank, slot]
                local_sse["source_on"] += float(
                    np.sum((source_on.astype(np.float64) - truth_q) ** 2)
                )
                local_sse["source_zero"] += float(
                    np.sum((source_zero.astype(np.float64) - truth_q) ** 2)
                )
                sources = np.stack((source_on, source_zero, truth_q)).astype(np.float64)
                phi = prepared.raw_selected[
                    rank, slot, :, feature_generator.RAW_PHI
                ].astype(np.float64)
                states = (
                    phi[None, :]
                    * (states.astype(np.float64) @ weights[layer].astype(np.float64))
                    + sources
                ).astype(np.float32)
            residual = prepared.residual_final[rank].astype(np.float64)
            base = float(np.mean(residual * residual))
            on = float(np.mean((residual - states[0].astype(np.float64)) ** 2))
            zero = float(np.mean((residual - states[1].astype(np.float64)) ** 2))
            oracle = float(np.mean((residual - states[2].astype(np.float64)) ** 2))
            if abs(base - prepared.base_mse[rank]) > 1.0e-20:
                raise AssertionError("base scoring replay changed")
            rows.append(
                {
                    "role_rank": rank,
                    "base_foldJ_mse": base,
                    "source_on_mse": on,
                    "source_zero_mse": zero,
                    "exact_q_oracle_mse": oracle,
                }
            )
            if (position + 1) % 32 == 0 or position + 1 == len(held):
                print(
                    json.dumps(
                        {
                            "stage": "score_fold0",
                            "rows": position + 1,
                            "maximum_process_rss_bytes": enforce_rss(
                                f"score held position {position}"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return {
        "rows": rows,
        "source_on_local_q_mse": local_sse["source_on"] / local_count,
        "source_zero_local_q_mse": local_sse["source_zero"] / local_count,
        "source_on_endpoint_mse": float(np.mean([row["source_on_mse"] for row in rows])),
        "source_zero_endpoint_mse": float(
            np.mean([row["source_zero_mse"] for row in rows])
        ),
        "base_foldJ_endpoint_mse": float(
            np.mean([row["base_foldJ_mse"] for row in rows])
        ),
        "exact_q_oracle_endpoint_mse": float(
            np.mean([row["exact_q_oracle_mse"] for row in rows])
        ),
    }


def write_terminal_oracle_result(
    release_path: Path,
    held: np.ndarray,
    base_values: np.ndarray,
    oracle_values: np.ndarray,
    started_utc: str,
) -> None:
    oracle = float(np.mean(oracle_values))
    base = float(np.mean(base_values))
    write_exclusive_durable(
        RESULT,
        {
            "schema": "public-full688-d11-cubic-action-transpose-fold0-result-v1",
            "status": "FAIL_PRETRAINING_ORACLE_GATE__NO_TRAINING__NO_RETRY",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "bindings": {
                "claim_sha256": sha256(CLAIM),
                "release_sha256": sha256(release_path),
                "runner_sha256": sha256(Path(__file__)),
                "feature_generator_sha256": sha256(FEATURE_SOURCE),
            },
            "fold": FOLD,
            "held_rows": held.tolist(),
            "pretraining_oracle_gate": {
                "base_foldJ_endpoint_mse": base,
                "start5_exact_q_oracle_endpoint_mse": oracle,
                "strictly_below_1e_8": oracle < ORACLE_GATE,
            },
            "training_runs": 0,
            "decision": "CLOSE_THIS_SPELLING__DO_NOT_RUN_FOLDS_1_TO_3",
        },
    )


def execute(release_path: Path) -> None:
    global ACTIVE_NONCE
    started_utc = utc_now()
    print(
        json.dumps(
            {
                "stage": "process_start",
                "pid": os.getpid(),
                "started_utc": started_utc,
                "nice": os.nice(0),
                "torch_threads_requested": 4,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if any(path.exists() for path in (CLAIM, RESULT, MODELS, FAILURE)):
        raise FileExistsError("fold-0 sentinel already claimed, completed, or failed")
    verify_runtime_and_launch_contract()
    verify_fixed_inputs(include_target_bytes=False)
    validate_release(release_path)
    verify_fixed_inputs(include_target_bytes=True)

    j = load_module("cubic_transpose_bound_j", J_SOURCE)
    qref = load_module("cubic_transpose_bound_qref", QREF_SOURCE)
    records, native_storage = qref.verify_native_rows()
    folds = qref.balanced_folds(records)
    if qref.NATIVE_AGGREGATE_SHA256 != NATIVE_AGGREGATE_SHA256:
        raise AssertionError("native aggregate literal changed")
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("fold assignment changed")
    fit = np.flatnonzero(folds != FOLD)
    held = np.flatnonzero(folds == FOLD)
    if len(fit) != FIT or len(held) != HELD:
        raise AssertionError("fold size changed")

    nonce = hashlib.sha256(
        os.getpid().to_bytes(8, "big", signed=False) + os.urandom(32)
    ).hexdigest()
    write_exclusive_durable(
        CLAIM,
        {
            "schema": "public-full688-d11-cubic-action-transpose-fold0-run-claim-v1",
            "status": "CLAIMED_AFTER_TARGET_FREE_VERIFICATION_BEFORE_TARGET_DECODE",
            "started_utc": started_utc,
            "claimed_utc": utc_now(),
            "owner_pid": os.getpid(),
            "owner_nonce": nonce,
            "one_shot_no_retry": True,
            "sentinel_fold": FOLD,
            "remaining_folds_authorized": False,
            "derived_feature_cache_authorized": False,
            "target_values_decoded_before_claim": 0,
            "native_storage_bytes_verified": native_storage,
            "bindings": {
                "release_sha256": sha256(release_path),
                "runner_sha256": sha256(Path(__file__)),
                "feature_generator_sha256": sha256(FEATURE_SOURCE),
                "static_verification_sha256": sha256(STATIC_VERIFICATION),
                "preregistration_sha256": sha256(PREREGISTRATION),
                "native22_models_sha256": NATIVE22_MODELS_SHA256,
                "target_bundle_sha256": TARGETS_SHA256,
                "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
                "fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
            },
            "terminal_rule": "ANY_FAILURE_IS_FINAL__NO_RETRY__NO_FOLDS_1_TO_3",
        },
    )
    ACTIVE_NONCE = nonce
    print(
        json.dumps(
            {
                "stage": "exclusive_claim_durable_before_target_decode",
                "claim_sha256": sha256(CLAIM),
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    artifact = torch.load(NATIVE22_MODELS, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "public-q24-native22-high-capacity-oof-models-v1":
        raise AssertionError("native22 model artifact schema changed")
    base_payload = artifact["folds"][str(FOLD)]

    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    gate_base_values, gate_oracle_values, held_predictions = held_oracle_gate(
        j, qref, records, held, base_payload
    )
    base_endpoint = float(np.mean(gate_base_values))
    oracle_endpoint = float(np.mean(gate_oracle_values))
    if abs(base_endpoint - EXPECTED_FOLD0_BASE) > 1.0e-18:
        raise AssertionError(
            f"fold-J base lineage changed: {base_endpoint} vs {EXPECTED_FOLD0_BASE}"
        )
    print(
        json.dumps(
            {
                "stage": "pretraining_oracle_gate",
                "base_foldJ_endpoint_mse": base_endpoint,
                "start5_exact_q_oracle_endpoint_mse": oracle_endpoint,
                "strictly_below_1e_8": oracle_endpoint < ORACLE_GATE,
                "fit_target_rows_materialized_returned_or_used": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not oracle_endpoint < ORACLE_GATE:
        write_terminal_oracle_result(
            release_path,
            held,
            gate_base_values,
            gate_oracle_values,
            started_utc,
        )
        return

    targets = qref.load_targets(records)
    prepared, dataset = prepare_fold0(
        j, qref, records, folds, targets, base_payload, held_predictions
    )
    del targets, artifact
    gc.collect()
    enforce_rss("post preparation")
    prepared_base_endpoint = float(np.mean(prepared.base_mse[held]))
    if abs(prepared_base_endpoint - base_endpoint) > 1.0e-18:
        raise AssertionError("held base reproduction changed during full preparation")

    source_on_model, source_on_training = train_arm(
        "source_on", True, base_payload, prepared, fit
    )
    source_zero_model, source_zero_training = train_arm(
        "source_zero", False, base_payload, prepared, fit
    )
    for key in ("initial_state_sha256", "row_order_sha256", "node_sample_sha256"):
        if source_on_training[key] != source_zero_training[key]:
            raise AssertionError(f"paired training law diverged: {key}")
    score = score_pair(
        source_on_model, source_zero_model, prepared, dataset, held
    )
    rows = score.pop("rows")
    base_values = np.asarray([row["base_foldJ_mse"] for row in rows], dtype=np.float64)
    on_values = np.asarray([row["source_on_mse"] for row in rows], dtype=np.float64)
    improvement = base_values - on_values
    win_fraction = float(np.mean(improvement > 0.0))
    z_value = paired_z(improvement)
    gates = {
        "start5_exact_q_oracle_strictly_below_1e_8": oracle_endpoint < ORACLE_GATE,
        "source_on_endpoint_at_most_2_79555972111794e_8": score[
            "source_on_endpoint_mse"
        ]
        <= ENDPOINT_GATE,
        "source_on_strictly_beats_source_zero_endpoint": score[
            "source_on_endpoint_mse"
        ]
        < score["source_zero_endpoint_mse"],
        "source_on_strictly_beats_source_zero_local_q": score[
            "source_on_local_q_mse"
        ]
        < score["source_zero_local_q_mse"],
        "source_on_strictly_beats_frozen_independent256": score[
            "source_on_endpoint_mse"
        ]
        < FROZEN_INDEPENDENT256_ENDPOINT,
        "network_win_fraction_vs_foldJ_at_least_0_60": win_fraction
        >= MIN_WIN_FRACTION,
        "paired_network_z_vs_foldJ_at_least_3": z_value >= MIN_Z,
        "all_scored_metrics_finite": bool(
            np.all(
                np.isfinite(
                    np.asarray(
                        [
                            *score.values(),
                            win_fraction,
                            z_value,
                        ],
                        dtype=np.float64,
                    )
                )
            )
        ),
    }
    passed = all(gates.values())
    model_payload = {
        "schema": "public-full688-d11-cubic-action-transpose-fold0-models-v1",
        "fold": FOLD,
        "source_on_state_dict": copy.deepcopy(source_on_model.state_dict()),
        "source_zero_state_dict": copy.deepcopy(source_zero_model.state_dict()),
        "source_on_training": source_on_training,
        "source_zero_training": source_zero_training,
    }
    save_torch_exclusive_durable(MODELS, model_payload)
    result = {
        "schema": "public-full688-d11-cubic-action-transpose-fold0-result-v1",
        "status": (
            "PASS_FOLD0_SENTINEL__HOLD_NO_REMAINING_FOLD_AUTHORITY"
            if passed
            else "FAIL_FOLD0_SENTINEL__TERMINAL_NO_RETRY"
        ),
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "fold": FOLD,
        "fit_rows": len(fit),
        "held_rows": held.tolist(),
        "bindings": {
            "claim_sha256": sha256(CLAIM),
            "release_sha256": sha256(release_path),
            "runner_sha256": sha256(Path(__file__)),
            "feature_generator_sha256": sha256(FEATURE_SOURCE),
            "static_verification_sha256": sha256(STATIC_VERIFICATION),
            "models_sha256": sha256(MODELS),
            "native22_models_sha256": NATIVE22_MODELS_SHA256,
            "target_bundle_sha256": TARGETS_SHA256,
        },
        "aggregate": {
            **score,
            "pretraining_start5_direct_transport_oracle_endpoint_mse": oracle_endpoint,
            "network_win_fraction_vs_foldJ_base": win_fraction,
            "paired_network_z_vs_foldJ_base": z_value,
            "frozen_independent256_fold0_endpoint_mse": FROZEN_INDEPENDENT256_ENDPOINT,
            "maximum_process_rss_bytes": enforce_rss("result"),
        },
        "paired_training": {
            "same_initial_state": source_on_training["initial_state_sha256"]
            == source_zero_training["initial_state_sha256"],
            "same_row_order": source_on_training["row_order_sha256"]
            == source_zero_training["row_order_sha256"],
            "same_node_samples": source_on_training["node_sample_sha256"]
            == source_zero_training["node_sample_sha256"],
            "source_on": source_on_training,
            "source_zero": source_zero_training,
        },
        "gates": gates,
        "passed_all_gates": passed,
        "per_network": rows,
        "access": {
            "target_role": "observer_fit",
            "target_columns": ["mlp_id", "all_layer_means"],
            "observer_selection_opened": False,
            "names_scores_board_opened": False,
            "derived_c_or_b_cache_created": False,
            "remaining_folds_run": [],
            "all_data_fit": False,
            "integration": False,
            "meter": False,
            "submission": False,
        },
        "decision": (
            "HOLD_AND_REQUEST_SEPARATE_AUTHORITY_FOR_UNTOUCHED_FOLDS_1_TO_3"
            if passed
            else "CLOSE_THIS_SPELLING__DO_NOT_RUN_FOLDS_1_TO_3"
        ),
    }
    write_exclusive_durable(RESULT, result)
    print(
        json.dumps(
            {
                "stage": "terminal_fold0_result",
                "status": result["status"],
                **result["aggregate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def preflight(release_path: Path | None) -> None:
    verify_fixed_inputs(include_target_bytes=False)
    dummy = {
        "j_centers": np.zeros((DEPTH, 19), dtype=np.float32),
        "j_scales": np.ones((DEPTH, 19), dtype=np.float32),
        "j_layer_rms": np.ones(DEPTH, dtype=np.float32),
    }
    model, _ = make_model(dummy)
    runtime_exact = (
        sys.version_info[:3] == (3, 11, 5)
        and np.__version__ == "1.26.4"
        and torch.__version__ == "2.7.0"
    )
    checks: dict[str, Any] = {
        "runtime_python": ".".join(map(str, sys.version_info[:3])),
        "runtime_numpy": np.__version__,
        "runtime_torch": torch.__version__,
        "runtime_exact": runtime_exact,
        "parameter_count_6539": sum(value.numel() for value in model.parameters())
        == 6_539,
        "claim_absent": not CLAIM.exists(),
        "result_absent": not RESULT.exists(),
        "models_absent": not MODELS.exists(),
        "failure_absent": not FAILURE.exists(),
        "release_supplied": release_path is not None,
        "target_values_decoded": 0,
        "derived_feature_caches_created": 0,
    }
    if release_path is not None:
        try:
            validate_release(release_path.resolve())
            checks["release_validation"] = "PASS"
        except Exception as error:
            checks["release_validation"] = f"HOLD:{type(error).__name__}:{error}"
    ready = (
        all(
            checks[key] is True
            for key in (
                "runtime_exact",
                "parameter_count_6539",
                "claim_absent",
                "result_absent",
                "models_absent",
                "failure_absent",
                "release_supplied",
            )
        )
        and checks.get("release_validation") == "PASS"
    )
    print(
        json.dumps(
            {
                "schema": "public-full688-d11-cubic-action-transpose-fold0-preflight-v1",
                "status": "READY_FOR_EXACTLY_ONE_FOLD0_RUN" if ready else "HOLD",
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path, default=RELEASE_DEFAULT)
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight(arguments.release)
        return
    try:
        execute(arguments.release.resolve())
    except BaseException as error:
        record_failure_if_owned(error)
        raise


if __name__ == "__main__":
    main()
