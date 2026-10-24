#!/Users/matt/anaconda3/bin/python3
"""One-shot fold-0 test of four owned L3/L3^2 fields on the sealed 82 parent.

The candidate is exactly the frozen 82-input q head plus four target-blind
product-before-transport fields.  The paired control has the identical
86-input architecture and zeros only those four coordinates after fit-fold
normalization.  No cache is written and no target is opened until a durable
claim follows a conditional release and two independent static audits.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
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
PARENT = P2 / "public_q24_abs_sq_message_qhead_oof_v1"
FROZEN_FIELDS = P2 / "public_q24_owned_l3_l3square_forward_fields_v1"
DIRECT = P2 / "public_full688_d11_cubic_action_transpose_q_sentinel_v1"

PREREGISTRATION = HERE / "PREREGISTRATION.json"
STATIC_VERIFICATION = HERE / "STATIC_VERIFICATION.json"
CLAIM_CONTRACT = HERE / "FOLD0_RUN_CLAIM_CONTRACT.json"
RELEASE = HERE / "FOLD0_EXECUTION_RELEASE.json"
H4_RUNNER_AUDIT = HERE / "INDEPENDENT_H4_RUNNER_AUDIT.json"
CUBIC_RUNNER_AUDIT = HERE / "INDEPENDENT_CUBIC_RUNNER_AUDIT.json"
CLAIM = HERE / "FOLD0_RUN_CLAIM.json"
RESULT = HERE / "FOLD0_RESULT.json"
MODELS = HERE / "FOLD0_MODELS.pt"
FAILURE = HERE / "FOLD0_TERMINAL_FAILURE.json"
PRETARGET_CLOSURE = HERE / "PRETARGET_TERMINAL_CLOSURE.json"

PARENT_RUNNER = PARENT / "run_oof.py"
PARENT_PREREGISTRATION = PARENT / "PREREGISTRATION.json"
PARENT_RESULT = PARENT / "RESULT.json"
PARENT_MODELS = PARENT / "OOF_MODELS.pt"
PARENT_INTEGRITY = PARENT / "INTEGRITY_SUMMARY.json"
SIGNED_SUMMARY = PARENT / "SIGNED_CACHE_SUMMARY.json"
ABS_SQ_SUMMARY = P2 / "public_q24_raw_abs_sq_messages_v1/CACHE_SUMMARY.json"
FROZEN_GENERATOR = FROZEN_FIELDS / "feature_generator.py"
FROZEN_IMPLEMENTATION = FROZEN_FIELDS / "IMPLEMENTATION_FREEZE.json"
FROZEN_STATIC = FROZEN_FIELDS / "STATIC_VERIFICATION.json"
GENERATOR_AUDIT = FROZEN_FIELDS / "INDEPENDENT_GENERATOR_AUDIT_RECEIPT.json"
DIRECT_RESULT = DIRECT / "FOLD0_RESULT.json"
DIRECT_H4_AUDIT = DIRECT / "INDEPENDENT_FOLD0_TERMINAL_AUDIT.json"
DIRECT_CUBIC_AUDIT = DIRECT / "INDEPENDENT_TERMINAL_RESULT_AUDIT.json"

J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
QSAMPLE_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
FOLD_MODELS = (
    P2
    / "public_q24_signed_pair_qinnovation_distillation_v1"
    / "NATIVE22_CAPACITY_OUTPUT_V1"
    / "MODELS.pt"
)
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"

PARENT_RUNNER_SHA256 = "e5028cfe8cd088abc6f4997d9995f2c6b1ba65afd1ee280e763856578a6390ba"
PARENT_PREREGISTRATION_SHA256 = "3e30b7f9c9ada81a7e15b2e6ac3792af5f47ddab7c176f808b162fafa665a4d1"
PARENT_RESULT_SHA256 = "87624c36e90e3db7e5592ff8eb56acf5696a37d9ad0f2dd7ed656394301f98b4"
PARENT_MODELS_SHA256 = "21572ccd20eb339057c9c32a127fd794c14355dc654df517a27091000ee807d2"
PARENT_INTEGRITY_SHA256 = "73a73b492469c861f68fbc7170c30c75225a72651ec0d45c913ee21ce73f1a5e"
SIGNED_SUMMARY_SHA256 = "fbe6656b064847ee5895f87e4333ff4753bb5339b0bd78d8353371d6bbf0aa9d"
ABS_SQ_SUMMARY_SHA256 = "180e2f3458f64552583877dec80c8558e61bf0bebda7676f0921df9db823e9e9"
FROZEN_GENERATOR_SHA256 = "5f4ed4a777f5ab80141bc2e3d1ced73b973f7a60b9f694dd330ca461b545c086"
FROZEN_IMPLEMENTATION_SHA256 = "ac8be2bb247e505f37919823e988c8622058bf45385bd0faa5cdc223610f9666"
FROZEN_STATIC_SHA256 = "905ccaf22a1444aede1df641155a3a36fadd8690958218a36566f64d095f0d94"
GENERATOR_AUDIT_SHA256 = "49813236452e98d92e8e71ba8c76b984d615f1a2db20f86b3f3221610fc7ee56"
DIRECT_RESULT_SHA256 = "e1ed27d0882dddb09698da78d167f7e34096ce44742b210b64792a3d20babd7f"
DIRECT_H4_AUDIT_SHA256 = "5c7d15bc158db599f638d0ee512d4dac435aac7c04d1f5793ac590f82d9e0349"
DIRECT_CUBIC_AUDIT_SHA256 = "b527b9c24bc4a74c3da0ee6259bbfe100b7d1a2696c454d548cec876610bd0da"
J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
QREF_SOURCE_SHA256 = "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2"
QSAMPLE_SOURCE_SHA256 = "1e4c906936f4685bd800f922ded6c582af8213ecad1fbe41da651faff82be9c3"
FOLD_MODELS_SHA256 = "cff45c489c743f5de31454875e7dedfeb7d5a0fb9102663bd756b91dea482ee4"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"

ROWS = 688
WIDTH = 1024
DEPTH = 16
FOLD = 0
FIT = 516
HELD = 172
SUFFIX = tuple(range(10, 16))
LOCAL_INPUTS = 22
PARENT_EXTRA_INPUTS = 60
FOUR_INPUTS = 4
INPUTS = 86
HIDDEN = 256
HIDDEN_LAYERS = 3
PARAMETERS = 155_398
MODEL_SEED = 1_362_178_069
EPOCHS = 13
NETWORK_BATCH = 4
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
MAX_RSS_BYTES = 8 * (1 << 30)

PARENT_FOLD0_ENDPOINT = 3.0023717906766396e-8
PARENT_FOLD0_LOCAL_Q = 5.276099810147771e-9
PARENT_FOLD0_ORACLE = 6.1455295488600225e-9
PARENT_FOLD0_FOLDJ = 3.0923227278552953e-8
MIN_GAP_RECOVERY = 0.08
ENDPOINT_THRESHOLD = 2.8113462838133887e-8
EXPECTED_Q_RMS = np.asarray(
    [
        7.048059342196211e-5,
        7.202130655059591e-5,
        7.333858957281336e-5,
        7.466313400072977e-5,
        7.585606363136321e-5,
        7.653995999135077e-5,
    ],
    dtype=np.float32,
)

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


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise AssertionError(f"frozen byte binding changed: {path}")


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


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


def verify_frozen_lineage(
    *,
    include_target_bytes: bool,
    include_cache_manifests: bool,
) -> None:
    expected = {
        PARENT_RUNNER: PARENT_RUNNER_SHA256,
        PARENT_PREREGISTRATION: PARENT_PREREGISTRATION_SHA256,
        PARENT_RESULT: PARENT_RESULT_SHA256,
        PARENT_MODELS: PARENT_MODELS_SHA256,
        PARENT_INTEGRITY: PARENT_INTEGRITY_SHA256,
        FROZEN_GENERATOR: FROZEN_GENERATOR_SHA256,
        FROZEN_IMPLEMENTATION: FROZEN_IMPLEMENTATION_SHA256,
        FROZEN_STATIC: FROZEN_STATIC_SHA256,
        GENERATOR_AUDIT: GENERATOR_AUDIT_SHA256,
        DIRECT_RESULT: DIRECT_RESULT_SHA256,
        DIRECT_H4_AUDIT: DIRECT_H4_AUDIT_SHA256,
        DIRECT_CUBIC_AUDIT: DIRECT_CUBIC_AUDIT_SHA256,
        J_SOURCE: J_SOURCE_SHA256,
        QREF_SOURCE: QREF_SOURCE_SHA256,
        QSAMPLE_SOURCE: QSAMPLE_SOURCE_SHA256,
        FOLD_MODELS: FOLD_MODELS_SHA256,
    }
    if include_cache_manifests:
        expected[SIGNED_SUMMARY] = SIGNED_SUMMARY_SHA256
        expected[ABS_SQ_SUMMARY] = ABS_SQ_SUMMARY_SHA256
    if include_target_bytes:
        expected[TARGETS] = TARGETS_SHA256
    for path, digest in expected.items():
        verify_file(path, digest)

    parent = read_json(PARENT_RESULT)
    parent_fold = next(row for row in parent["fold_results"] if row["fold"] == FOLD)
    direct = read_json(DIRECT_RESULT)
    if (
        parent.get("status") != "FAIL_MATERIAL_OOF__STOP_NO_REFIT"
        or float(parent_fold["candidate_endpoint_mse"]) != PARENT_FOLD0_ENDPOINT
        or float(parent_fold["candidate_local_q_mse"]) != PARENT_FOLD0_LOCAL_Q
        or float(parent_fold["exact_q_oracle_endpoint_mse"]) != PARENT_FOLD0_ORACLE
        or float(parent_fold["foldJ_base_endpoint_mse"]) != PARENT_FOLD0_FOLDJ
        or direct.get("status") != "FAIL_FOLD0_SENTINEL__TERMINAL_NO_RETRY"
        or "FOUR_FIELD_SUCCESSOR_INTERLOCK_SATISFIED"
        not in read_json(DIRECT_CUBIC_AUDIT).get("status", "")
        or not read_json(DIRECT_H4_AUDIT).get("status", "").startswith(
            "PASS_INTEGRITY__FAIL"
        )
        or not read_json(GENERATOR_AUDIT).get("status", "").startswith(
            "PASS_STATIC_INTERFACE"
        )
    ):
        raise AssertionError("parent/direct/four-field promotion interlock changed")
    computed = PARENT_FOLD0_ENDPOINT - MIN_GAP_RECOVERY * (
        PARENT_FOLD0_ENDPOINT - PARENT_FOLD0_ORACLE
    )
    if computed != ENDPOINT_THRESHOLD or not PARENT_FOLD0_ORACLE < 1.0e-8:
        raise AssertionError("frozen fold0 oracle/gap threshold changed")


def expected_audit_bindings() -> dict[str, str]:
    return {
        "runner_sha256": sha256(Path(__file__)),
        "feature_adapter_sha256": sha256(HERE / "feature_generator.py"),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "release_sha256": sha256(RELEASE),
        "claim_contract_sha256": sha256(CLAIM_CONTRACT),
        "static_verification_sha256": sha256(STATIC_VERIFICATION),
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "parent_integrity_sha256": PARENT_INTEGRITY_SHA256,
        "frozen_generator_sha256": FROZEN_GENERATOR_SHA256,
        "frozen_implementation_sha256": FROZEN_IMPLEMENTATION_SHA256,
        "direct_result_sha256": DIRECT_RESULT_SHA256,
        "direct_h4_terminal_audit_sha256": DIRECT_H4_AUDIT_SHA256,
        "direct_cubic_terminal_audit_sha256": DIRECT_CUBIC_AUDIT_SHA256,
        "independent_generator_audit_sha256": GENERATOR_AUDIT_SHA256,
        "signed_cache_summary_sha256": SIGNED_SUMMARY_SHA256,
        "abs_sq_cache_summary_sha256": ABS_SQ_SUMMARY_SHA256,
    }


def validate_independent_audit(path: Path, role: str) -> str:
    if not path.is_file():
        raise PermissionError(f"missing independent {role} runner audit")
    audit = read_json(path)
    if (
        audit.get("schema") != "owned-l3-l3square-fold0-independent-runner-audit-v1"
        or audit.get("auditor_role") != role
        or not str(audit.get("status", "")).startswith("PASS_")
    ):
        raise PermissionError(f"invalid independent {role} runner audit")
    for key, value in expected_audit_bindings().items():
        if audit.get("bindings", {}).get(key) != value:
            raise PermissionError(f"independent {role} audit binding changed: {key}")
    access = audit.get("access", {})
    for key in (
        "target_values_opened_or_decoded",
        "target_blind_cache_rows_opened",
        "training_or_scoring_runs",
        "derived_feature_caches_created",
        "estimator_calls",
    ):
        if access.get(key) != 0:
            raise PermissionError(f"independent {role} audit access changed: {key}")
    return sha256(path)


def validate_release(path: Path) -> dict[str, Any]:
    release = read_json(path)
    if (
        path.resolve() != RELEASE.resolve()
        or release.get("schema")
        != "owned-l3-l3square-fold0-conditional-execution-release-v1"
        or release.get("status")
        != "RELEASE_EXACTLY_ONE_FOLD0_PAIRED_SENTINEL_AFTER_TWO_STATIC_AUDITS"
        or release.get("sentinel_fold") != 0
        or release.get("one_shot") is not True
        or release.get("retry") is not False
        or release.get("role") != "observer_fit"
        or release.get("target_columns") != ["mlp_id", "all_layer_means"]
        or release.get("target_decode") is not True
        or release.get("sentinel_execution") is not True
        or release.get("target_blind_cache_read") is not True
        or release.get("derived_cache_write") is not False
        or release.get("training") is not True
        or release.get("paired_arms") != ["source_on", "source_zero"]
        or release.get("feature_subset_or_ablation") is not False
        or release.get("remaining_folds") is not False
        or release.get("observer_selection") is not False
        or release.get("all_data_fit") is not False
        or release.get("package") is not False
        or release.get("names") is not False
        or release.get("scores") is not False
        or release.get("network") is not False
        or release.get("integration") is not False
        or release.get("meter") is not False
        or release.get("estimator") is not False
        or release.get("board") is not False
        or release.get("submission") is not False
    ):
        raise PermissionError("release is not the exact narrow fold-0 authority")
    expected = {
        **{key: value for key, value in expected_audit_bindings().items() if key != "release_sha256"},
        "frozen_static_sha256": FROZEN_STATIC_SHA256,
        "parent_models_sha256": PARENT_MODELS_SHA256,
        "fold_models_sha256": FOLD_MODELS_SHA256,
        "target_bundle_sha256": TARGETS_SHA256,
        "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
        "fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
    }
    for key, value in expected.items():
        if release.get("bindings", {}).get(key) != value:
            raise PermissionError(f"release binding changed: {key}")
    validate_independent_audit(H4_RUNNER_AUDIT, "h4_reverse_adjoint")
    validate_independent_audit(CUBIC_RUNNER_AUDIT, "cubic_vertex_provider")
    return release


def verify_runtime_contract() -> None:
    if (
        sys.version_info[:3] != (3, 11, 5)
        or np.__version__ != "1.26.4"
        or torch.__version__ != "2.7.0"
    ):
        raise RuntimeError("runtime tuple changed")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(name) != "4":
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
    return value.get("owner_pid") == os.getpid() and value.get("owner_nonce") == ACTIVE_NONCE


def record_failure_if_owned(error: BaseException) -> None:
    if not claim_owned() or RESULT.exists() or FAILURE.exists():
        return
    write_exclusive_durable(
        FAILURE,
        {
            "schema": "owned-l3-l3square-fold0-terminal-failure-v1",
            "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
            "failed_utc": utc_now(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "claim_sha256": sha256(CLAIM),
            "decision": "STOP_NO_RETRY_NO_FOLDS_1_TO_3",
        },
    )


def write_pretarget_candidate_failure(
    reason: str,
    diagnostic: dict[str, Any],
) -> None:
    """Durably close a measured frozen pretarget candidate falsifier."""

    write_exclusive_durable(
        PRETARGET_CLOSURE,
        {
            "schema": "owned-l3-l3square-fold0-pretarget-terminal-closure-v1",
            "status": "TERMINAL_PRETARGET_CANDIDATE_FALSIFIER__NO_RETRY",
            "failed_utc": utc_now(),
            "reason": reason,
            "diagnostic": diagnostic,
            "claim_present": CLAIM.exists(),
            "target_values_opened_or_decoded": 0,
            "training_or_scoring_runs": 0,
            "derived_feature_cache_written": False,
            "decision": "CLOSE_FOUR_FIELD_BUNDLE__NO_RETRY_NO_FOLDS_1_TO_3",
        },
    )


@dataclass
class FeatureData:
    parent_extra: np.ndarray
    four_extra: np.ndarray
    parent_center: np.ndarray
    parent_scale: np.ndarray
    four_center: np.ndarray
    four_scale: np.ndarray
    parent_diagnostic: dict[str, float]
    four_diagnostic: dict[str, float]


class MessageHead86(nn.Module):
    def __init__(
        self,
        fold_j: nn.Module,
        four_center: np.ndarray,
        four_scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("centers", fold_j.centers.detach().clone())
        self.register_buffer("scales", fold_j.scales.detach().clone())
        self.register_buffer("layer_rms", fold_j.layer_rms.detach().clone())
        self.register_buffer("four_center", torch.as_tensor(four_center).detach().clone())
        self.register_buffer("four_scale", torch.as_tensor(four_scale).detach().clone())
        layers: list[nn.Module] = [nn.Linear(INPUTS, HIDDEN), nn.SiLU()]
        for _ in range(HIDDEN_LAYERS - 1):
            layers.extend((nn.Linear(HIDDEN, HIDDEN), nn.SiLU()))
        self.trunk = nn.Sequential(*layers)
        self.head_weight = nn.Parameter(torch.zeros(len(SUFFIX), HIDDEN))
        self.head_bias = nn.Parameter(torch.zeros(len(SUFFIX)))

    def source(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        base_state: torch.Tensor,
        base_q: torch.Tensor,
        parent_extra: torch.Tensor,
        four_extra: torch.Tensor,
        layer: int,
        *,
        source_on: bool,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        local = torch.cat(
            (
                raw[..., :1],
                tail,
                torch.clamp(base_state / rms, -8.0, 8.0)[..., None],
                torch.clamp(
                    base_q / torch.clamp(sigma * rms, min=2.0**-30), -8.0, 8.0
                )[..., None],
            ),
            dim=-1,
        )
        if source_on:
            four = four_extra
        else:
            # Zero only the four appended coordinates after fit normalization.
            four = torch.zeros_like(four_extra)
        features = torch.cat((local, parent_extra, four), dim=-1)
        if features.shape[-1] != INPUTS:
            raise AssertionError("86-input feature width changed")
        hidden = self.trunk(features)
        slot = SUFFIX.index(layer)
        standard = (
            torch.einsum("...h,h->...", hidden, self.head_weight[slot])
            + self.head_bias[slot]
        )
        return sigma * rms * standard


def canonical_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for value in model.state_dict().values():
        array = value.detach().cpu().numpy()
        digest.update(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def four_fit_statistics(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros((len(SUFFIX), FOUR_INPUTS), dtype=np.float64)
    square = np.zeros_like(total)
    for rank_value in fit:
        row = values[int(rank_value)].astype(np.float64)
        total += np.sum(row, axis=1, dtype=np.float64)
        square += np.sum(row * row, axis=1, dtype=np.float64)
    count = float(len(fit) * WIDTH)
    center = total / count
    variance = np.maximum(square / count - center * center, (2.0**-20) ** 2)
    return center.astype(np.float32), np.sqrt(variance).astype(np.float32)


def build_four_fields(dataset: Any, layer_rms: np.ndarray) -> np.ndarray:
    values = np.empty((ROWS, len(SUFFIX), WIDTH, FOUR_INPUTS), dtype=np.float32)
    for rank in range(ROWS):
        row = dataset.load(rank)
        weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
        built = feature_generator.build_features(
            row.raw_locals,
            row.sigma,
            weights,
            layer_rms,
            first_layer=SUFFIX[0],
            last_layer=SUFFIX[-1],
        )
        if built.layers != SUFFIX or built.field_order != feature_generator.FIELD_ORDER:
            raise AssertionError("four-field order changed")
        values[rank] = built.appended_features
        if (rank + 1) % 32 == 0 or rank + 1 == ROWS:
            print(
                json.dumps(
                    {
                        "stage": "build_four_fields_target_blind_in_memory",
                        "rows": rank + 1,
                        "derived_feature_cache_written": False,
                        "maximum_process_rss_bytes": enforce_rss(f"four rank {rank}"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return values


def normalize_four_in_place(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[str, float]:
    clipped = 0
    maximum = 0.0
    for rank in range(ROWS):
        standard = (
            values[rank].astype(np.float64) - center[:, None, :]
        ) / scale[:, None, :]
        clipped += int(np.sum(np.abs(standard) > 8.0))
        maximum = max(maximum, float(np.max(np.abs(standard))))
        values[rank] = np.clip(standard, -8.0, 8.0).astype(np.float32)
    return {
        "maximum_absolute_before_clip": maximum,
        "clip_fraction": clipped / values.size,
        "scale_minimum": float(np.min(scale)),
        "scale_maximum": float(np.max(scale)),
    }


def conditional_four_rank(
    parent_extra: np.ndarray,
    four_extra: np.ndarray,
    fit: np.ndarray,
    qsample: Any,
    dataset: Any,
    fold_j: nn.Module,
) -> dict[str, Any]:
    """Frozen target-free Schur diagnostic of new4 given the full parent82.

    To keep this pre-target gate cheap, it uses the 32 lowest fit ranks and
    the already frozen 256-node sampler at epoch zero.  Neither choice depends
    on a target or on the candidate result.
    """

    old_dim = 1 + LOCAL_INPUTS + PARENT_EXTRA_INPUTS
    gram_old = np.zeros((len(SUFFIX), old_dim, old_dim), dtype=np.float64)
    cross = np.zeros((len(SUFFIX), old_dim, FOUR_INPUTS), dtype=np.float64)
    gram_new = np.zeros((len(SUFFIX), FOUR_INPUTS, FOUR_INPUTS), dtype=np.float64)
    diagnostic_ranks = np.asarray(fit[:32], dtype=np.int64)
    with torch.no_grad():
        for rank_value in diagnostic_ranks:
            rank = int(rank_value)
            row = dataset.load(rank)
            raw = row.raw_locals
            sigma = row.sigma
            weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
            prediction = fold_j(
                torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))[None],
                torch.from_numpy(np.array(sigma, dtype=np.float32, copy=True))[None],
                torch.from_numpy(np.array(weights, dtype=np.float32, copy=True))[None],
                exposure=1.0,
            )
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("fold J returned non-tensor during rank diagnostic")
            state = prediction[0].cpu().numpy().astype(np.float32, copy=False)
            for slot, layer in enumerate(SUFFIX):
                indices = qsample.sample_indices(rank, layer, 0)
                phi = raw[layer, :, feature_generator.RAW_PHI].astype(np.float64)
                base_q = (
                    state[layer].astype(np.float64)
                    - phi
                    * (
                        state[layer - 1].astype(np.float64)
                        @ weights[layer].astype(np.float64)
                    )
                ).astype(np.float32)
                local = np.empty((WIDTH, LOCAL_INPUTS), dtype=np.float32)
                local[:, 0] = raw[layer, :, 0]
                local[:, 1:20] = np.clip(
                    (raw[layer, :, 1:] - fold_j.centers[layer].cpu().numpy())
                    / fold_j.scales[layer].cpu().numpy(),
                    -8.0,
                    8.0,
                )
                rms = float(fold_j.layer_rms[layer])
                local[:, 20] = np.clip(state[layer] / rms, -8.0, 8.0)
                local[:, 21] = np.clip(
                    base_q / np.maximum(sigma[layer] * rms, 2.0**-30),
                    -8.0,
                    8.0,
                )
                old = np.empty((len(indices), old_dim), dtype=np.float64)
                old[:, 0] = 1.0
                old[:, 1 : 1 + LOCAL_INPUTS] = local[indices].astype(np.float64)
                old[:, 1 + LOCAL_INPUTS :] = parent_extra[
                    rank, slot, indices
                ].astype(np.float64)
                new = four_extra[rank, slot, indices].astype(np.float64)
                gram_old[slot] += old.T @ old
                cross[slot] += old.T @ new
                gram_new[slot] += new.T @ new
    nodes_per_rank = int(qsample.SAMPLES_PER_LAYER)
    count = float(len(diagnostic_ranks) * nodes_per_rank)
    layer_receipts: list[dict[str, Any]] = []
    minimum_rank = FOUR_INPUTS
    for slot, layer in enumerate(SUFFIX):
        old = gram_old[slot] / count
        old_new = cross[slot] / count
        new = gram_new[slot] / count
        schur = new - old_new.T @ np.linalg.pinv(old, rcond=1.0e-12, hermitian=True) @ old_new
        schur = 0.5 * (schur + schur.T)
        eigenvalues = np.linalg.eigvalsh(schur)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
        tolerance = scale * max(int(count), old_dim + FOUR_INPUTS) * np.finfo(np.float64).eps
        rank = int(np.sum(eigenvalues > tolerance))
        minimum_rank = min(minimum_rank, rank)
        layer_receipts.append(
            {
                "layer": layer,
                "conditional_eigenvalues": eigenvalues.tolist(),
                "numerical_rank": rank,
                "rank_tolerance": tolerance,
                "conditional_trace": float(np.trace(schur)),
            }
        )
    return {
        "conditioning_features": "constant_plus_exact_local22_plus_fit_normalized_clipped_parent60",
        "new_features": list(feature_generator.FIELD_ORDER),
        "rank_selection": "32_lowest_fold0_fit_ranks",
        "node_selection": "frozen_qsample.sample_indices(rank,layer,epoch=0)",
        "fit_rows": int(len(diagnostic_ranks)),
        "nodes_per_layer": int(count),
        "minimum_layer_rank": minimum_rank,
        "passes_nonzero_conditional_rank_gate": minimum_rank > 0,
        "layers": layer_receipts,
    }


def q_rms_for(prepared: Any, fit: np.ndarray) -> np.ndarray:
    value = np.maximum(
        np.sqrt(np.mean(prepared.delta_q[fit].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)
    if not np.array_equal(value, EXPECTED_Q_RMS):
        raise AssertionError(f"fold0 q-rms supervision changed: {value.tolist()}")
    return value


def train_arm(
    name: str,
    source_on: bool,
    fold_j: nn.Module,
    prepared: Any,
    features: FeatureData,
    fit: np.ndarray,
    qsample: Any,
) -> tuple[MessageHead86, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = MessageHead86(fold_j, features.four_center, features.four_scale)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != PARAMETERS:
        raise AssertionError(f"parameter count changed: {parameter_count}")
    initial_hash = canonical_model_hash(model)
    q_rms = q_rms_for(prepared, fit)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
    )
    steps_per_epoch = len(fit) // NETWORK_BATCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * steps_per_epoch,
        eta_min=MIN_LEARNING_RATE,
    )
    order_seed = (MODEL_SEED ^ 0xA5A5A5A5 ^ (FOLD * 0x9E3779B1)) & 0xFFFF_FFFF
    rng = np.random.Generator(np.random.PCG64(order_seed))
    order_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        objective = 0.0
        model.train()
        for offset in range(0, len(order), NETWORK_BATCH):
            ranks = order[offset : offset + NETWORK_BATCH]
            if len(ranks) != NETWORK_BATCH:
                raise AssertionError("fit fold no longer divides into batches of four")
            losses: list[torch.Tensor] = []
            for slot, layer in enumerate(SUFFIX):
                indices = [qsample.sample_indices(int(rank), layer, epoch) for rank in ranks]
                for index in indices:
                    sample_digest.update(
                        np.ascontiguousarray(index.astype("<i4", copy=False)).tobytes()
                    )
                def gather(array: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(
                        np.ascontiguousarray(
                            np.concatenate(
                                [
                                    array[int(rank), slot, index]
                                    for rank, index in zip(ranks, indices, strict=True)
                                ]
                            )
                        )
                    )

                source = model.source(
                    gather(prepared.raw_suffix),
                    gather(prepared.sigma_suffix),
                    gather(prepared.base_state),
                    gather(prepared.base_q),
                    gather(features.parent_extra),
                    gather(features.four_extra),
                    layer,
                    source_on=source_on,
                )
                truth = gather(prepared.delta_q)
                losses.append(torch.mean(((source - truth) / float(q_rms[slot])).square()))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {name} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective += float(loss.detach()) * len(ranks)
        record = {
            "arm": name,
            "epoch": epoch,
            "sampled_normalized_q_loss": objective / len(fit),
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
        "sample_index_sha256": sample_digest.hexdigest(),
        "samples_per_layer_per_network_per_epoch": int(qsample.SAMPLES_PER_LAYER),
        "q_rms": q_rms.tolist(),
        "epochs": EPOCHS,
        "scheduler_t_max_steps": EPOCHS * steps_per_epoch,
        "source_zero_coordinates_zero_after_normalization": not source_on,
        "trace": trace,
    }


def paired_z(values: np.ndarray) -> float:
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    if standard_error == 0.0:
        return float("inf") if float(np.mean(values)) > 0.0 else 0.0
    return float(np.mean(values) / standard_error)


def score_pair(
    source_on_model: MessageHead86,
    source_zero_model: MessageHead86,
    prepared: Any,
    features: FeatureData,
    dataset: Any,
    held: np.ndarray,
    parent_by_rank: dict[int, float],
) -> dict[str, Any]:
    local_sse = {"source_on": 0.0, "source_zero": 0.0}
    local_count = len(held) * len(SUFFIX) * WIDTH
    rows: list[dict[str, Any]] = []
    source_on_model.eval()
    source_zero_model.eval()
    with torch.no_grad():
        for position, rank_value in enumerate(held):
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            states = np.zeros((3, WIDTH), dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                def tensor(array: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(np.ascontiguousarray(array[rank, slot]))

                args = (
                    tensor(prepared.raw_suffix),
                    tensor(prepared.sigma_suffix),
                    tensor(prepared.base_state),
                    tensor(prepared.base_q),
                    tensor(features.parent_extra),
                    tensor(features.four_extra),
                    layer,
                )
                source_on = source_on_model.source(*args, source_on=True).cpu().numpy()
                source_zero = source_zero_model.source(*args, source_on=False).cpu().numpy()
                truth = prepared.delta_q[rank, slot]
                local_sse["source_on"] += float(
                    np.sum((source_on.astype(np.float64) - truth) ** 2)
                )
                local_sse["source_zero"] += float(
                    np.sum((source_zero.astype(np.float64) - truth) ** 2)
                )
                sources = np.stack((source_on, source_zero, truth)).astype(np.float64)
                phi = prepared.raw_suffix[rank, slot, :, feature_generator.RAW_PHI].astype(
                    np.float64
                )
                states = (
                    phi[None, :]
                    * (states.astype(np.float64) @ weights[layer].astype(np.float64))
                    + sources
                ).astype(np.float32)
            residual = prepared.residual_final[rank].astype(np.float64)
            rows.append(
                {
                    "role_rank": rank,
                    "parent82_mse": parent_by_rank[rank],
                    "source_on_mse": float(
                        np.mean((residual - states[0].astype(np.float64)) ** 2)
                    ),
                    "source_zero_mse": float(
                        np.mean((residual - states[1].astype(np.float64)) ** 2)
                    ),
                    "exact_q_oracle_mse": float(
                        np.mean((residual - states[2].astype(np.float64)) ** 2)
                    ),
                    "foldJ_base_mse": float(np.mean(residual * residual)),
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
        "parent82_endpoint_mse": float(np.mean([row["parent82_mse"] for row in rows])),
        "exact_q_oracle_endpoint_mse": float(
            np.mean([row["exact_q_oracle_mse"] for row in rows])
        ),
        "foldJ_base_endpoint_mse": float(
            np.mean([row["foldJ_base_mse"] for row in rows])
        ),
    }


def execute(release_path: Path) -> None:
    global ACTIVE_NONCE
    if any(
        path.exists()
        for path in (CLAIM, RESULT, MODELS, FAILURE, PRETARGET_CLOSURE)
    ):
        raise FileExistsError("fold0 sentinel already claimed, completed, or failed")
    verify_runtime_contract()
    verify_frozen_lineage(
        include_target_bytes=False,
        include_cache_manifests=False,
    )
    validate_release(release_path)
    verify_frozen_lineage(
        include_target_bytes=True,
        include_cache_manifests=True,
    )

    parent = load_module("owned_four_parent82", PARENT_RUNNER)
    j = load_module("owned_four_j", J_SOURCE)
    qref = load_module("owned_four_qref", QREF_SOURCE)
    qsample = load_module("owned_four_qsample", QSAMPLE_SOURCE)
    if int(qsample.SAMPLES_PER_LAYER) != 256 or tuple(qsample.SUFFIX) != SUFFIX:
        raise AssertionError("frozen q node sampler changed")

    # Release explicitly allows these target-blind cache reads.  No target or
    # derived cache is touched while every receipt and live row is rehashed.
    parent.verify_cache_summary(
        ABS_SQ_SUMMARY,
        "q24-raw-abs-sq-message-cache-summary-v1",
        parent.abs_sq_path,
    )
    parent.verify_cache_summary(
        SIGNED_SUMMARY,
        "q24-target-blind-signed-raw20-message-cache-v1",
        parent.signed_path,
    )
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

    saved_folds = torch.load(FOLD_MODELS, map_location="cpu", weights_only=False)
    entry = saved_folds["folds"][str(FOLD)]
    fold_j = j.TransportEnergyMultiStateResidual(
        entry["j_centers"], entry["j_scales"], entry["j_layer_rms"]
    )
    fold_j.load_state_dict(entry["j_state_dict"], strict=True)
    fold_j.eval()
    dataset = qref.All688Dataset(records)

    # Build and normalize every target-blind feature in RAM before the claim.
    # The arrays are never serialized, and only fold-0 fit ranks set moments.
    parent_center, parent_scale = parent.extra_stats(fit)
    parent_extra, parent_diagnostic = parent.load_normalized_extra(
        parent_center, parent_scale
    )
    four_extra = build_four_fields(
        dataset, np.asarray(entry["j_layer_rms"], dtype=np.float32)
    )
    four_center, four_scale = four_fit_statistics(four_extra, fit)
    four_diagnostic = normalize_four_in_place(four_extra, four_center, four_scale)
    if not np.isfinite(parent_extra).all() or not np.isfinite(four_extra).all():
        write_pretarget_candidate_failure(
            "NONFINITE_NORMALIZED_TARGET_BLIND_FIELD",
            {
                "parent60_all_finite": bool(np.isfinite(parent_extra).all()),
                "new4_all_finite": bool(np.isfinite(four_extra).all()),
            },
        )
        raise FloatingPointError("nonfinite normalized target-blind feature")
    rank_diagnostic = conditional_four_rank(
        parent_extra,
        four_extra,
        fit,
        qsample,
        dataset,
        fold_j,
    )
    if rank_diagnostic["passes_nonzero_conditional_rank_gate"] is not True:
        write_pretarget_candidate_failure(
            "ZERO_CONDITIONAL_NEW4_RANK_GIVEN_FULL_PARENT82",
            rank_diagnostic,
        )
        raise RuntimeError("target-free four-field conditional rank is zero")
    features = FeatureData(
        parent_extra,
        four_extra,
        parent_center,
        parent_scale,
        four_center,
        four_scale,
        parent_diagnostic,
        four_diagnostic,
    )
    nonce = hashlib.sha256(
        os.getpid().to_bytes(8, "big", signed=False) + os.urandom(32)
    ).hexdigest()
    write_exclusive_durable(
        CLAIM,
        {
            "schema": "owned-l3-l3square-fold0-run-claim-v1",
            "status": "CLAIMED_AFTER_RELEASE_CACHE_VERIFICATION_AND_FEATURE_BUILD_BEFORE_TARGET_DECODE",
            "owner_pid": os.getpid(),
            "owner_nonce": nonce,
            "claimed_utc": utc_now(),
            "one_shot_no_retry": True,
            "sentinel_fold": FOLD,
            "target_values_decoded_before_claim": 0,
            "target_bundle_bytes_hashed_before_claim_without_semantic_decode": True,
            "target_blind_cache_rows_verified": 2 * ROWS,
            "derived_feature_cache_written": False,
            "native_storage_bytes_verified": native_storage,
            "four_field_joint_arm_only": True,
            "feature_subset_or_retry_authorized": False,
            "target_free_minimum_layer_conditional_rank": rank_diagnostic[
                "minimum_layer_rank"
            ],
            "bindings": {
                "runner_sha256": sha256(Path(__file__)),
                "preregistration_sha256": sha256(PREREGISTRATION),
                "release_sha256": sha256(RELEASE),
                "claim_contract_sha256": sha256(CLAIM_CONTRACT),
                "h4_runner_audit_sha256": sha256(H4_RUNNER_AUDIT),
                "cubic_runner_audit_sha256": sha256(CUBIC_RUNNER_AUDIT),
                "parent_result_sha256": PARENT_RESULT_SHA256,
                "frozen_generator_sha256": FROZEN_GENERATOR_SHA256,
                "target_bundle_sha256": TARGETS_SHA256,
            },
            "terminal_rule": "ANY_POST_CLAIM_FAILURE_IS_FINAL__NO_RETRY__NO_FOLDS_1_TO_3",
        },
    )
    ACTIVE_NONCE = nonce

    # First semantic target read.  Everything above is target blind.
    targets = qref.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack(
            [
                np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False)
                for rank in range(ROWS)
            ]
        ),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    prepared = qref.prepare_all(j, fold_j, dataset, error)
    del targets, hosts, error, saved_folds

    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED & 0xFFFF_FFFF)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    source_on_model, source_on_training = train_arm(
        "source_on", True, fold_j, prepared, features, fit, qsample
    )
    source_zero_model, source_zero_training = train_arm(
        "source_zero", False, fold_j, prepared, features, fit, qsample
    )
    for key in (
        "initial_state_sha256",
        "row_order_sha256",
        "sample_index_sha256",
        "q_rms",
        "scheduler_t_max_steps",
    ):
        if source_on_training[key] != source_zero_training[key]:
            raise AssertionError(f"paired training law diverged: {key}")
    if source_on_training["row_order_sha256"] != "763b453ac7c9d284d00ffccb38ff7ddd4938f98167dbf35806a444d82472eadb":
        raise AssertionError("parent fold0 row ordering changed")
    if source_on_training["sample_index_sha256"] != "db70f29f998354929a34c67e23a145372f271426b967d923e92e7c3f9fed2569":
        raise AssertionError("parent fold0 node sampling changed")

    parent_result = read_json(PARENT_RESULT)
    parent_by_rank = {
        int(row["role_rank"]): float(row["candidate_mse"])
        for row in parent_result["per_network"]
    }
    score = score_pair(
        source_on_model,
        source_zero_model,
        prepared,
        features,
        dataset,
        held,
        parent_by_rank,
    )
    rows = score.pop("rows")
    if (
        abs(score["parent82_endpoint_mse"] - PARENT_FOLD0_ENDPOINT) > 1.0e-18
        or abs(score["exact_q_oracle_endpoint_mse"] - PARENT_FOLD0_ORACLE) > 1.0e-18
        or abs(score["foldJ_base_endpoint_mse"] - PARENT_FOLD0_FOLDJ) > 1.0e-18
    ):
        raise AssertionError("fold0 parent/oracle/base replay changed")

    on = score["source_on_endpoint_mse"]
    zero = score["source_zero_endpoint_mse"]
    oracle = score["exact_q_oracle_endpoint_mse"]
    recovery = (PARENT_FOLD0_ENDPOINT - on) / (
        PARENT_FOLD0_ENDPOINT - PARENT_FOLD0_ORACLE
    )
    isolated_recovery = (zero - on) / (zero - oracle) if zero > oracle else float("-inf")
    on_local = score["source_on_local_q_mse"]
    zero_local = score["source_zero_local_q_mse"]
    source_improvement = np.asarray(
        [row["source_zero_mse"] - row["source_on_mse"] for row in rows],
        dtype=np.float64,
    )
    gates = {
        "parent_fold0_exact_q_oracle_strictly_below_1e_8": oracle < 1.0e-8,
        "source_on_endpoint_at_most_2_8113462838133887e_8": on
        <= ENDPOINT_THRESHOLD,
        "parent82_to_exact_q_oracle_gap_recovery_at_least_0_08": recovery
        >= MIN_GAP_RECOVERY,
        "source_on_strictly_beats_source_zero_endpoint": on < zero,
        "source_on_strictly_beats_source_zero_local_q": on_local < zero_local,
        "all_scored_metrics_finite": bool(
            np.all(
                np.isfinite(
                    np.asarray(
                        [
                            *score.values(),
                            recovery,
                            isolated_recovery,
                            float(np.mean(source_improvement > 0.0)),
                            paired_z(source_improvement),
                        ],
                        dtype=np.float64,
                    )
                )
            )
        ),
    }
    passed = all(gates.values())
    model_payload = {
        "schema": "owned-l3-l3square-fold0-models-v1",
        "fold": FOLD,
        "source_on_state_dict": copy.deepcopy(source_on_model.state_dict()),
        "source_zero_state_dict": copy.deepcopy(source_zero_model.state_dict()),
        "parent_extra_center": parent_center,
        "parent_extra_scale": parent_scale,
        "four_center": four_center,
        "four_scale": four_scale,
        "source_on_training": source_on_training,
        "source_zero_training": source_zero_training,
    }
    save_torch_exclusive_durable(MODELS, model_payload)
    result = {
        "schema": "owned-l3-l3square-fold0-result-v1",
        "status": (
            "PASS_FOLD0_FOUR_FIELD_SENTINEL__HOLD_NO_REMAINING_FOLD_AUTHORITY"
            if passed
            else "FAIL_FOLD0_FOUR_FIELD_SENTINEL__TERMINAL_NO_RETRY"
        ),
        "completed_utc": utc_now(),
        "fold": FOLD,
        "fit_rows": len(fit),
        "held_rows": held.tolist(),
        "bindings": {
            "claim_sha256": sha256(CLAIM),
            "release_sha256": sha256(RELEASE),
            "runner_sha256": sha256(Path(__file__)),
            "feature_adapter_sha256": sha256(HERE / "feature_generator.py"),
            "models_sha256": sha256(MODELS),
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "frozen_generator_sha256": FROZEN_GENERATOR_SHA256,
            "h4_runner_audit_sha256": sha256(H4_RUNNER_AUDIT),
            "cubic_runner_audit_sha256": sha256(CUBIC_RUNNER_AUDIT),
            "target_bundle_sha256": TARGETS_SHA256,
        },
        "architecture": {
            "inputs": INPUTS,
            "local22": LOCAL_INPUTS,
            "sealed_parent_messages": PARENT_EXTRA_INPUTS,
            "appended_four_fields": FOUR_INPUTS,
            "hidden": HIDDEN,
            "hidden_layers": HIDDEN_LAYERS,
            "heads": len(SUFFIX),
            "parameters": PARAMETERS,
        },
        "aggregate": {
            **score,
            "frozen_parent82_fold0_endpoint_mse": PARENT_FOLD0_ENDPOINT,
            "frozen_parent82_fold0_local_q_mse": PARENT_FOLD0_LOCAL_Q,
            "frozen_parent82_fold0_exact_q_oracle_endpoint_mse": PARENT_FOLD0_ORACLE,
            "source_on_gain_vs_parent82_endpoint": PARENT_FOLD0_ENDPOINT - on,
            "source_on_gain_vs_source_zero_endpoint": zero - on,
            "source_on_gain_vs_source_zero_local_q": zero_local - on_local,
            "parent82_to_exact_q_oracle_gap_recovery": recovery,
            "isolated_source_zero_to_oracle_gap_recovery": isolated_recovery,
            "minimum_required_gap_recovery": MIN_GAP_RECOVERY,
            "frozen_endpoint_threshold": ENDPOINT_THRESHOLD,
            "network_win_fraction_source_on_vs_zero": float(
                np.mean(source_improvement > 0.0)
            ),
            "paired_network_z_source_on_vs_zero": paired_z(source_improvement),
            "maximum_process_rss_bytes": enforce_rss("result"),
        },
        "feature_diagnostic": {
            "parent60": parent_diagnostic,
            "four_fields": four_diagnostic,
            "target_free_conditional_rank": rank_diagnostic,
            "four_field_order": list(feature_generator.FIELD_ORDER),
            "four_fields_constructed_in_memory_only": True,
            "derived_feature_cache_written": False,
        },
        "paired_training": {
            "same_initial_state": source_on_training["initial_state_sha256"]
            == source_zero_training["initial_state_sha256"],
            "same_row_order": source_on_training["row_order_sha256"]
            == source_zero_training["row_order_sha256"],
            "same_node_samples": source_on_training["sample_index_sha256"]
            == source_zero_training["sample_index_sha256"],
            "same_q_rms": source_on_training["q_rms"] == source_zero_training["q_rms"],
            "source_on": source_on_training,
            "source_zero": source_zero_training,
        },
        "gates": gates,
        "passed_all_gates": passed,
        "per_network": rows,
        "access": {
            "target_role": "observer_fit",
            "target_columns": ["mlp_id", "all_layer_means"],
            "target_blind_parent_cache_rows_read": 2 * ROWS,
            "derived_feature_cache_written": False,
            "observer_selection_opened": False,
            "names_scores_board_opened": False,
            "remaining_folds_run": [],
            "feature_subset_or_ablation_runs": 0,
            "all_data_fit": False,
            "integration": False,
            "meter": False,
            "package": False,
            "submission": False,
        },
        "decision": (
            "HOLD_AND_REQUEST_SEPARATE_AUTHORITY_FOR_UNTOUCHED_FOLDS_1_TO_3"
            if passed
            else "CLOSE_FOUR_FIELD_BUNDLE__DO_NOT_RUN_FOLDS_1_TO_3"
        ),
    }
    write_exclusive_durable(RESULT, result)
    print(
        json.dumps(
            {"stage": "terminal_fold0_result", "status": result["status"], **result["aggregate"]},
            sort_keys=True,
        ),
        flush=True,
    )


def preflight(release_path: Path) -> None:
    verify_frozen_lineage(
        include_target_bytes=False,
        include_cache_manifests=False,
    )
    checks: dict[str, Any] = {
        "architecture_input86_hidden256x3_heads6_parameters155398": True,
        "suffix_exactly_10_through_15": SUFFIX == tuple(range(10, 16)),
        "frozen_parent82_endpoint": PARENT_FOLD0_ENDPOINT,
        "frozen_parent82_oracle": PARENT_FOLD0_ORACLE,
        "frozen_eight_percent_threshold": ENDPOINT_THRESHOLD,
        "claim_absent": not CLAIM.exists(),
        "result_absent": not RESULT.exists(),
        "models_absent": not MODELS.exists(),
        "failure_absent": not FAILURE.exists(),
        "pretarget_terminal_closure_absent": not PRETARGET_CLOSURE.exists(),
        "target_values_opened_or_decoded": 0,
        "target_blind_cache_rows_opened": 0,
        "training_or_scoring_runs": 0,
        "derived_feature_caches_created": 0,
    }
    try:
        validate_release(release_path)
        checks["conditional_release_and_two_independent_audits"] = "PASS"
    except Exception as error:
        checks["conditional_release_and_two_independent_audits"] = (
            f"HOLD:{type(error).__name__}:{error}"
        )
    ready = (
        checks["suffix_exactly_10_through_15"] is True
        and checks["claim_absent"] is True
        and checks["result_absent"] is True
        and checks["models_absent"] is True
        and checks["failure_absent"] is True
        and checks["pretarget_terminal_closure_absent"] is True
        and checks["conditional_release_and_two_independent_audits"] == "PASS"
    )
    print(
        json.dumps(
            {
                "schema": "owned-l3-l3square-fold0-preflight-v1",
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
    parser.add_argument("--release", type=Path, default=RELEASE)
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight(arguments.release.resolve())
        return
    try:
        execute(arguments.release.resolve())
    except BaseException as error:
        record_failure_if_owned(error)
        raise


if __name__ == "__main__":
    main()
