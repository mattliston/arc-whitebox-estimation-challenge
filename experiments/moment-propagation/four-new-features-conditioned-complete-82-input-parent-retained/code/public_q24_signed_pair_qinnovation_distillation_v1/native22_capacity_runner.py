#!/Users/matt/anaconda3/bin/python3
"""One-shot fold-pure native22 late-q capacity discriminator.

Preflight is target-free.  Execution requires a distinct, independently bound
release; the checked-in scaffold is intentionally not such a release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREREG = HERE / "NATIVE22_CAPACITY_PREREGISTRATION_V1.json"
PACKAGE_STATIC = HERE / "STATIC_VERIFICATION_V1.json"
BUNDLE = HERE / "NATIVE22_CAPACITY_OUTPUT_V1"
RESULT = BUNDLE / "RESULT.json"
MODELS = BUNDLE / "MODELS.pt"
CLAIM = HERE / "NATIVE22_CAPACITY_RUN_CLAIM_V1.json"
FAILURE = HERE / "NATIVE22_CAPACITY_TERMINAL_FAILURE_V1.json"

J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
QREF_SOURCE = P2 / "public_q24_qhead_full688_refit_v1/run_refit.py"
RAW_STATS = P2 / "public_full688_dynamic_multistate_residual_cv_v1/cache/raw_sufficient_stats.npz"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
FEATURE_MANIFEST = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/manifest.json"
)

J_SOURCE_SHA256 = "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810"
QREF_SOURCE_SHA256 = "7ffaae1a3d03a7d1f3fcfe3e9026d1674892c52291f7899978ff379f8da78cd2"
RAW_STATS_SHA256 = "1648a140768cdc254a7bb6957dcb8e82f4271118bc64b552bb08de294ec62208"
TARGETS_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
NATIVE_AGGREGATE_SHA256 = "204097b1e43387fdc2080eb690606d7dc34aeda7ca4794cb98382137ebfa3db3"
FOLD_ASSIGNMENT_SHA256 = "0f8619f00185aa5413e06c2b9aad8c1f2a6743ec8c628994bbbc9ca25fec047c"

ROWS = 688
WIDTH = 1024
DEPTH = 16
FOLDS = 4
FIT = 516
HELD = 172
SUFFIX = tuple(range(10, 16))
INPUTS = 22
MODEL_SEED = 1_362_178_069
EPOCHS = 13
NETWORK_BATCH = 4
LEARNING_RATE = 1.2e-3
MIN_LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
LOCAL_GATE = 1.375544984492742e-9
ENDPOINT_GATE = 1.2427319908150922e-8
MIN_WIN_FRACTION = 0.60
MIN_Z = 3.0
MAX_RSS_BYTES = 8 * (1 << 30)
ACTIVE_CLAIM_NONCE: str | None = None


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
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def claim_owned_by(path: Path, nonce: str | None) -> bool:
    if nonce is None or not path.is_file():
        return False
    try:
        claim = read_json(path)
    except Exception:
        return False
    return claim.get("owner_nonce") == nonce and claim.get("owner_pid") == os.getpid()


def record_terminal_failure_if_owned(error: BaseException) -> bool:
    """Commit a terminal failure only for this process's successfully acquired claim."""
    if (
        not claim_owned_by(CLAIM, ACTIVE_CLAIM_NONCE)
        or BUNDLE.exists()
        or FAILURE.exists()
    ):
        return False
    failure = {
        "schema": "public-q24-native22-high-capacity-terminal-failure-v1",
        "status": "TERMINAL_POST_CLAIM_FAILURE__NO_RETRY",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "claim_sha256": sha256(CLAIM),
        "partial_output_bundle_present": BUNDLE.exists(),
        "decision": "NO_RETRY_NO_TARGET_REOPEN__ATTRIBUTE_FAILURE_BEFORE_ANY_SUCCESSOR",
    }
    write_exclusive_durable(FAILURE, failure)
    return True


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


def observe_rss(_where: str) -> int:
    value = resident_bytes()
    return value


def call_with_observational_resource_guard(
    module: Any, guard_name: str, operation: Any
) -> Any:
    """Run one accuracy operation with an inherited resource guard made observational."""
    original_guard = getattr(module, guard_name)
    if not callable(original_guard) or not callable(operation):
        raise TypeError(f"non-callable resource guard or operation: {guard_name}")
    setattr(module, guard_name, observe_rss)
    try:
        return operation()
    finally:
        setattr(module, guard_name, original_guard)


def verify_frozen_inputs(*, include_target_bundle: bool) -> None:
    expected = {
        J_SOURCE: J_SOURCE_SHA256,
        QREF_SOURCE: QREF_SOURCE_SHA256,
        RAW_STATS: RAW_STATS_SHA256,
        FEATURE_MANIFEST: FEATURE_MANIFEST_SHA256,
    }
    if include_target_bundle:
        expected[TARGETS] = TARGETS_SHA256
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise AssertionError(f"frozen input changed: {path}")


def validate_release(path: Path) -> dict[str, Any]:
    release = read_json(path)
    if (
        release.get("schema") != "public-q24-native22-high-capacity-execution-release-v1"
        or release.get("status") != "RELEASE_EXACTLY_ONE_NATIVE22_FOUR_FOLD_CAPACITY_RUN"
        or release.get("role") != "observer_fit"
        or release.get("columns") != ["mlp_id", "all_layer_means"]
        or release.get("observer_selection_targets") is not False
        or release.get("names") is not False
        or release.get("scores") is not False
        or release.get("board") is not False
        or release.get("network") is not False
        or release.get("submission") is not False
    ):
        raise PermissionError("no narrow native22 execution release")
    bindings = release.get("bindings", {})
    expected = {
        "native22_preregistration_sha256": sha256(PREREG),
        "native22_runner_sha256": sha256(Path(__file__)),
        "package_static_verification_sha256": sha256(PACKAGE_STATIC),
        "target_bundle_sha256": TARGETS_SHA256,
        "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
        "fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise PermissionError(f"release binding changed: {key}")
    review_path_text = release.get("independent_review_path")
    review_hash = bindings.get("independent_review_sha256")
    if not isinstance(review_path_text, str) or not isinstance(review_hash, str):
        raise PermissionError("independent review binding missing")
    review_path = Path(review_path_text)
    if not review_path.is_absolute():
        review_path = path.parent / review_path
    review = read_json(review_path.resolve())
    if (
        sha256(review_path.resolve()) != review_hash
        or review.get("status") != "GO_NATIVE22_EXACTLY_ONE_OOF_RUN"
        or review.get("preregistration_sha256") != sha256(PREREG)
        or review.get("runner_sha256") != sha256(Path(__file__))
    ):
        raise PermissionError("independent review is not a GO for these exact bytes")
    return release


class QHead(nn.Module):
    def __init__(
        self,
        hidden: int,
        hidden_layers: int,
        independent_trunks: bool,
        centers: torch.Tensor,
        scales: torch.Tensor,
        layer_rms: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("centers", centers.detach().clone())
        self.register_buffer("scales", scales.detach().clone())
        self.register_buffer("layer_rms", layer_rms.detach().clone())
        def make_trunk() -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(INPUTS, hidden), nn.SiLU()]
            for _ in range(hidden_layers - 1):
                layers.extend((nn.Linear(hidden, hidden), nn.SiLU()))
            return nn.Sequential(*layers)

        self.independent_trunks = bool(independent_trunks)
        if self.independent_trunks:
            self.trunks = nn.ModuleList([make_trunk() for _ in SUFFIX])
        else:
            self.trunk = make_trunk()
        self.head_weight = nn.Parameter(torch.zeros(len(SUFFIX), hidden))
        self.head_bias = nn.Parameter(torch.zeros(len(SUFFIX)))

    def source(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        base_state: torch.Tensor,
        base_q: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        tail = torch.clamp(
            (raw[..., 1:] - self.centers[layer]) / self.scales[layer], -8.0, 8.0
        )
        rms = self.layer_rms[layer]
        features = torch.cat(
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
        slot = SUFFIX.index(layer)
        hidden = self.trunks[slot](features) if self.independent_trunks else self.trunk(features)
        standard = torch.einsum("...h,h->...", hidden, self.head_weight[slot])
        return sigma * rms * (standard + self.head_bias[slot])


ARMS = {
    "control64": {
        "hidden": 64,
        "hidden_layers": 2,
        "independent_trunks": False,
        "parameters": 6_022,
    },
    "shared256": {
        "hidden": 256,
        "hidden_layers": 3,
        "independent_trunks": False,
        "parameters": 139_014,
    },
    "independent256": {
        "hidden": 256,
        "hidden_layers": 3,
        "independent_trunks": True,
        "parameters": 826_374,
    },
}


def canonical_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for _name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        if value.dtype != np.float32:
            raise AssertionError("model state is not float32")
        digest.update(np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def make_model(arm: str, base) -> tuple[QHead, str]:
    recipe = ARMS[arm]
    torch.manual_seed(MODEL_SEED)
    model = QHead(
        int(recipe["hidden"]),
        int(recipe["hidden_layers"]),
        bool(recipe["independent_trunks"]),
        base.centers,
        base.scales,
        base.layer_rms,
    )
    with torch.no_grad():
        model.head_weight.zero_()
        model.head_bias.zero_()
    count = sum(value.numel() for value in model.parameters())
    if count != int(recipe["parameters"]):
        raise AssertionError(f"{arm} parameter count changed: {count}")
    return model, canonical_model_hash(model)


def q_rms_for(prepared, fit: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sqrt(np.mean(prepared.delta_q[fit].astype(np.float64) ** 2, axis=(0, 2))),
        2.0**-20,
    ).astype(np.float32)


def train_arm(
    arm: str,
    base,
    prepared,
    fit: np.ndarray,
    fold: int,
) -> tuple[QHead, dict[str, Any]]:
    model, initial_hash = make_model(arm, base)
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
        optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=MIN_LEARNING_RATE
    )
    order_seed = (MODEL_SEED ^ 0xA5A5A5A5 ^ (fold * 0x9E3779B1)) & 0xFFFF_FFFF
    order_rng = np.random.Generator(np.random.PCG64(order_seed))
    order_digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        order = order_rng.permutation(fit).astype(np.int32, copy=False)
        order_digest.update(np.ascontiguousarray(order.astype("<i4", copy=False)).tobytes())
        model.train()
        objective = 0.0
        for offset in range(0, len(order), NETWORK_BATCH):
            ranks = order[offset : offset + NETWORK_BATCH]
            if len(ranks) != NETWORK_BATCH:
                raise AssertionError("fit fold no longer divides into batches of four")
            losses = []
            for slot, layer in enumerate(SUFFIX):
                raw = torch.from_numpy(
                    np.ascontiguousarray(prepared.raw_suffix[ranks, slot].reshape(-1, 20))
                )
                sigma = torch.from_numpy(
                    np.ascontiguousarray(prepared.sigma_suffix[ranks, slot].reshape(-1))
                )
                state = torch.from_numpy(
                    np.ascontiguousarray(prepared.base_state[ranks, slot].reshape(-1))
                )
                base_q = torch.from_numpy(
                    np.ascontiguousarray(prepared.base_q[ranks, slot].reshape(-1))
                )
                truth_q = torch.from_numpy(
                    np.ascontiguousarray(prepared.delta_q[ranks, slot].reshape(-1))
                )
                source = model.source(raw, sigma, state, base_q, layer)
                losses.append(torch.mean(((source - truth_q) / float(q_rms[slot])).square()))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite {arm} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective += float(loss.detach()) * len(ranks)
        trace.append(
            {
                "epoch": epoch,
                "full_node_normalized_q_loss": objective / len(fit),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "maximum_process_rss_bytes_observational_only": observe_rss(
                    f"{arm} fold {fold} epoch {epoch}"
                ),
            }
        )
    model.eval()
    return model, {
        "initial_state_sha256": initial_hash,
        "row_order_sha256": order_digest.hexdigest(),
        "q_rms": q_rms.tolist(),
        "trace": trace,
    }


def score_arm(model: QHead, prepared, dataset, held: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    local_sse = np.zeros(len(SUFFIX), dtype=np.float64)
    local_count = np.zeros(len(SUFFIX), dtype=np.int64)
    transported_error_energy = np.zeros(len(SUFFIX), dtype=np.float64)
    with torch.no_grad():
        for rank_value in held:
            rank = int(rank_value)
            weights = np.load(dataset.load(rank).weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(WIDTH, dtype=np.float32)
            oracle = np.zeros(WIDTH, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                raw = torch.from_numpy(np.ascontiguousarray(prepared.raw_suffix[rank, slot]))
                sigma = torch.from_numpy(np.ascontiguousarray(prepared.sigma_suffix[rank, slot]))
                base_state = torch.from_numpy(np.ascontiguousarray(prepared.base_state[rank, slot]))
                base_q = torch.from_numpy(np.ascontiguousarray(prepared.base_q[rank, slot]))
                source = model.source(raw, sigma, base_state, base_q, layer).cpu().numpy()
                truth_q = prepared.delta_q[rank, slot]
                q_error = source.astype(np.float64) - truth_q.astype(np.float64)
                local_sse[slot] += float(np.dot(q_error, q_error))
                local_count[slot] += WIDTH
                phi = prepared.raw_suffix[rank, slot, :, 5].astype(np.float64)
                state = (
                    phi * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                    + source.astype(np.float64)
                ).astype(np.float32)
                oracle = (
                    phi * (oracle.astype(np.float64) @ weights[layer].astype(np.float64))
                    + truth_q.astype(np.float64)
                ).astype(np.float32)
                transported_error = state.astype(np.float64) - oracle.astype(np.float64)
                transported_error_energy[slot] += float(np.dot(transported_error, transported_error))
            residual = prepared.residual_final[rank].astype(np.float64)
            base_mse = float(np.mean(residual * residual))
            candidate_mse = float(np.mean((residual - state.astype(np.float64)) ** 2))
            oracle_mse = float(np.mean((residual - oracle.astype(np.float64)) ** 2))
            rows.append(
                {
                    "role_rank": rank,
                    "base_foldJ_mse": base_mse,
                    "candidate_mse": candidate_mse,
                    "exact_q_oracle_mse": oracle_mse,
                }
            )
    per_layer_local = (local_sse / local_count).tolist()
    return {
        "rows": rows,
        "local_q_mse": float(np.sum(local_sse) / np.sum(local_count)),
        "per_layer_local_q_mse": per_layer_local,
        "per_layer_transported_q_error_mse": (
            transported_error_energy / (len(held) * WIDTH)
        ).tolist(),
        "endpoint_mse": float(np.mean([row["candidate_mse"] for row in rows])),
        "base_mse": float(np.mean([row["base_foldJ_mse"] for row in rows])),
        "oracle_mse": float(np.mean([row["exact_q_oracle_mse"] for row in rows])),
    }


def paired_z(improvement: np.ndarray) -> float:
    standard_error = float(np.std(improvement, ddof=1) / math.sqrt(len(improvement)))
    if standard_error == 0.0:
        return float("inf") if float(np.mean(improvement)) > 0.0 else 0.0
    return float(np.mean(improvement) / standard_error)


def execute(release_path: Path) -> None:
    global ACTIVE_CLAIM_NONCE
    if BUNDLE.exists() or CLAIM.exists() or FAILURE.exists():
        raise FileExistsError("native22 one-shot run already claimed or completed")
    verify_frozen_inputs(include_target_bundle=False)
    release = validate_release(release_path)
    # The narrow release authorizes a byte-level rehash, still before semantic
    # decoding.  Preflight never opens the target bundle.
    verify_frozen_inputs(include_target_bundle=True)

    # Importing the bound q reference imports pyarrow but decodes no target.
    # Target-free native receipts and folds are verified before the claim.
    j = load_module("native22_bound_fold_j", J_SOURCE)
    qref = load_module("native22_bound_q_reference", QREF_SOURCE)
    records, native_storage = qref.verify_native_rows()
    folds = qref.balanced_folds(records)
    if qref.NATIVE_AGGREGATE_SHA256 != NATIVE_AGGREGATE_SHA256:
        raise AssertionError("native aggregate literal changed")
    if hashlib.sha256(folds.tobytes()).hexdigest() != FOLD_ASSIGNMENT_SHA256:
        raise AssertionError("fold vector changed")
    dataset = qref.All688Dataset(records)

    owner_nonce = hashlib.sha256(
        os.getpid().to_bytes(8, "big", signed=False) + os.urandom(32)
    ).hexdigest()
    claim = {
        "schema": "public-q24-native22-high-capacity-run-claim-v1",
        "status": "CLAIMED_AFTER_TARGET_FREE_ROW_VERIFICATION_BEFORE_TARGET_DECODE__NO_RETRY",
        "bindings": {
            "release_sha256": sha256(release_path),
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "j_source_sha256": J_SOURCE_SHA256,
            "qref_source_sha256": QREF_SOURCE_SHA256,
            "target_bundle_sha256": TARGETS_SHA256,
            "native_row_aggregate_sha256": NATIVE_AGGREGATE_SHA256,
            "fold_assignment_uint8_sha256": FOLD_ASSIGNMENT_SHA256,
        },
        "native_storage_bytes_verified": native_storage,
        "target_values_decoded_before_claim": 0,
        "owner_pid": os.getpid(),
        "owner_nonce": owner_nonce,
    }
    write_exclusive_durable(CLAIM, claim)
    ACTIVE_CLAIM_NONCE = owner_nonce

    targets = qref.load_targets(records)
    hosts = np.ascontiguousarray(
        np.stack([np.load(qref.row_dir(rank) / "host.npy", allow_pickle=False) for rank in range(ROWS)]),
        dtype=np.float32,
    )
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)
    stats = np.load(RAW_STATS, allow_pickle=False)
    sums = stats["sum"]
    sums2 = stats["sum2"]
    if sums.dtype != np.float64 or sums.shape != (ROWS, DEPTH, 19):
        raise AssertionError("raw sum schema changed")
    if sums2.dtype != np.float64 or sums2.shape != sums.shape:
        raise AssertionError("raw sum2 schema changed")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    fold_results: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    saved: dict[str, Any] = {
        "schema": "public-q24-native22-high-capacity-oof-models-v1",
        "folds": {},
    }
    for fold in range(FOLDS):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if len(fit) != FIT or len(held) != HELD:
            raise AssertionError("balanced fold size changed")
        centers, scales, layer_rms = j.moments_for(fit, error, sums, sums2)
        torch.manual_seed(j.SEED + fold)
        fold_j = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
        # Both bound sources contain deployment-oriented RSS guards.  This
        # accuracy discriminator records RSS but cannot reject science on it.
        j_trace, j_max_rss = call_with_observational_resource_guard(
            j,
            "enforce_resources",
            lambda: j.fit_fold(fold, fold_j, dataset, fit, error),
        )
        fold_j.eval()
        prepared = call_with_observational_resource_guard(
            qref,
            "enforce_rss",
            lambda: qref.prepare_all(j, fold_j, dataset, error),
        )
        arm_scores: dict[str, Any] = {}
        arm_payloads: dict[str, Any] = {}
        order_hashes: list[str] = []
        for arm in ARMS:
            model, training = train_arm(arm, fold_j, prepared, fit, fold)
            score = score_arm(model, prepared, dataset, held)
            order_hashes.append(str(training["row_order_sha256"]))
            arm_scores[arm] = {key: value for key, value in score.items() if key != "rows"}
            all_rows[arm].extend(score["rows"])
            arm_payloads[arm] = {
                "state_dict": copy.deepcopy(model.state_dict()),
                "training": training,
            }
        if len(set(order_hashes)) != 1:
            raise AssertionError("control/candidate row order mismatch")
        candidate = arm_scores["independent256"]
        control = arm_scores["control64"]
        fold_results.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "held_rows": len(held),
                "j_maximum_process_rss_bytes": j_max_rss,
                "j_trace": j_trace,
                "row_order_sha256": order_hashes[0],
                "control64": control,
                "shared256": arm_scores["shared256"],
                "independent256": candidate,
                "candidate_beats_control_local": candidate["local_q_mse"] < control["local_q_mse"],
                "candidate_beats_control_endpoint": candidate["endpoint_mse"] < control["endpoint_mse"],
                "candidate_beats_base_endpoint": candidate["endpoint_mse"] < candidate["base_mse"],
            }
        )
        saved["folds"][str(fold)] = {
            "j_state_dict": copy.deepcopy(fold_j.state_dict()),
            "j_centers": centers,
            "j_scales": scales,
            "j_layer_rms": layer_rms,
            "arms": arm_payloads,
        }
        del prepared, fold_j
        observe_rss(f"fold {fold} complete")

    for arm in ARMS:
        all_rows[arm].sort(key=lambda row: int(row["role_rank"]))
        if [int(row["role_rank"]) for row in all_rows[arm]] != list(range(ROWS)):
            raise AssertionError(f"{arm} OOF coverage changed")
    candidate_mse = np.asarray(
        [float(row["candidate_mse"]) for row in all_rows["independent256"]], dtype=np.float64
    )
    control_mse = np.asarray(
        [float(row["candidate_mse"]) for row in all_rows["control64"]], dtype=np.float64
    )
    base_mse = np.asarray(
        [float(row["base_foldJ_mse"]) for row in all_rows["independent256"]], dtype=np.float64
    )
    oracle_mse = np.asarray(
        [float(row["exact_q_oracle_mse"]) for row in all_rows["independent256"]], dtype=np.float64
    )
    candidate_local = float(
        np.mean([float(row["independent256"]["local_q_mse"]) for row in fold_results])
    )
    control_local = float(
        np.mean([float(row["control64"]["local_q_mse"]) for row in fold_results])
    )
    improvement = base_mse - candidate_mse
    win_fraction = float(np.mean(improvement > 0.0))
    z_value = paired_z(improvement)
    gates = {
        "candidate_local_q_mse_at_most_1_375544984492742e_9": candidate_local <= LOCAL_GATE,
        "candidate_endpoint_raw_strictly_below_1_2427319908150922e_8": float(np.mean(candidate_mse)) < ENDPOINT_GATE,
        "candidate_beats_control_local_all_folds": all(
            bool(row["candidate_beats_control_local"]) for row in fold_results
        ),
        "candidate_beats_control_endpoint_all_folds": all(
            bool(row["candidate_beats_control_endpoint"]) for row in fold_results
        ),
        "candidate_beats_foldJ_base_all_folds": all(
            bool(row["candidate_beats_base_endpoint"]) for row in fold_results
        ),
        "network_win_fraction_at_least_0_60": win_fraction >= MIN_WIN_FRACTION,
        "paired_network_z_at_least_3": z_value >= MIN_Z,
        "all_endpoint_metrics_finite": bool(
            np.all(np.isfinite(np.concatenate((candidate_mse, control_mse, base_mse, oracle_mse))))
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "public-q24-native22-high-capacity-ceiling-result-v1",
        "status": "PASS_NATIVE22_CAPACITY__NO_SUBMISSION_AUTHORITY" if passed else "FAIL_NATIVE22_FROZEN_CAPACITY__NO_RETRY",
        "access_declarations": {
            "role": "observer_fit",
            "target_columns_opened": ["mlp_id", "all_layer_means"],
            "observer_selection_targets_opened": [],
            "names_opened": [],
            "scores_opened": [],
            "board_opened": False,
            "network": False,
            "submission": False,
        },
        "bindings": {
            "claim_sha256": sha256(CLAIM),
            "release_sha256": sha256(release_path),
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "target_bundle_sha256": TARGETS_SHA256,
            "oof_models_sha256": None,
        },
        "aggregate": {
            "networks": ROWS,
            "independent256_local_q_mse": candidate_local,
            "shared256_local_q_mse": float(
                np.mean([float(row["shared256"]["local_q_mse"]) for row in fold_results])
            ),
            "control64_local_q_mse": control_local,
            "independent256_endpoint_mse": float(np.mean(candidate_mse)),
            "shared256_endpoint_mse": float(
                np.mean([float(row["shared256"]["endpoint_mse"]) for row in fold_results])
            ),
            "control64_endpoint_mse": float(np.mean(control_mse)),
            "foldJ_base_endpoint_mse": float(np.mean(base_mse)),
            "exact_q_oracle_endpoint_mse": float(np.mean(oracle_mse)),
            "network_win_fraction_vs_foldJ_base": win_fraction,
            "paired_network_z_vs_foldJ_base": z_value,
        },
        "fold_results": fold_results,
        "gates": gates,
        "decision": (
            "GO_NATIVE22_DIRECT_Q_INTEGRATION_CHILD__PAIR_SOURCE_NOT_YET_NEEDED__NO_SUBMISSION"
            if passed
            else "CLOSE_CURRENT_NATIVE22_CAPACITY_SPELLING__PRIORITIZE_SIGNED_PAIR_Q1__NO_RETRY"
        ),
        "per_network": all_rows,
    }
    temporary_bundle = HERE / f".{BUNDLE.name}.{os.getpid()}.tmp"
    if temporary_bundle.exists() or BUNDLE.exists():
        raise FileExistsError("output bundle collision")
    temporary_bundle.mkdir(mode=0o700)
    temporary_models = temporary_bundle / "MODELS.pt"
    temporary_result = temporary_bundle / "RESULT.json"
    torch.save(saved, temporary_models)
    with temporary_models.open("rb") as handle:
        os.fsync(handle.fileno())
    result["bindings"]["oof_models_sha256"] = sha256(temporary_models)
    temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary_result.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary_directory = os.open(temporary_bundle, os.O_RDONLY)
    try:
        os.fsync(temporary_directory)
    finally:
        os.close(temporary_directory)
    temporary_bundle.replace(BUNDLE)
    parent_directory = os.open(HERE, os.O_RDONLY)
    try:
        os.fsync(parent_directory)
    finally:
        os.close(parent_directory)
    print(json.dumps({"status": result["status"], **result["aggregate"]}, sort_keys=True))


def preflight(release_path: Path | None) -> None:
    verify_frozen_inputs(include_target_bundle=False)
    checks: dict[str, Any] = {
        "runtime_python": ".".join(map(str, sys.version_info[:3])),
        "runtime_numpy": np.__version__,
        "runtime_torch": torch.__version__,
        "runtime_exact": (
            sys.version_info[:3] == (3, 11, 5)
            and np.__version__ == "1.26.4"
            and torch.__version__ == "2.7.0"
        ),
        "preregistration_present": PREREG.is_file(),
        "package_static_verification_present": PACKAGE_STATIC.is_file(),
        "atomic_output_bundle_absent": not BUNDLE.exists(),
        "claim_absent": not CLAIM.exists(),
        "terminal_failure_absent": not FAILURE.exists(),
        "release_supplied": release_path is not None,
        "target_values_decoded": 0,
        "observer_selection_targets_opened": 0,
        "names_scores_board_opened": 0,
    }
    if release_path is not None:
        try:
            validate_release(release_path.resolve())
            checks["release_validation"] = "PASS"
        except Exception as error:
            checks["release_validation"] = f"HOLD:{type(error).__name__}:{error}"
    ready = all(
        checks.get(key) is True
        for key in (
            "runtime_exact",
            "preregistration_present",
            "package_static_verification_present",
            "atomic_output_bundle_absent",
            "claim_absent",
            "terminal_failure_absent",
            "release_supplied",
        )
    ) and checks.get("release_validation") == "PASS"
    print(
        json.dumps(
            {
                "schema": "public-q24-native22-high-capacity-preflight-v1",
                "status": "READY_FOR_ONE_SHOT_EXECUTION" if ready else "HOLD_NO_EXACT_RELEASE",
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
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.preflight:
        preflight(args.release)
        return
    if args.release is None:
        raise SystemExit("--execute requires --release")
    try:
        execute(args.release.resolve())
    except BaseException as error:
        record_terminal_failure_if_owned(error)
        raise


if __name__ == "__main__":
    main()
