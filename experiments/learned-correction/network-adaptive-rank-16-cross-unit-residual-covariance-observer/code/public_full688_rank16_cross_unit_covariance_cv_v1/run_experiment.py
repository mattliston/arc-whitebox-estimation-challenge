#!/Users/matt/anaconda3/bin/python3
"""Frozen P2R0F2N rank-16 cross-unit covariance observer CV."""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
M_DIR = P2 / "public_full688_backward_adjoint_state_cv_v1"
NATIVE_DIR = P2 / "public_q24_native_transport_energy_pilot_v1"
LOADER_DIR = P2 / "public_full688_local_innovation_transport_v1"
D_DIR = P2 / "public_full688_dynamic_multistate_residual_cv_v1"
FEATURE_STORE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
TARGET_ROOT = P2 / "data_gate/sealed_role_target_bundle"
sys.path.insert(0, str(LOADER_DIR))
from dataset import DEPTH, LOCAL_FEATURES, WIDTH, PublicResidualDataset  # noqa: E402

J_SPEC = importlib.util.spec_from_file_location(
    "p2r0f2j_frozen_runner", J_DIR / "run_experiment.py"
)
if J_SPEC is None or J_SPEC.loader is None:
    raise ImportError("cannot load frozen J runner")
J = importlib.util.module_from_spec(J_SPEC)
J_SPEC.loader.exec_module(J)


PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
SMOKE_RECEIPT = HERE / "SMOKE_RECEIPT.json"
TRAINING_RELEASE = HERE / "TRAINING_RELEASE.json"
FOLD_DIR = HERE / "folds"
CV_RESULT = HERE / "CV_RESULTS.json"
SHARED_CACHE_MANIFEST = D_DIR / "cache/MANIFEST.json"
SHARED_ERROR = D_DIR / "cache/patched_error.npy"
SHARED_STATS = D_DIR / "cache/raw_sufficient_stats.npz"

NODE = "P2R0F2N"
SEED = 1_753_358_438
FOLDS = 4
INNER_FOLDS = 3
RANK = 16
HIDDEN = 128
INPUT_FEATURES = 36
HEAD_OUTPUTS = 34
FIRST_CORRECTED_LAYER = 2
CORRECTED_LAYERS = 14
TRANSITIONS = 13
THREADS = 2
BATCH_SIZE = 3
EPOCHS = 8
LEARNING_RATE = 8.0e-4
MIN_LEARNING_RATE = 8.0e-5
WEIGHT_DECAY = 2.0e-5
GRADIENT_CLIP = 5.0
GAUSSIAN_WEIGHT = 0.05
LOG_SCALE_PENALTY = 1.0e-5
GRAM_FLOOR = 2.0**-10
POSITIVE_OFFSET = 2.0**-12
POSITIVE_MINIMUM = 2.0**-10
POSITIVE_MAXIMUM = 2.0**6
COHERENCE_FLOOR = 1.0e-4
OOF_GATE = 1.0e-8
FOLD_GATE = 1.1e-8
MAX_NETWORK_GATE = 1.0e-7
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
J_CORRECTION_FLOPS = 1_051_656_192
N_ADDED_FLOPS = 1_226_926_080
CORRECTION_FLOPS = 2_278_582_272
NATIVE_HOST_TAPE_FLOPS = 2_131_692_137_721
TOTAL_FLOPS = 2_133_970_719_993
EXACT_SCORING_BUDGET = 1 << 41
PHASE2_BUDGET = 2_200_000_000_000
EXPECTED_FOLD_SIZES = [147, 172, 171, 198]
TAIL_RANKS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
_ROW_MANIFESTS_VERIFIED = False


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


def fold_of(mlp_id: int) -> int:
    payload = f"{SEED}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % FOLDS


def inner_seed(outer_fold: int) -> int:
    return SEED ^ ((outer_fold + 1) * 0x9E3779B1)


def inner_fold_of(outer_fold: int, mlp_id: int) -> int:
    payload = f"{inner_seed(outer_fold)}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % INNER_FOLDS


def verify_seal() -> dict[str, object]:
    if not SEAL.is_file():
        raise FileNotFoundError("PREREGISTRATION_SEAL.json is required")
    seal = read_json(SEAL)
    if seal.get("status") != "frozen-before-any-n-metrics":
        raise AssertionError("seal status changed")
    bindings = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "loader_sha256": sha256(LOADER_DIR / "dataset.py"),
        "shared_cache_manifest_sha256": sha256(SHARED_CACHE_MANIFEST),
        "shared_error_sha256": sha256(SHARED_ERROR),
        "shared_stats_sha256": sha256(SHARED_STATS),
        "j_program_sha256": sha256(J_DIR / "run_experiment.py"),
        "j_cv_results_sha256": sha256(J_DIR / "CV_RESULTS.json"),
        "j_verification_sha256": sha256(J_DIR / "VERIFICATION.json"),
        "m_decomposition_sha256": sha256(
            M_DIR / "M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json"
        ),
        "native_transfer_result_sha256": sha256(NATIVE_DIR / "PILOT_RESULT.json"),
        "native_transfer_model_sha256": sha256(NATIVE_DIR / "PILOT_MODEL.pt"),
        "feature_store_manifest_sha256": sha256(FEATURE_STORE_ROOT / "manifest.json"),
        "feature_store_finalization_sha256": sha256(
            FEATURE_STORE_ROOT / "FINALIZATION_RECEIPT_V3.json"
        ),
        "target_bundle_manifest_sha256": sha256(TARGET_ROOT / "manifest.json"),
        "observer_fit_target_parquet_sha256": sha256(
            TARGET_ROOT / "observer_fit.parquet"
        ),
    }
    for key, value in bindings.items():
        if seal.get(key) != value:
            raise AssertionError(f"sealed binding changed: {key}")
    expected_models = [
        sha256(J_DIR / "folds" / f"fold_{fold}.pt") for fold in range(FOLDS)
    ]
    expected_results = [
        sha256(J_DIR / "folds" / f"fold_{fold}.json") for fold in range(FOLDS)
    ]
    if seal.get("j_outer_model_sha256") != expected_models:
        raise AssertionError("sealed J outer models changed")
    if seal.get("j_outer_result_sha256") != expected_results:
        raise AssertionError("sealed J outer results changed")
    prereg = read_json(PREREGISTRATION)
    manifest = read_json(RUN_MANIFEST)
    if prereg.get("node") != NODE or manifest.get("node") != NODE:
        raise AssertionError("node binding changed")
    if prereg.get("status") != "frozen-before-any-n-metrics":
        raise AssertionError("preregistration status changed")
    if manifest.get("status") != "authorized-by-root-design-frozen-launch-held-until-seal-and-smoke":
        raise AssertionError("run authorization changed")
    if prereg["scope"]["observer_selection_targets_allowed"] is not False:
        raise AssertionError("selection firewall changed")
    if prereg["scope"]["public_name_join_allowed"] is not False:
        raise AssertionError("name firewall changed")
    if seal["data_gate"]["observer_selection_targets_allowed"] is not False:
        raise AssertionError("seal selection firewall changed")
    return seal


def verify_release(seal: dict[str, object]) -> dict[str, object]:
    if not TRAINING_RELEASE.is_file():
        raise FileNotFoundError("TRAINING_RELEASE.json is required for fold work")
    release = read_json(TRAINING_RELEASE)
    if release.get("status") != "authorized-after-sealed-smoke-by-root":
        raise AssertionError("training release status changed")
    if release.get("root_release") is not True:
        raise AssertionError("root release is absent")
    expected = {
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "smoke_receipt_sha256": sha256(SMOKE_RECEIPT),
    }
    for key, value in expected.items():
        if release.get(key) != value:
            raise AssertionError(f"training release binding changed: {key}")
    if release.get("observer_selection_targets_opened") != 0:
        raise AssertionError("release selection firewall changed")
    smoke = read_json(SMOKE_RECEIPT)
    smoke_expected = {
        "node": NODE,
        "status": "PASS_SYNTHETIC_AND_ONE_INPUT_NO_TRUTH_METRICS",
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    for key, value in smoke_expected.items():
        if smoke.get(key) != value:
            raise AssertionError(f"smoke receipt semantic changed: {key}")
    return release


def verify_completed_fold_record(
    record: dict[str, object], fold: int, seal: dict[str, object]
) -> None:
    expected = {
        "schema": "public-full688-rank16-cross-unit-covariance-fold-v1",
        "status": "complete",
        "node": NODE,
        "fold": fold,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
        "outer_j_model_sha256": seal["j_outer_model_sha256"][fold],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"fold {fold}: completed receipt mismatch at {key}")


def verify_feature_row_manifests(dataset: PublicResidualDataset) -> None:
    global _ROW_MANIFESTS_VERIFIED
    if _ROW_MANIFESTS_VERIFIED:
        return
    manifest = read_json(FEATURE_STORE_ROOT / "manifest.json")
    if manifest.get("status") != "SEALED_TARGET_BLIND_FEATURE_STORE":
        raise AssertionError("feature store is not sealed")
    role = manifest["roles"]["observer_fit"]
    entries = role["row_manifests"]
    if len(entries) != 688:
        raise AssertionError("feature row manifest coverage changed")
    ids_digest = hashlib.sha256(
        ("\n".join(str(int(value)) for value in dataset.mlp_ids) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    if ids_digest != role["ordered_mlp_ids_sha256"]:
        raise AssertionError("ordered observer_fit mlp_id roster changed")
    for rank, entry in enumerate(entries):
        if int(entry["role_rank"]) != rank:
            raise AssertionError(f"feature manifest rank changed at {rank}")
        if int(entry["mlp_id"]) != int(dataset.mlp_ids[rank]):
            raise AssertionError(f"feature manifest mlp_id changed at {rank}")
        path = FEATURE_STORE_ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise AssertionError(f"feature row manifest hash changed at {rank}")
    _ROW_MANIFESTS_VERIFIED = True


def readiness_or_raise(dataset: PublicResidualDataset) -> dict[str, object]:
    verify_feature_row_manifests(dataset)
    readiness = dataset.readiness()
    if not readiness["ready"]:
        raise RuntimeError(f"observer_fit host is not ready: {readiness}")
    if readiness["available"] != 688 or readiness["missing"]:
        raise AssertionError("base scan coverage changed")
    if readiness["data_derived_tail_ranks"] != TAIL_RANKS:
        raise AssertionError("tail roster changed")
    if readiness["repaired_tail_ranks"] != TAIL_RANKS:
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
    sizes = [int(np.sum(folds == fold)) for fold in range(FOLDS)]
    if sizes != EXPECTED_FOLD_SIZES:
        raise AssertionError(f"fold assignment changed: {sizes}")
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


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


class Rank16CrossUnitCovarianceObserver(nn.Module):
    """Dynamic whitened covariance factors applied to a learned local score."""

    def __init__(self, centers: np.ndarray, scales: np.ndarray, layer_rms: np.ndarray):
        super().__init__()
        self.register_buffer("centers", torch.from_numpy(centers.copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("layer_rms", torch.from_numpy(layer_rms.copy()))
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(
            torch.empty(CORRECTED_LAYERS, HIDDEN, HEAD_OUTPUTS)
        )
        self.head_bias = nn.Parameter(torch.zeros(CORRECTED_LAYERS, HEAD_OUTPUTS))
        self.factor_gain_delta = nn.Parameter(torch.zeros(TRANSITIONS, RANK))
        self.factor_mix_delta = nn.Parameter(torch.zeros(TRANSITIONS, RANK, RANK))
        nn.init.normal_(self.head_weight, mean=0.0, std=2.0e-3)

    def local_features(
        self,
        raw_layer: torch.Tensor,
        sigma_layer: torch.Tensor,
        messages: torch.Tensor,
        energies: torch.Tensor,
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
        return torch.cat(
            (raw_layer[..., :1], tail, message_rms, message_sigma, log_energy, coherence),
            dim=-1,
        )

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        j_states: torch.Tensor,
        j_energies: torch.Tensor,
        *,
        return_covariance: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        batch = raw.shape[0]
        factors = torch.zeros(batch, WIDTH, RANK, dtype=raw.dtype, device=raw.device)
        outputs = [
            torch.zeros(batch, WIDTH, dtype=raw.dtype, device=raw.device)
            for _ in range(FIRST_CORRECTED_LAYER)
        ]
        all_d: list[torch.Tensor] = []
        all_lambda: list[torch.Tensor] = []
        all_u: list[torch.Tensor] = []
        scale_penalties: list[torch.Tensor] = []
        identity = torch.eye(RANK, dtype=raw.dtype, device=raw.device)
        for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
            local_layer = layer - FIRST_CORRECTED_LAYER
            if layer == FIRST_CORRECTED_LAYER:
                messages = torch.zeros(
                    batch, WIDTH, J.CHANNELS, dtype=raw.dtype, device=raw.device
                )
                transported_factors = torch.zeros_like(factors)
            else:
                layer_weights = weights[:, layer]
                messages = torch.bmm(
                    j_states[:, layer - 1].transpose(1, 2), layer_weights
                ).transpose(1, 2)
                messages = raw[:, layer, :, 5, None] * messages
                transported_factors = torch.bmm(
                    factors.transpose(1, 2), layer_weights
                ).transpose(1, 2)
                transported_factors = (
                    raw[:, layer, :, 5, None] * transported_factors
                )
            features = self.local_features(
                raw[:, layer], sigma[:, layer], messages, j_energies[:, layer], layer
            )
            hidden = self.encoder(features)
            head = torch.einsum(
                "bwh,ho->bwo", hidden, self.head_weight[local_layer]
            ) + self.head_bias[local_layer]
            source = torch.tanh(head[..., :RANK])
            log_lambda_local = head[..., RANK : 2 * RANK]
            log_diagonal = head[..., 2 * RANK]
            score = torch.tanh(head[..., 2 * RANK + 1])
            if layer == FIRST_CORRECTED_LAYER:
                candidate = source
            else:
                transition = layer - FIRST_CORRECTED_LAYER - 1
                gain = 1.0 + 0.25 * torch.tanh(self.factor_gain_delta[transition])
                mixing = identity + 0.10 * torch.tanh(
                    self.factor_mix_delta[transition]
                )
                candidate = torch.einsum(
                    "bwr,rs->bws", gain * transported_factors, mixing
                ) + source
            gram = torch.bmm(candidate.transpose(1, 2), candidate) / float(WIDTH)
            gram = gram + GRAM_FLOOR * identity[None]
            # The forward whitening is exact.  Stop-gradient on its eigensystem
            # avoids undefined eigenvector derivatives when Gram eigenvalues tie;
            # gradients still flow through factors=candidate@inverse_sqrt.
            with torch.no_grad():
                eigenvalues, eigenvectors = torch.linalg.eigh(gram.detach())
                eigenvalues = torch.clamp(eigenvalues, min=GRAM_FLOOR)
                inverse_sqrt = torch.bmm(
                    eigenvectors * torch.rsqrt(eigenvalues)[:, None, :],
                    eigenvectors.transpose(1, 2),
                )
            factors = torch.bmm(candidate, inverse_sqrt)
            log_lambda = log_lambda_local.mean(dim=1)
            lambdas = torch.clamp(
                torch.nn.functional.softplus(log_lambda) + POSITIVE_OFFSET,
                min=POSITIVE_MINIMUM,
                max=POSITIVE_MAXIMUM,
            )
            diagonal = torch.clamp(
                torch.nn.functional.softplus(log_diagonal) + POSITIVE_OFFSET,
                min=POSITIVE_MINIMUM,
                max=POSITIVE_MAXIMUM,
            )
            projection = torch.bmm(
                factors.transpose(1, 2), score[..., None]
            ).squeeze(-1) / float(WIDTH)
            cross = torch.bmm(
                factors, (lambdas * projection)[..., None]
            ).squeeze(-1)
            delta = self.layer_rms[layer] * (diagonal * score + cross)
            outputs.append(delta)
            all_d.append(diagonal)
            all_lambda.append(lambdas)
            all_u.append(factors)
            scale_penalties.append(
                log_diagonal.square().mean(dim=1) + log_lambda.square().mean(dim=1)
            )
        prediction = torch.stack(outputs, dim=1)
        if return_covariance:
            return (
                prediction,
                torch.stack(all_d, dim=1),
                torch.stack(all_lambda, dim=1),
                torch.stack(all_u, dim=1),
                torch.stack(scale_penalties, dim=1),
            )
        return prediction


def gaussian_score(
    normalized_residual: torch.Tensor,
    diagonal: torch.Tensor,
    lambdas: torch.Tensor,
    factors: torch.Tensor,
) -> torch.Tensor:
    """Mean exact Gaussian score for C=diag(d)+U diag(lambda) U^T/W."""
    scores: list[torch.Tensor] = []
    identity = torch.eye(RANK, dtype=factors.dtype, device=factors.device)
    for layer in range(CORRECTED_LAYERS):
        residual = normalized_residual[:, layer]
        d = diagonal[:, layer]
        lam = lambdas[:, layer]
        u = factors[:, layer]
        inverse_d = torch.reciprocal(d)
        weighted_u = inverse_d[..., None] * u
        small = torch.bmm(u.transpose(1, 2), weighted_u) / float(WIDTH)
        small = small + identity[None] / lam[:, :, None]
        rhs = torch.bmm(
            u.transpose(1, 2), (inverse_d * residual)[..., None]
        ).squeeze(-1) / float(WIDTH)
        solved = torch.linalg.solve(small, rhs[..., None]).squeeze(-1)
        quadratic = (inverse_d * residual.square()).mean(dim=1) - (
            rhs * solved
        ).sum(dim=1)
        sign, logabsdet = torch.linalg.slogdet(small)
        if not bool((sign > 0).all()):
            raise FloatingPointError("covariance Woodbury system lost positivity")
        log_determinant = d.log().mean(dim=1) + (
            lam.log().sum(dim=1) + logabsdet
        ) / float(WIDTH)
        scores.append(0.5 * (quadratic + log_determinant))
    return torch.stack(scores, dim=1).mean()


def cost_record(model: nn.Module) -> dict[str, object]:
    dense = 2 * WIDTH * WIDTH - WIDTH
    factor_transport = TRANSITIONS * RANK * dense
    trunk = (2 * INPUT_FEATURES - 1) * HIDDEN + (2 * HIDDEN - 1) * HIDDEN
    heads = (2 * HIDDEN - 1) * HEAD_OUTPUTS
    local = CORRECTED_LAYERS * WIDTH * (trunk + heads)
    gram = CORRECTED_LAYERS * (
        2 * WIDTH * RANK * RANK + 20 * RANK**3
    )
    woodbury = CORRECTED_LAYERS * (4 * WIDTH * RANK + 20 * WIDTH)
    source_reserve = CORRECTED_LAYERS * WIDTH * RANK * 256
    output = CORRECTED_LAYERS * WIDTH
    added = factor_transport + local + gram + woodbury + source_reserve + output
    correction = J_CORRECTION_FLOPS + added
    total = NATIVE_HOST_TAPE_FLOPS + correction
    if added != N_ADDED_FLOPS or correction != CORRECTION_FLOPS or total != TOTAL_FLOPS:
        raise AssertionError((added, correction, total))
    if parameter_count(model) != 86_188:
        raise AssertionError(parameter_count(model))
    return {
        "new_parameter_count": parameter_count(model),
        "frozen_j_parameter_count": 25_600,
        "total_parameter_count": parameter_count(model) + 25_600,
        "frozen_j_correction_upper": J_CORRECTION_FLOPS,
        "rank16_factor_transport_upper": factor_transport,
        "local_encoder_and_heads_upper": local,
        "gram_whitening_and_small_eigensolve_upper": gram,
        "woodbury_apply_and_diagonal_upper": woodbury,
        "factor_source_gate_reserve": source_reserve,
        "output_addition": output,
        "added_covariance_observer_upper": added,
        "j_plus_covariance_observer_correction_upper": correction,
        "native_q24_host_tape_baseline": NATIVE_HOST_TAPE_FLOPS,
        "native_total_upper": total,
        "exact_scoring_budget": EXACT_SCORING_BUDGET,
        "headroom_below_exact_2pow41": EXACT_SCORING_BUDGET - total,
        "phase2_budget": PHASE2_BUDGET,
        "within_exact_2pow41": total < EXACT_SCORING_BUDGET,
        "within_phase2_budget": total < PHASE2_BUDGET,
        "authority": "frozen conservative source-structure estimate; literal meter only after strict accuracy pass",
    }


def build_j_model(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    model = J.TransportEnergyMultiStateResidual(
        state_dict["centers"].cpu().numpy(),
        state_dict["scales"].cpu().numpy(),
        state_dict["layer_rms"].cpu().numpy(),
    )
    model.load_state_dict(state_dict, strict=True)
    if J.parameter_count(model) != 25_600:
        raise AssertionError("frozen J parameter count changed")
    return model


def load_outer_j_model(outer_fold: int, seal: dict[str, object]) -> nn.Module:
    path = J_DIR / "folds" / f"fold_{outer_fold}.pt"
    if sha256(path) != seal["j_outer_model_sha256"][outer_fold]:
        raise AssertionError("outer J model changed")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("node") != "P2R0F2J" or artifact.get("fold") != outer_fold:
        raise AssertionError("outer J checkpoint identity changed")
    model = build_j_model(artifact["state_dict"])
    model.eval()
    return model


def fit_inner_j(
    outer_fold: int,
    inner_fold: int,
    model: nn.Module,
    dataset: PublicResidualDataset,
    train: np.ndarray,
    error: np.ndarray,
) -> list[dict[str, object]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=J.LEARNING_RATE, weight_decay=J.WEIGHT_DECAY
    )
    total_steps = J.EPOCHS * math.ceil(len(train) / J.BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=J.MIN_LEARNING_RATE
    )
    layer_weights = torch.linspace(0.5, 1.5, DEPTH - FIRST_CORRECTED_LAYER)
    seed = SEED ^ ((outer_fold + 1) * 0x45D9F3B) ^ ((inner_fold + 1) * 0x119DE1F3)
    order_rng = np.random.default_rng(seed & 0xFFFF_FFFF)
    trace: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch, exposure in enumerate(J.EXPOSURE_SCHEDULE, start=1):
        order = order_rng.permutation(train)
        objective_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), J.BATCH_SIZE):
            ranks = order[offset : offset + J.BATCH_SIZE]
            raw, sigma, weights, truth = load_batch(dataset, ranks, error)
            optimizer.zero_grad(set_to_none=True)
            prediction, states, _ = model(
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
            auxiliary = auxiliary / model.layer_rms[
                FIRST_CORRECTED_LAYER:
            ][None, :, None, None]
            identity_penalty = J.IDENTITY_PENALTY * (
                model.transport_delta.square().mean()
                + model.mix_delta.square().mean()
            )
            loss = (
                rollout
                + final_focus
                + J.AUX_STATE_PENALTY * auxiliary.square().mean()
                + identity_penalty
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"outer {outer_fold} inner {inner_fold} epoch {epoch}: nonfinite J loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), J.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            examples += len(ranks)
        rss = enforce_resources(
            f"outer {outer_fold} inner {inner_fold} J epoch {epoch}"
        )
        record = {
            "stage": "inner_oof_j",
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "epoch": epoch,
            "exposure": exposure,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "max_rss_bytes": rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    model.eval()
    return trace


def predict_j_observables(
    model: nn.Module,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.empty((len(ranks), DEPTH, WIDTH), dtype=np.float32)
    states = np.empty((len(ranks), DEPTH, WIDTH, J.CHANNELS), dtype=np.float32)
    energies = np.empty_like(states)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            raw, sigma, weights, _ = load_batch(dataset, group, error)
            prediction, state, energy = model(
                raw, sigma, weights, exposure=1.0, return_states=True
            )
            predictions[offset : offset + len(group)] = prediction.cpu().numpy()
            states[offset : offset + len(group)] = state.cpu().numpy()
            energies[offset : offset + len(group)] = energy.cpu().numpy()
    return predictions, states, energies


def build_inner_oof_cache(
    outer_fold: int,
    dataset: PublicResidualDataset,
    outer_train: np.ndarray,
    error: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    labels = np.asarray(
        [inner_fold_of(outer_fold, int(dataset.mlp_ids[int(rank)])) for rank in outer_train],
        dtype=np.int64,
    )
    inner_sizes = [int(np.sum(labels == inner)) for inner in range(INNER_FOLDS)]
    if min(inner_sizes) == 0 or sum(inner_sizes) != len(outer_train):
        raise AssertionError(f"invalid inner fold sizes: {inner_sizes}")
    predictions = np.empty((len(outer_train), DEPTH, WIDTH), dtype=np.float32)
    states = np.empty(
        (len(outer_train), DEPTH, WIDTH, J.CHANNELS), dtype=np.float32
    )
    energies = np.empty_like(states)
    all_trace: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for inner in range(INNER_FOLDS):
        valid_positions = np.flatnonzero(labels == inner)
        train_positions = np.flatnonzero(labels != inner)
        valid = outer_train[valid_positions]
        train = outer_train[train_positions]
        if set(valid.tolist()).intersection(train.tolist()):
            raise AssertionError("inner train/valid overlap")
        centers, scales, layer_rms = moments_for(train, error, sums, sums2)
        torch.manual_seed(
            (SEED + 10_000 + 101 * outer_fold + inner) & 0x7FFF_FFFF
        )
        model = J.TransportEnergyMultiStateResidual(centers, scales, layer_rms)
        trace = fit_inner_j(
            outer_fold, inner, model, dataset, train, error
        )
        prediction, state, energy = predict_j_observables(
            model, dataset, valid, error
        )
        predictions[valid_positions] = prediction
        states[valid_positions] = state
        energies[valid_positions] = energy
        all_trace.extend(trace)
        receipts.append(
            {
                "inner_fold": inner,
                "train_networks": len(train),
                "valid_networks": len(valid),
                "valid_role_ranks": valid.tolist(),
                "outer_validation_rows_in_train": 0,
                "fixed_final_epoch": J.EPOCHS,
            }
        )
        del model, prediction, state, energy
        gc.collect()
    return (
        {
            "ranks": outer_train.copy(),
            "prediction": predictions,
            "states": states,
            "energies": energies,
        },
        all_trace,
        receipts,
    )


def cache_rows(cache: dict[str, np.ndarray], ranks: np.ndarray) -> tuple[torch.Tensor, ...]:
    all_ranks = cache["ranks"]
    positions = np.searchsorted(all_ranks, ranks)
    if not np.array_equal(all_ranks[positions], ranks):
        raise AssertionError("inner OOF cache rank lookup failed")
    return (
        torch.from_numpy(np.ascontiguousarray(cache["prediction"][positions])),
        torch.from_numpy(np.ascontiguousarray(cache["states"][positions])),
        torch.from_numpy(np.ascontiguousarray(cache["energies"][positions])),
    )


def fit_observer(
    outer_fold: int,
    model: Rank16CrossUnitCovarianceObserver,
    dataset: PublicResidualDataset,
    train: np.ndarray,
    error: np.ndarray,
    cache: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * math.ceil(len(train) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=MIN_LEARNING_RATE
    )
    layer_weights = torch.linspace(0.5, 1.5, CORRECTED_LAYERS)
    order_rng = np.random.default_rng(SEED ^ ((outer_fold + 1) * 0xD1B54A35))
    trace: list[dict[str, object]] = []
    maximum_rss = enforce_resources(f"outer {outer_fold} N training start")
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = order_rng.permutation(train)
        objective_sum = 0.0
        mse_sum = 0.0
        gaussian_sum = 0.0
        examples = 0
        model.train()
        for offset in range(0, len(order), BATCH_SIZE):
            ranks = order[offset : offset + BATCH_SIZE]
            raw, sigma, weights, truth = load_batch(dataset, ranks, error)
            j_prediction, j_states, j_energies = cache_rows(cache, ranks)
            optimizer.zero_grad(set_to_none=True)
            delta, diagonal, lambdas, factors, scale_penalty = model(
                raw,
                sigma,
                weights,
                j_states,
                j_energies,
                return_covariance=True,
            )
            remaining = truth - j_prediction
            normalized = (
                delta[:, FIRST_CORRECTED_LAYER:]
                - remaining[:, FIRST_CORRECTED_LAYER:]
            ) / model.layer_rms[FIRST_CORRECTED_LAYER:][None, :, None]
            layer_mse = normalized.square().mean(dim=(0, 2))
            mse_loss = torch.mean(layer_weights * layer_mse) + layer_mse[-1]
            covariance_loss = gaussian_score(
                normalized, diagonal, lambdas, factors
            )
            loss = (
                mse_loss
                + GAUSSIAN_WEIGHT * covariance_loss
                + LOG_SCALE_PENALTY * scale_penalty.mean()
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"outer {outer_fold} N epoch {epoch}: nonfinite loss"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            objective_sum += float(loss.detach()) * len(ranks)
            mse_sum += float(mse_loss.detach()) * len(ranks)
            gaussian_sum += float(covariance_loss.detach()) * len(ranks)
            examples += len(ranks)
        maximum_rss = max(
            maximum_rss, enforce_resources(f"outer {outer_fold} N epoch {epoch}")
        )
        record = {
            "stage": "rank16_covariance_observer",
            "outer_fold": outer_fold,
            "epoch": epoch,
            "fit_objective": objective_sum / examples,
            "normalized_mse_objective": mse_sum / examples,
            "gaussian_score": gaussian_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "factor_gain_delta_max_abs": float(
                model.factor_gain_delta.detach().abs().max()
            ),
            "factor_mix_delta_max_abs": float(
                model.factor_mix_delta.detach().abs().max()
            ),
            "max_rss_bytes": maximum_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, maximum_rss


def score_outer(
    observer: Rank16CrossUnitCovarianceObserver,
    j_model: nn.Module,
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> dict[str, object]:
    layer_sse = np.zeros(DEPTH, dtype=np.float64)
    j_layer_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[dict[str, int | float]] = []
    maximum_delta = 0.0
    finite = True
    observer.eval()
    j_model.eval()
    with torch.no_grad():
        for offset in range(0, len(ranks), 2):
            group = ranks[offset : offset + 2]
            raw, sigma, weights, truth = load_batch(dataset, group, error)
            j_prediction, j_states, j_energies = j_model(
                raw, sigma, weights, exposure=1.0, return_states=True
            )
            delta = observer(raw, sigma, weights, j_states, j_energies)
            assert isinstance(delta, torch.Tensor)
            prediction = j_prediction + delta
            finite = finite and bool(torch.isfinite(prediction).all())
            maximum_delta = max(maximum_delta, float(delta.abs().max()))
            squared = (prediction - truth).double().square()
            j_squared = (j_prediction - truth).double().square()
            layer_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            j_layer_sse += j_squared.sum(dim=(0, 2)).cpu().numpy()
            finals = squared[:, -1].mean(dim=1).cpu().numpy()
            per_network.extend(
                {
                    "role_rank": int(rank),
                    "mlp_id": int(dataset.mlp_ids[int(rank)]),
                    "final_mse": float(value),
                }
                for rank, value in zip(group, finals, strict=True)
            )
    denominator = len(ranks) * WIDTH
    layer_mse = layer_sse / denominator
    j_layer_mse = j_layer_sse / denominator
    return {
        "networks": len(ranks),
        "finite_full_rollout": finite,
        "maximum_absolute_added_delta": maximum_delta,
        "final_mse": float(layer_mse[-1]),
        "layer_mse": layer_mse.tolist(),
        "frozen_j_final_mse": float(j_layer_mse[-1]),
        "frozen_j_layer_mse": j_layer_mse.tolist(),
        "per_network": per_network,
    }


def fold_paths(fold: int) -> tuple[Path, Path]:
    return FOLD_DIR / f"fold_{fold}.json", FOLD_DIR / f"fold_{fold}.pt"


def run_fold(outer_fold: int) -> dict[str, object]:
    if outer_fold < 0 or outer_fold >= FOLDS:
        raise ValueError(outer_fold)
    seal = verify_seal()
    release = verify_release(seal)
    result_path, model_path = fold_paths(outer_fold)
    if result_path.exists() or model_path.exists():
        if not result_path.is_file() or not model_path.is_file():
            raise FileExistsError("partial fold artifact collision")
        existing = read_json(result_path)
        verify_completed_fold_record(existing, outer_fold, seal)
        return existing
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    valid = np.flatnonzero(folds == outer_fold)
    train = np.flatnonzero(folds != outer_fold)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    started = time.perf_counter()
    cache, inner_trace, inner_receipts = build_inner_oof_cache(
        outer_fold, dataset, train, error, sums, sums2
    )
    torch.manual_seed(SEED + 20_000 + outer_fold)
    observer = Rank16CrossUnitCovarianceObserver(centers, scales, layer_rms)
    if parameter_count(observer) != 86_188:
        raise AssertionError(parameter_count(observer))
    cost = cost_record(observer)
    if not cost["within_exact_2pow41"] or not cost["within_phase2_budget"]:
        raise RuntimeError("frozen N model misses cost admission")
    trace, max_rss = fit_observer(
        outer_fold, observer, dataset, train, error, cache
    )
    del cache
    gc.collect()
    j_model = load_outer_j_model(outer_fold, seal)
    autonomous = score_outer(observer, j_model, dataset, valid, error)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-rank16-cross-unit-covariance-fold-v1",
        "status": "complete",
        "node": NODE,
        "fold": outer_fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "inner_stacking": {
            "folds": INNER_FOLDS,
            "outer_validation_rows_used": 0,
            "receipts": inner_receipts,
            "trace": inner_trace,
        },
        "architecture": {
            "covariance_rank": RANK,
            "local_input_features": INPUT_FEATURES,
            "hidden": HIDDEN,
            "head_outputs": HEAD_OUTPUTS,
            "new_parameters": parameter_count(observer),
            "total_parameters_including_frozen_j": parameter_count(observer) + 25_600,
            "factor_transport": "Phi*(U@W)",
            "covariance": "R^2[diag(d)+U diag(lambda) U^T/W]",
            "deployment_truth_inputs": 0,
            "network_identity_inputs": 0,
            "selectors": 0,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "fixed_final_checkpoint": True,
            "trace": trace,
            "observed_max_rss_bytes": max_rss,
        },
        "host": {
            "final_mse": float(host_layer[-1]),
            "layer_mse": host_layer.tolist(),
        },
        "autonomous": autonomous,
        "cost": cost,
        "capacity_only_pending_n_specific_native_verification": True,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
        "outer_j_model_sha256": seal["j_outer_model_sha256"][outer_fold],
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "schema": "public-full688-rank16-cross-unit-covariance-fold-model-v1",
        "node": NODE,
        "fold": outer_fold,
        "state_dict": observer.state_dict(),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
        "outer_j_model_sha256": seal["j_outer_model_sha256"][outer_fold],
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
                "fold": outer_fold,
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
    release = verify_release(seal)
    if CV_RESULT.exists():
        existing = read_json(CV_RESULT)
        expected = {
            "schema": "public-full688-rank16-cross-unit-covariance-cv-v1",
            "status": "complete",
            "node": NODE,
            "observer_selection_targets_opened": 0,
            "public_name_joins": 0,
            "network_access": 0,
            "preregistration_sha256": seal["preregistration_sha256"],
            "program_sha256": seal["program_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "seal_sha256": sha256(SEAL),
            "training_release_sha256": sha256(TRAINING_RELEASE),
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise AssertionError(f"completed CV receipt mismatch at {key}")
        return existing
    records = []
    for fold in range(FOLDS):
        result_path, model_path = fold_paths(fold)
        if not result_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"fold {fold} is incomplete")
        record = read_json(result_path)
        verify_completed_fold_record(record, fold, seal)
        if record.get("model_sha256") != sha256(model_path):
            raise AssertionError(f"fold {fold}: model hash mismatch")
        records.append(record)
    total = sum(int(record["valid_networks"]) for record in records)
    valid_union = [int(rank) for record in records for rank in record["valid_role_ranks"]]
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
    per_network = [row for record in records for row in record["autonomous"]["per_network"]]
    maximum_row = max(per_network, key=lambda row: float(row["final_mse"]))
    maximum_network = float(maximum_row["final_mse"])
    finite = all(record["autonomous"]["finite_full_rollout"] for record in records)
    fold_pass = all(
        float(record["autonomous"]["final_mse"]) < FOLD_GATE for record in records
    )
    mean_pass = autonomous_final < OOF_GATE
    maximum_pass = maximum_network < MAX_NETWORK_GATE
    result: dict[str, object] = {
        "schema": "public-full688-rank16-cross-unit-covariance-cv-v1",
        "status": "complete",
        "node": NODE,
        "observer_fit_networks": total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "capacity_only_pending_n_specific_native_verification": True,
        "architecture": records[0]["architecture"],
        "training": {
            "epochs": EPOCHS,
            "batch_size_networks": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "minimum_learning_rate": MIN_LEARNING_RATE,
            "fixed_final_checkpoint": True,
            "inner_j_folds_per_outer_fold": INNER_FOLDS,
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
            "finite_all_688_outer_oof_rollouts": finite,
            "oof_final_mse_below_1e_8": mean_pass,
            "every_outer_fold_below_1_1e_8": fold_pass,
            "maximum_per_network_below_1e_7": maximum_pass,
            "observer_selection_targets_opened_zero": True,
            "conjunctive_pass": finite and mean_pass and fold_pass and maximum_pass,
        },
        "cost": records[0]["cost"],
        "fold_results": [
            {
                "fold": record["fold"],
                "valid_networks": record["valid_networks"],
                "host_final_mse": record["host"]["final_mse"],
                "frozen_j_final_mse": record["autonomous"]["frozen_j_final_mse"],
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
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
    }
    write_json_once(CV_RESULT, result)
    print(json.dumps({"oof": result["oof"], "gate": result["decision_gate"]}, sort_keys=True))
    return result


def static_check() -> None:
    centers = np.zeros((DEPTH, LOCAL_FEATURES - 1), dtype=np.float32)
    scales = np.ones_like(centers)
    layer_rms = np.ones(DEPTH, dtype=np.float32)
    model = Rank16CrossUnitCovarianceObserver(centers, scales, layer_rms)
    raw = torch.zeros(1, DEPTH, WIDTH, LOCAL_FEATURES)
    raw[..., 5] = 0.5
    sigma = torch.ones(1, DEPTH, WIDTH)
    weights = torch.zeros(1, DEPTH, WIDTH, WIDTH)
    j_states = torch.zeros(1, DEPTH, WIDTH, J.CHANNELS)
    j_energies = torch.zeros_like(j_states)
    delta, diagonal, lambdas, factors, penalty = model(
        raw,
        sigma,
        weights,
        j_states,
        j_energies,
        return_covariance=True,
    )
    normalized = delta[:, FIRST_CORRECTED_LAYER:]
    synthetic_loss = (
        normalized.square().mean()
        + GAUSSIAN_WEIGHT * gaussian_score(normalized, diagonal, lambdas, factors)
        + LOG_SCALE_PENALTY * penalty.mean()
    )
    synthetic_loss.backward()
    finite_gradient = all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
    )
    if not bool(torch.isfinite(synthetic_loss)) or not finite_gradient:
        raise FloatingPointError("static synthetic forward/backward is nonfinite")
    payload = {
        "node": NODE,
        "status": "PASS_STATIC_NO_DATA_NO_METRICS",
        "architecture": {
            "parameters": parameter_count(model),
            "rank": RANK,
            "input_features": INPUT_FEATURES,
            "head_outputs": HEAD_OUTPUTS,
            "synthetic_output_shape": list(delta.shape),
            "synthetic_factor_shape": list(factors.shape),
            "synthetic_forward_backward_finite": True,
        },
        "cost": cost_record(model),
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def smoke() -> None:
    seal = verify_seal()
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    train = np.flatnonzero(folds != 0)
    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    observer = Rank16CrossUnitCovarianceObserver(centers, scales, layer_rms)
    j_model = load_outer_j_model(0, seal)
    row = dataset.load(0)
    raw = torch.from_numpy(np.asarray(row.raw_locals)[None].copy())
    sigma = torch.from_numpy(np.asarray(row.sigma)[None].copy())
    weights = torch.from_numpy(
        np.asarray(np.load(row.weights_path, mmap_mode="r", allow_pickle=False))[None].copy()
    )
    with torch.no_grad():
        j_prediction, j_states, j_energies = j_model(
            raw, sigma, weights, exposure=1.0, return_states=True
        )
    delta, diagonal, lambdas, factors, penalty = observer(
        raw,
        sigma,
        weights,
        j_states,
        j_energies,
        return_covariance=True,
    )
    if tuple(delta.shape) != (1, DEPTH, WIDTH):
        raise AssertionError("smoke output shape changed")
    if tuple(factors.shape) != (1, CORRECTED_LAYERS, WIDTH, RANK):
        raise AssertionError("smoke factor shape changed")
    tensors = [j_prediction, delta, diagonal, lambdas, factors, penalty]
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("smoke tensor is nonfinite")
    synthetic_normalized = delta[:, FIRST_CORRECTED_LAYER:] / observer.layer_rms[
        FIRST_CORRECTED_LAYER:
    ][None, :, None]
    synthetic_loss = (
        synthetic_normalized.square().mean()
        + GAUSSIAN_WEIGHT
        * gaussian_score(synthetic_normalized, diagonal, lambdas, factors)
        + LOG_SCALE_PENALTY * penalty.mean()
    )
    if not bool(torch.isfinite(synthetic_loss)):
        raise FloatingPointError("synthetic smoke loss is nonfinite")
    synthetic_loss.backward()
    if not all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in observer.parameters()
    ):
        raise FloatingPointError("synthetic smoke gradient is missing or nonfinite")
    gram = torch.bmm(factors[:, -1].transpose(1, 2), factors[:, -1]) / float(WIDTH)
    inner_sizes = []
    for outer_fold in range(FOLDS):
        outer_train = np.flatnonzero(folds != outer_fold)
        labels = np.asarray(
            [
                inner_fold_of(outer_fold, int(dataset.mlp_ids[int(rank)]))
                for rank in outer_train
            ],
            dtype=np.int64,
        )
        sizes = [int(np.sum(labels == inner)) for inner in range(INNER_FOLDS)]
        if min(sizes) == 0 or sum(sizes) != len(outer_train):
            raise AssertionError("smoke inner split coverage changed")
        inner_sizes.append(sizes)
    payload = {
        "node": NODE,
        "status": "PASS_SYNTHETIC_AND_ONE_INPUT_NO_TRUTH_METRICS",
        "launch_status": "held_after_sealed_smoke",
        "readiness": readiness,
        "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        "inner_fold_sizes_by_outer_fold": inner_sizes,
        "architecture": {
            "parameters": parameter_count(observer),
            "delta_shape": list(delta.shape),
            "factor_shape": list(factors.shape),
            "finite": True,
            "finite_full_backward_gradient": True,
            "diagonal_positive": bool((diagonal > 0).all()),
            "lambda_positive": bool((lambdas > 0).all()),
            "whitened_last_gram_max_abs_from_identity": float(
                (gram - torch.eye(RANK)[None]).abs().max()
            ),
        },
        "cost": cost_record(observer),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    configure()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fold", type=int)
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.static:
        static_check()
    elif args.smoke:
        smoke()
    elif args.fold is not None:
        run_fold(args.fold)
    else:
        aggregate()


if __name__ == "__main__":
    main()
