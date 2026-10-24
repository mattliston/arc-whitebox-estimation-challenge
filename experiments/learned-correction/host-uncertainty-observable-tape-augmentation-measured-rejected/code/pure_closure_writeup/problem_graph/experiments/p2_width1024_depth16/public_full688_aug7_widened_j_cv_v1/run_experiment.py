#!/Users/matt/anaconda3/bin/python3
"""Frozen P2R0F2Q aug7-widened J CV: the exact P2R0F2J architecture with 27
raw local channels (43 cell inputs) and nothing else changed."""

from __future__ import annotations

import argparse
import gc
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


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
P_DIR = P2 / "public_full688_fullrank_roundtrip_residual_cv_v1"
AUG_DIR = P2 / "public_aug7_source_observable_tape_v1"
AUG_ROWS = AUG_DIR / "rows/observer_fit"
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


PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUN_MANIFEST = HERE / "RUN_MANIFEST.json"
SEAL = HERE / "PREREGISTRATION_SEAL.json"
STATIC_RECEIPT = HERE / "STATIC_RECEIPT.json"
AUDIT_RECEIPT = HERE / "AUDIT_RECEIPT.json"
SMOKE_RECEIPT = HERE / "SMOKE_RECEIPT.json"
TRAINING_RELEASE = HERE / "TRAINING_RELEASE.json"
FOLD_DIR = HERE / "folds"
CV_RESULT = HERE / "CV_RESULTS.json"
SHARED_CACHE_MANIFEST = D_DIR / "cache/MANIFEST.json"
SHARED_ERROR = D_DIR / "cache/patched_error.npy"
SHARED_STATS = D_DIR / "cache/raw_sufficient_stats.npz"
GENERATION_RECEIPT = AUG_DIR / "GENERATION_RECEIPT.json"

NODE = "P2R0F2Q"
SEED = 1_753_358_438
FOLDS = 4
CHANNELS = 4
HIDDEN = 96
AUG_CHANNELS = 7
RAW_CHANNELS = LOCAL_FEATURES + AUG_CHANNELS
INPUT_FEATURES = RAW_CHANNELS + 4 * CHANNELS
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
EXPECTED_PARAMETERS = 26_272
OOF_GATE = 1.0e-8
FOLD_GATE = 1.1e-8
MAX_NETWORK_GATE = 1.0e-7
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_STORAGE_BYTES = 2 * (1 << 30)
MEAN_TRANSPORT_FLOPS = 217_997_312
RAW_MATERIALIZATION_RESERVE = 385_875_968
RAW_STANDARDIZATION = 884_736
LOCAL_CELL_FLOPS = 583_819_264
OUTPUT_ADDITION = 16_384
CORRECTION_FLOPS = 1_188_593_664
AUG_NATIVE_RESERVE = 500_000_000
NATIVE_HOST_TAPE_FLOPS = 2_131_692_137_721
TOTAL_FLOPS = 2_133_380_731_385
EXACT_SCORING_BUDGET = 1 << 41
EXPECTED_FOLD_SIZES = [147, 172, 171, 198]
TAIL_RANKS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
_ROW_MANIFESTS_VERIFIED = False
_AUG_VERIFIED: set[int] = set()
_GENERATION = None


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


def load_generation_receipt() -> dict[str, object]:
    global _GENERATION
    if _GENERATION is None:
        receipt = read_json(GENERATION_RECEIPT)
        if receipt.get("status") != "PASS_GENERATION_688_PARITY":
            raise AssertionError("generation receipt did not pass")
        if receipt.get("rows") != 688 or receipt.get("parity_pass_rows") != 688:
            raise AssertionError("generation coverage changed")
        if len(receipt.get("row_manifest_sha256", [])) != 688:
            raise AssertionError("generation row roster changed")
        _GENERATION = receipt
    return _GENERATION


def verify_seal() -> dict[str, object]:
    if not SEAL.is_file():
        raise FileNotFoundError("PREREGISTRATION_SEAL.json is required")
    seal = read_json(SEAL)
    if seal.get("status") != "frozen-before-any-q-metrics":
        raise AssertionError("seal status changed")
    bindings = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "program_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "static_receipt_sha256": sha256(STATIC_RECEIPT),
        "audit_receipt_sha256": sha256(AUDIT_RECEIPT),
        "loader_sha256": sha256(LOADER_DIR / "dataset.py"),
        "shared_cache_manifest_sha256": sha256(SHARED_CACHE_MANIFEST),
        "shared_error_sha256": sha256(SHARED_ERROR),
        "shared_stats_sha256": sha256(SHARED_STATS),
        "generation_preregistration_sha256": sha256(AUG_DIR / "PREREGISTRATION.json"),
        "generation_program_sha256": sha256(AUG_DIR / "generate_aug7.py"),
        "generation_audit_sha256": sha256(AUG_DIR / "AUDIT_RECEIPT.json"),
        "generation_receipt_sha256": sha256(GENERATION_RECEIPT),
        "j_program_sha256": sha256(J_DIR / "run_experiment.py"),
        "j_cv_results_sha256": sha256(J_DIR / "CV_RESULTS.json"),
        "j_verification_sha256": sha256(J_DIR / "VERIFICATION.json"),
        "p_decomposition_sha256": sha256(
            P_DIR / "P_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json"
        ),
        "p_successor_proposal_sha256": sha256(
            P_DIR / "ONE_GRAPH_READY_SUCCESSOR_PROPOSAL.json"
        ),
        "feature_store_manifest_sha256": sha256(FEATURE_STORE_ROOT / "manifest.json"),
        "observer_fit_target_parquet_sha256": sha256(
            TARGET_ROOT / "observer_fit.parquet"
        ),
    }
    for key, value in bindings.items():
        if seal.get(key) != value:
            raise AssertionError(f"sealed binding changed: {key}")
    audit = read_json(AUDIT_RECEIPT)
    if audit.get("status") != "PASS_INDEPENDENT_PRELAUNCH_AUDIT":
        raise AssertionError("audit receipt did not pass")
    if audit.get("node") != NODE:
        raise AssertionError("audit receipt node changed")
    load_generation_receipt()
    prereg = read_json(PREREGISTRATION)
    manifest = read_json(RUN_MANIFEST)
    if prereg.get("node") != NODE or manifest.get("node") != NODE:
        raise AssertionError("node binding changed")
    if prereg.get("status") != "frozen-before-any-q-metrics":
        raise AssertionError("preregistration status changed")
    if (
        manifest.get("status")
        != "authorized-by-root-design-frozen-launch-held-until-generation-seal-audit-and-smoke"
    ):
        raise AssertionError("run authorization changed")
    if prereg["scope"]["observer_selection_targets_allowed"] is not False:
        raise AssertionError("selection firewall changed")
    if prereg["scope"]["public_name_join_allowed"] is not False:
        raise AssertionError("name firewall changed")
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
        "audit_receipt_sha256": seal["audit_receipt_sha256"],
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
        "schema": "public-full688-aug7-widened-j-fold-v1",
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


def aug_row(rank: int, dataset: PublicResidualDataset) -> np.ndarray:
    generation = load_generation_receipt()
    row_dir = AUG_ROWS / f"{rank:04d}"
    manifest_path = row_dir / "row_manifest.json"
    if rank not in _AUG_VERIFIED:
        if sha256(manifest_path) != generation["row_manifest_sha256"][rank]:
            raise AssertionError(f"rank {rank}: aug row manifest changed")
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") != "SEALED_ROW_AUG7"
            or manifest.get("parity_pass") is not True
            or int(manifest.get("role_rank", -1)) != rank
            or int(manifest.get("mlp_id", -1)) != int(dataset.mlp_ids[rank])
        ):
            raise AssertionError(f"rank {rank}: aug row manifest semantics changed")
        if sha256(row_dir / "aug7_locals.npy") != manifest["aug7_sha256"]:
            raise AssertionError(f"rank {rank}: aug row payload changed")
        _AUG_VERIFIED.add(rank)
    aug = np.load(row_dir / "aug7_locals.npy", mmap_mode="r", allow_pickle=False)
    if aug.dtype != np.float32 or aug.shape != (DEPTH, WIDTH, AUG_CHANNELS):
        raise AssertionError(f"rank {rank}: aug schema changed")
    return aug


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


def aug_sufficient_stats(dataset: PublicResidualDataset) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((688, DEPTH, AUG_CHANNELS), dtype=np.float64)
    sums2 = np.zeros((688, DEPTH, AUG_CHANNELS), dtype=np.float64)
    for rank in range(688):
        aug = np.asarray(aug_row(rank, dataset), dtype=np.float64)
        sums[rank] = aug.sum(axis=1)
        sums2[rank] = np.square(aug).sum(axis=1)
    return sums, sums2


def moments_for(
    train: np.ndarray,
    error: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    aug_sums: np.ndarray,
    aug_sums2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = float(len(train) * WIDTH)
    total = np.concatenate(
        (sums[train].sum(axis=0, dtype=np.float64), aug_sums[train].sum(axis=0)), axis=1
    )
    total2 = np.concatenate(
        (sums2[train].sum(axis=0, dtype=np.float64), aug_sums2[train].sum(axis=0)),
        axis=1,
    )
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
        np.ascontiguousarray(
            np.concatenate(
                (
                    np.stack([row.raw_locals for row in rows]),
                    np.stack([aug_row(int(rank), dataset) for rank in ranks]),
                ),
                axis=3,
            ),
            dtype=np.float32,
        )
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


class AugWidenedTransportEnergyResidual(nn.Module):
    """P2R0F2J's four-channel observer with 27 raw local channels."""

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
        self.source_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN, CHANNELS))
        self.source_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.gate_weight = nn.Parameter(torch.empty(DEPTH, HIDDEN, CHANNELS))
        self.gate_bias = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.transport_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS))
        self.mix_delta = nn.Parameter(torch.zeros(DEPTH, CHANNELS, CHANNELS))
        nn.init.normal_(self.source_weight, mean=0.0, std=2.0e-3)
        nn.init.normal_(self.gate_weight, mean=0.0, std=2.0e-3)

    def local_hidden(
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
        features = torch.cat(
            (raw_layer[..., :1], tail, message_rms, message_sigma, log_energy, coherence),
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
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                layer_weights = weights[:, layer]
                messages = torch.bmm(
                    prior.transpose(1, 2), layer_weights
                ).transpose(1, 2)
                energies = torch.bmm(
                    prior.square().transpose(1, 2), layer_weights.square()
                ).transpose(1, 2)
                phi = raw[:, layer, :, 5, None]
                messages = phi * messages
                energies = phi.square() * energies
            hidden = self.local_hidden(
                raw[:, layer], sigma[:, layer], messages, energies, layer
            )
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
        prediction = torch.stack(outputs, dim=1)
        if return_states:
            return (
                prediction,
                torch.stack(state_history, dim=1),
                torch.stack(energy_history, dim=1),
            )
        return prediction


def cost_record(model: nn.Module) -> dict[str, object]:
    corrected = DEPTH - FIRST_CORRECTED_LAYER
    transitions = corrected - 1
    dense_vector_matrix = 2 * WIDTH * WIDTH - WIDTH
    mean_plus_energy_transport = transitions * 2 * CHANNELS * dense_vector_matrix
    raw_materialization_reserve = DEPTH * (16 + AUG_CHANNELS) * WIDTH * WIDTH
    raw_standardization = 2 * RAW_CHANNELS * DEPTH * WIDTH
    per_unit_dense = (
        (2 * INPUT_FEATURES - 1) * HIDDEN
        + (2 * HIDDEN - 1) * HIDDEN
        + (2 * HIDDEN - 1) * (2 * CHANNELS)
    )
    nonlinear_reserve = 64 * (2 * HIDDEN + CHANNELS)
    state_vector_and_mix_reserve = 128 + CHANNELS * (2 * CHANNELS - 1)
    local_cell = corrected * WIDTH * (
        per_unit_dense + nonlinear_reserve + state_vector_and_mix_reserve
    )
    output_addition = DEPTH * WIDTH
    correction = (
        mean_plus_energy_transport
        + raw_materialization_reserve
        + raw_standardization
        + local_cell
        + output_addition
    )
    total = NATIVE_HOST_TAPE_FLOPS + correction + AUG_NATIVE_RESERVE
    if (
        mean_plus_energy_transport != MEAN_TRANSPORT_FLOPS
        or raw_materialization_reserve != RAW_MATERIALIZATION_RESERVE
        or raw_standardization != RAW_STANDARDIZATION
        or local_cell != LOCAL_CELL_FLOPS
        or output_addition != OUTPUT_ADDITION
        or correction != CORRECTION_FLOPS
        or total != TOTAL_FLOPS
    ):
        raise AssertionError((correction, total))
    if parameter_count(model) != EXPECTED_PARAMETERS:
        raise AssertionError(parameter_count(model))
    return {
        "parameter_count": parameter_count(model),
        "mean_plus_energy_transport_upper": mean_plus_energy_transport,
        "raw_materialization_reserve_27ch": raw_materialization_reserve,
        "raw_standardization": raw_standardization,
        "local_cell_dense_plus_special_reserve": local_cell,
        "output_addition": output_addition,
        "correction_upper_estimate": correction,
        "aug_native_observable_reserve": AUG_NATIVE_RESERVE,
        "native_q24_host_tape_baseline": NATIVE_HOST_TAPE_FLOPS,
        "native_total_upper": total,
        "exact_scoring_budget": EXACT_SCORING_BUDGET,
        "headroom_below_exact_2pow41": EXACT_SCORING_BUDGET - total,
        "within_exact_2pow41": total < EXACT_SCORING_BUDGET,
        "authority": "conservative capacity-stage admission; the exact native ledger is deferred to the native stage's own preregistration",
    }


def fit_fold(
    fold: int,
    model: AugWidenedTransportEnergyResidual,
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
            "stage": "aug7_widened_j",
            "fold": fold,
            "epoch": epoch,
            "own_primary_state_exposure": exposure,
            "fit_objective": objective_sum / examples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "transport_delta_max_abs": float(model.transport_delta.detach().abs().max()),
            "mix_delta_max_abs": float(model.mix_delta.detach().abs().max()),
            "max_rss_bytes": max_rss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return trace, max_rss


def score(
    model: AugWidenedTransportEnergyResidual,
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
    verify_release(seal)
    result_path, model_path = fold_paths(fold)
    if result_path.exists() or model_path.exists():
        if not result_path.is_file() or not model_path.is_file():
            raise FileExistsError("partial fold artifact collision")
        existing = read_json(result_path)
        verify_completed_fold_record(existing, fold, seal)
        return existing
    dataset = PublicResidualDataset()
    readiness = readiness_or_raise(dataset)
    error, sums, sums2, folds = load_shared_cache(dataset)
    aug_sums, aug_sums2 = aug_sufficient_stats(dataset)
    valid = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    centers, scales, layer_rms = moments_for(
        train, error, sums, sums2, aug_sums, aug_sums2
    )
    torch.manual_seed(SEED + 40_000 + fold)
    model = AugWidenedTransportEnergyResidual(centers, scales, layer_rms)
    cost = cost_record(model)
    if not cost["within_exact_2pow41"]:
        raise RuntimeError("frozen Q model misses cost admission")
    started = time.perf_counter()
    trace, max_rss = fit_fold(fold, model, dataset, train, error)
    autonomous = score(model, dataset, valid, error)
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    result: dict[str, object] = {
        "schema": "public-full688-aug7-widened-j-fold-v1",
        "status": "complete",
        "node": NODE,
        "fold": fold,
        "train_networks": len(train),
        "valid_networks": len(valid),
        "train_role_ranks": train.tolist(),
        "valid_role_ranks": valid.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "readiness": readiness,
        "architecture": {
            "channels": CHANNELS,
            "hidden": HIDDEN,
            "raw_channels": RAW_CHANNELS,
            "input_features": INPUT_FEATURES,
            "parameters": parameter_count(model),
            "first_corrected_layer": FIRST_CORRECTED_LAYER,
            "signed_transport": "Phi*(s@W)",
            "energy_observable": "Phi^2*(s^2@W^2)",
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
        "capacity_only_pending_native_stage": True,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
        "elapsed_seconds": time.perf_counter() - started,
    }
    artifact = {
        "schema": "public-full688-aug7-widened-j-fold-model-v1",
        "node": NODE,
        "fold": fold,
        "state_dict": model.state_dict(),
        "preregistration_sha256": seal["preregistration_sha256"],
        "program_sha256": seal["program_sha256"],
        "run_manifest_sha256": seal["run_manifest_sha256"],
        "seal_sha256": sha256(SEAL),
        "training_release_sha256": sha256(TRAINING_RELEASE),
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
    verify_release(seal)
    if CV_RESULT.exists():
        existing = read_json(CV_RESULT)
        expected = {
            "schema": "public-full688-aug7-widened-j-cv-v1",
            "status": "complete",
            "node": NODE,
            "observer_selection_targets_opened": 0,
            "public_name_joins": 0,
            "network_access": 0,
            "preregistration_sha256": seal["preregistration_sha256"],
            "program_sha256": seal["program_sha256"],
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
        "schema": "public-full688-aug7-widened-j-cv-v1",
        "status": "complete",
        "node": NODE,
        "observer_fit_networks": total,
        "folds": FOLDS,
        "group_key": "mlp_id",
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "capacity_only_pending_native_stage": True,
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
        "j_baseline": {
            "oof_final_mse": 3.2334958229470984e-8,
            "source": "public_full688_transport_energy_coherence_cv_v1/CV_RESULTS.json",
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
    centers = np.zeros((DEPTH, RAW_CHANNELS - 1), dtype=np.float32)
    scales = np.ones_like(centers)
    layer_rms = np.ones(DEPTH, dtype=np.float32)
    torch.manual_seed(0)
    model = AugWidenedTransportEnergyResidual(centers, scales, layer_rms)
    raw = torch.zeros(1, DEPTH, WIDTH, RAW_CHANNELS)
    raw[..., 5] = 0.5
    sigma = torch.ones(1, DEPTH, WIDTH)
    weights = 0.03 * torch.randn(1, DEPTH, WIDTH, WIDTH)
    prediction, states, energies = model(
        raw, sigma, weights, exposure=1.0, return_states=True
    )
    if tuple(prediction.shape) != (1, DEPTH, WIDTH):
        raise AssertionError("static output shape changed")
    synthetic_loss = prediction[:, FIRST_CORRECTED_LAYER:].square().mean()
    synthetic_loss.backward()
    gradients = [value.grad for value in model.parameters()]
    finite_gradient = all(
        grad is not None and bool(torch.isfinite(grad).all()) for grad in gradients
    )
    nonzero_gradient = any(
        grad is not None and float(grad.abs().max()) > 0.0 for grad in gradients
    )
    if not bool(torch.isfinite(synthetic_loss)) or not finite_gradient:
        raise FloatingPointError("static synthetic forward/backward is nonfinite")
    if not nonzero_gradient:
        raise AssertionError("static synthetic backward produced no gradient signal")
    payload = {
        "node": NODE,
        "status": "PASS_STATIC_NO_DATA_NO_METRICS",
        "architecture": {
            "parameters": parameter_count(model),
            "raw_channels": RAW_CHANNELS,
            "input_features": INPUT_FEATURES,
            "hidden": HIDDEN,
            "synthetic_output_shape": list(prediction.shape),
            "synthetic_state_shape": list(states.shape),
            "synthetic_forward_backward_finite": True,
            "synthetic_gradient_nonzero": True,
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
    aug_sums, aug_sums2 = aug_sufficient_stats(dataset)
    train = np.flatnonzero(folds != 0)
    centers, scales, layer_rms = moments_for(
        train, error, sums, sums2, aug_sums, aug_sums2
    )
    torch.manual_seed(SEED + 40_000)
    model = AugWidenedTransportEnergyResidual(centers, scales, layer_rms)
    raw, sigma, weights, _ = load_batch(dataset, np.asarray([0]), error)
    if tuple(raw.shape) != (1, DEPTH, WIDTH, RAW_CHANNELS):
        raise AssertionError("smoke widened raw shape changed")
    prediction, states, energies = model(
        raw, sigma, weights, exposure=1.0, return_states=True
    )
    tensors = [prediction, states, energies]
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("smoke tensor is nonfinite")
    synthetic_loss = (
        prediction[:, FIRST_CORRECTED_LAYER:]
        / model.layer_rms[FIRST_CORRECTED_LAYER:][None, :, None]
    ).square().mean()
    synthetic_loss.backward()
    if not all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
    ):
        raise FloatingPointError("synthetic smoke gradient is missing or nonfinite")
    payload = {
        "node": NODE,
        "status": "PASS_SYNTHETIC_AND_ONE_INPUT_NO_TRUTH_METRICS",
        "launch_status": "held_after_sealed_smoke",
        "readiness": readiness,
        "fold_sizes": [int(np.sum(folds == fold)) for fold in range(FOLDS)],
        "architecture": {
            "parameters": parameter_count(model),
            "raw_channels": RAW_CHANNELS,
            "input_features": INPUT_FEATURES,
            "prediction_shape": list(prediction.shape),
            "finite": True,
            "finite_full_backward_gradient": True,
        },
        "cost": cost_record(model),
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
