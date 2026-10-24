#!/Users/matt/anaconda3/bin/python3
"""Independent literal terminal replay for frozen P2R0F2N.

This verifier does not import either frozen training runner.  It reconstructs
the frozen J and N forward equations from their four hash-bound checkpoints,
uses only the observer_fit feature/host/target bundle, and repeats the exact
batch-two outer-OOF scoring order.  It performs no training, selection access,
name join, network action, or graph edit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
M_DIR = P2 / "public_full688_backward_adjoint_state_cv_v1"
D_DIR = P2 / "public_full688_dynamic_multistate_residual_cv_v1"
LOADER_DIR = P2 / "public_full688_local_innovation_transport_v1"
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
TARGET_ROOT = P2 / "data_gate/sealed_role_target_bundle"
sys.path.insert(0, str(LOADER_DIR))
from dataset import DEPTH, LOCAL_FEATURES, WIDTH, PublicResidualDataset  # noqa: E402


SEED = 1_753_358_438
FOLDS = 4
FIRST = 2
J_CHANNELS = 4
J_HIDDEN = 96
N_RANK = 16
N_HIDDEN = 128
N_INPUTS = 36
N_HEAD = 34
N_LAYERS = 14
N_TRANSITIONS = 13
GRAM_FLOOR = 2.0**-10
POSITIVE_OFFSET = 2.0**-12
POSITIVE_MINIMUM = 2.0**-10
POSITIVE_MAXIMUM = 2.0**6
COHERENCE_FLOOR = 1.0e-4
TAIL_RANKS = [55, 71, 106, 317, 339, 458, 485, 507, 540, 575, 602]
FOLD_SIZES = [147, 172, 171, 198]
J_EXPOSURE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]

EXPECTED = {
    "preregistration": "b7d84a4630c0d7819e025e6dabe74e0f09f8e4c0c6271c7e67d34be3d4afee93",
    "run_manifest": "e81b7cb646a7a78998315961f5414d987a7a0d307c20a2b924f0960cfe1bca0f",
    "seal": "6b9856bb9e08473148b7d75952d70258e991c90b87284516d3fcd90a0ee9a1bb",
    "static_receipt": "f22cf6ef41321698d30a7582c5f31a2b8dd6f6b6113aa1d4b20304cbb6573b97",
    "smoke_receipt": "2f16540324677e5f6dde69fad8124c08b093cd4a896d0c8091231d9ba94fa613",
    "training_release": "5951a956784a5d6fbeeff294d14ea307e1f553684e494a4fb74fd7d9658453a7",
    "program": "6b7c28711ac4b95a5a9c34c5e930b5fb104ecdb75a7a97357cbb49f49cac7502",
    "cv_results": "37187da1fabdc84951a5bc68ef3f12fba1d4b940bc19027c8426f59f1b61be37",
    "loader": "babd7733966f7b94b358a5b30906bcf19c8cbe47eb49a8e9cb7e49f8f7f1e3c1",
    "shared_cache_manifest": "ab463ae4887afd51b3816cbe2275f2f49a9b6e860eeaf0d9d62d7743056d80ca",
    "shared_error": "1414515cfbb098e731e4f00d3f4ba7a310f17ac96e886846cd08aaaa0a679ca1",
    "shared_stats": "1648a140768cdc254a7bb6957dcb8e82f4271118bc64b552bb08de294ec62208",
    "feature_manifest": "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50",
    "feature_finalization": "ae455586e14b5039397bb514162e363560ef115320c76a4d553b526b4b494b17",
    "target_manifest": "2c0ea0bb09d0c86d46ca07fdde3563c4cda046a1fff152cdff26fde965e1f7e6",
    "observer_fit_target": "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7",
    "j_program": "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810",
    "j_cv_results": "70a0917258a2d656ab9593a67b3a04483af153d8ef59162be9570a6a62b9b5b4",
    "j_verification": "c50a8a6a59b28ab11c1e2db8ce14ce24181e8a7c9d6ecfc1ff31019825f27dcb",
    "m_cv_results": "1c8fc334b1ba20fba82a82ced63361c9ff42ad079bb9a7ec54a5b1c9d7e2ce7b",
    "m_decomposition": "8ee24497e1bd763ec1f4c001af4a0ae74b558c7ee888fddf351f2ad0249b7fdb",
}

N_RESULT_SHA = [
    "9b13b0634295e55e24cc912259c4b03d8b9f696e3610c59f791357ab46b7a90d",
    "a2749e6d2347fae5da0b5c5f92a268bcd37845b4930b6321ea74abe4855f269f",
    "f1a5fa10d9dc442092d350c7c59b0cb17f994569689b8d13dd3dffa413fd2093",
    "65dc47610a08b7bbc2aea854ec8b387433bb65919667282c40488d3cb5367339",
]
N_MODEL_SHA = [
    "b403f1e4d9f72c9caa30886e0695fe4c659c9512090df0f9f61b3d2ddec3eb49",
    "8427cccd9364af7dd3f69f40772ee2697875b4bf04d79bfef455331dc6052486",
    "87a5f633976456266c6fb5a74c77d972930bc992251de2296d14772fef531e7f",
    "20882e58d1080eb2aef9f0036671cc12ea61c8c741559d5ded5d6c39ddcede55",
]
J_RESULT_SHA = [
    "d96aa4e7357c07f2cbf9000f8dbfa223528e9ce39771867c6f557556f83861a6",
    "750355ee7dc1d1c515154af392c4061d5e9cc5569f8cb92bd95d9be82346ae92",
    "e77916590b63f77590d21709ff89d5ac0e034dad73d4da3db50051fc0687279d",
    "5a8208f7c691216b65e5e3a90553ac5ea335bdd29ac4083cc36deddd59cb94e0",
]
J_MODEL_SHA = [
    "5bf8a65a7a73ffc4ce06fa493c4e720ff0d3f06edb3f1186e7adf6bcf26396a2",
    "c26dbe6cd37ef96c268d6dc144d25d15793ea69b8dae135b6844c6809c21ec26",
    "8c1d4abf025b62e29391c2f59302ba204a83779f6eb11d2f11328295a23c8759",
    "760e70dd32b16f36fe2809f6bbe52669ec5dd0293b909995fdb743ce8e737cc3",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, *, atol: float = 1.0e-20) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2.0e-12, abs_tol=atol):
        raise AssertionError((left, right))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fold_of(mlp_id: int) -> int:
    payload = f"{SEED}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % FOLDS


def inner_fold_of(outer_fold: int, mlp_id: int) -> int:
    inner_seed = SEED ^ ((outer_fold + 1) * 0x9E3779B1)
    payload = f"{inner_seed}:{int(mlp_id)}".encode("ascii")
    value = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(value, "little") % 3


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


class FrozenJ(nn.Module):
    """Independent reconstruction of the frozen four-channel J observer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("centers", torch.empty(DEPTH, LOCAL_FEATURES - 1))
        self.register_buffer("scales", torch.empty(DEPTH, LOCAL_FEATURES - 1))
        self.register_buffer("layer_rms", torch.empty(DEPTH))
        self.trunk = nn.Sequential(
            nn.Linear(LOCAL_FEATURES + 4 * J_CHANNELS, J_HIDDEN),
            nn.SiLU(),
            nn.Linear(J_HIDDEN, J_HIDDEN),
            nn.SiLU(),
        )
        self.source_weight = nn.Parameter(torch.empty(DEPTH, J_HIDDEN, J_CHANNELS))
        self.source_bias = nn.Parameter(torch.empty(DEPTH, J_CHANNELS))
        self.gate_weight = nn.Parameter(torch.empty(DEPTH, J_HIDDEN, J_CHANNELS))
        self.gate_bias = nn.Parameter(torch.empty(DEPTH, J_CHANNELS))
        self.transport_delta = nn.Parameter(torch.empty(DEPTH, J_CHANNELS))
        self.mix_delta = nn.Parameter(torch.empty(DEPTH, J_CHANNELS, J_CHANNELS))

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
            messages / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
            -8.0,
            8.0,
        )
        log_energy = torch.clamp(
            torch.log1p(torch.clamp(energies / rms2, min=0.0)), 0.0, 8.0
        )
        coherence = torch.clamp(
            messages / torch.sqrt(energies + COHERENCE_FLOOR * rms2), -8.0, 8.0
        )
        return self.trunk(
            torch.cat(
                (
                    raw_layer[..., :1],
                    tail,
                    message_rms,
                    message_sigma,
                    log_energy,
                    coherence,
                ),
                dim=-1,
            )
        )

    def forward(
        self, raw: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = raw.shape[0]
        states = torch.zeros(batch, WIDTH, J_CHANNELS, dtype=raw.dtype)
        outputs: list[torch.Tensor] = []
        state_history: list[torch.Tensor] = []
        energy_history: list[torch.Tensor] = []
        identity = torch.eye(J_CHANNELS, dtype=raw.dtype)
        for layer in range(DEPTH):
            if layer < FIRST:
                states = torch.zeros_like(states)
                energies = torch.zeros_like(states)
                outputs.append(states[..., 0])
                state_history.append(states)
                energy_history.append(energies)
                continue
            if layer == FIRST:
                messages = torch.zeros_like(states)
                energies = torch.zeros_like(states)
            else:
                layer_weights = weights[:, layer]
                messages = torch.bmm(
                    states.transpose(1, 2), layer_weights
                ).transpose(1, 2)
                energies = torch.bmm(
                    states.square().transpose(1, 2), layer_weights.square()
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
        return (
            torch.stack(outputs, dim=1),
            torch.stack(state_history, dim=1),
            torch.stack(energy_history, dim=1),
        )


class FrozenN(nn.Module):
    """Independent reconstruction of the frozen rank-16 N observer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("centers", torch.empty(DEPTH, LOCAL_FEATURES - 1))
        self.register_buffer("scales", torch.empty(DEPTH, LOCAL_FEATURES - 1))
        self.register_buffer("layer_rms", torch.empty(DEPTH))
        self.encoder = nn.Sequential(
            nn.Linear(N_INPUTS, N_HIDDEN),
            nn.SiLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.SiLU(),
        )
        self.head_weight = nn.Parameter(torch.empty(N_LAYERS, N_HIDDEN, N_HEAD))
        self.head_bias = nn.Parameter(torch.empty(N_LAYERS, N_HEAD))
        self.factor_gain_delta = nn.Parameter(torch.empty(N_TRANSITIONS, N_RANK))
        self.factor_mix_delta = nn.Parameter(
            torch.empty(N_TRANSITIONS, N_RANK, N_RANK)
        )

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
            messages / torch.clamp(sigma_layer[..., None] * rms, min=2.0**-30),
            -8.0,
            8.0,
        )
        log_energy = torch.clamp(
            torch.log1p(torch.clamp(energies / rms2, min=0.0)), 0.0, 8.0
        )
        coherence = torch.clamp(
            messages / torch.sqrt(energies + COHERENCE_FLOOR * rms2), -8.0, 8.0
        )
        return torch.cat(
            (
                raw_layer[..., :1],
                tail,
                message_rms,
                message_sigma,
                log_energy,
                coherence,
            ),
            dim=-1,
        )

    def forward(
        self,
        raw: torch.Tensor,
        sigma: torch.Tensor,
        weights: torch.Tensor,
        j_states: torch.Tensor,
        j_energies: torch.Tensor,
    ) -> torch.Tensor:
        batch = raw.shape[0]
        factors = torch.zeros(batch, WIDTH, N_RANK, dtype=raw.dtype)
        outputs = [
            torch.zeros(batch, WIDTH, dtype=raw.dtype) for _ in range(FIRST)
        ]
        identity = torch.eye(N_RANK, dtype=raw.dtype)
        for layer in range(FIRST, DEPTH):
            local_layer = layer - FIRST
            if layer == FIRST:
                messages = torch.zeros(batch, WIDTH, J_CHANNELS, dtype=raw.dtype)
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
                transported_factors = raw[:, layer, :, 5, None] * transported_factors
            features = self.local_features(
                raw[:, layer], sigma[:, layer], messages, j_energies[:, layer], layer
            )
            hidden = self.encoder(features)
            head = torch.einsum(
                "bwh,ho->bwo", hidden, self.head_weight[local_layer]
            ) + self.head_bias[local_layer]
            source = torch.tanh(head[..., :N_RANK])
            log_lambda_local = head[..., N_RANK : 2 * N_RANK]
            log_diagonal = head[..., 2 * N_RANK]
            score = torch.tanh(head[..., 2 * N_RANK + 1])
            if layer == FIRST:
                candidate = source
            else:
                transition = layer - FIRST - 1
                gain = 1.0 + 0.25 * torch.tanh(self.factor_gain_delta[transition])
                mixing = identity + 0.10 * torch.tanh(
                    self.factor_mix_delta[transition]
                )
                candidate = torch.einsum(
                    "bwr,rs->bws", gain * transported_factors, mixing
                ) + source
            gram = torch.bmm(candidate.transpose(1, 2), candidate) / float(WIDTH)
            gram = gram + GRAM_FLOOR * identity[None]
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
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
        return torch.stack(outputs, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def assert_state_finite(state: dict[str, torch.Tensor]) -> None:
    if not state:
        raise AssertionError("empty state_dict")
    for key, value in state.items():
        if value.dtype != torch.float32:
            raise AssertionError((key, value.dtype))
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"nonfinite checkpoint tensor: {key}")


def validate_static_bindings() -> tuple[dict[str, object], dict[str, object]]:
    paths = {
        "preregistration": HERE / "PREREGISTRATION.json",
        "run_manifest": HERE / "RUN_MANIFEST.json",
        "seal": HERE / "PREREGISTRATION_SEAL.json",
        "static_receipt": HERE / "STATIC_RECEIPT.json",
        "smoke_receipt": HERE / "SMOKE_RECEIPT.json",
        "training_release": HERE / "TRAINING_RELEASE.json",
        "program": HERE / "run_experiment.py",
        "cv_results": HERE / "CV_RESULTS.json",
        "loader": LOADER_DIR / "dataset.py",
        "shared_cache_manifest": D_DIR / "cache/MANIFEST.json",
        "shared_error": D_DIR / "cache/patched_error.npy",
        "shared_stats": D_DIR / "cache/raw_sufficient_stats.npz",
        "feature_manifest": FEATURE_ROOT / "manifest.json",
        "feature_finalization": FEATURE_ROOT / "FINALIZATION_RECEIPT_V3.json",
        "target_manifest": TARGET_ROOT / "manifest.json",
        "observer_fit_target": TARGET_ROOT / "observer_fit.parquet",
        "j_program": J_DIR / "run_experiment.py",
        "j_cv_results": J_DIR / "CV_RESULTS.json",
        "j_verification": J_DIR / "VERIFICATION.json",
        "m_cv_results": M_DIR / "CV_RESULTS.json",
        "m_decomposition": M_DIR / "M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json",
    }
    for key, path in paths.items():
        actual = sha256(path)
        if actual != EXPECTED[key]:
            raise AssertionError((key, actual, EXPECTED[key]))
    seal = read_json(HERE / "PREREGISTRATION_SEAL.json")
    release = read_json(HERE / "TRAINING_RELEASE.json")
    seal_expected = {
        "preregistration_sha256": EXPECTED["preregistration"],
        "program_sha256": EXPECTED["program"],
        "run_manifest_sha256": EXPECTED["run_manifest"],
        "static_receipt_sha256": EXPECTED["static_receipt"],
        "loader_sha256": EXPECTED["loader"],
        "shared_cache_manifest_sha256": EXPECTED["shared_cache_manifest"],
        "shared_error_sha256": EXPECTED["shared_error"],
        "shared_stats_sha256": EXPECTED["shared_stats"],
        "feature_store_manifest_sha256": EXPECTED["feature_manifest"],
        "feature_store_finalization_sha256": EXPECTED["feature_finalization"],
        "target_bundle_manifest_sha256": EXPECTED["target_manifest"],
        "observer_fit_target_parquet_sha256": EXPECTED["observer_fit_target"],
        "j_program_sha256": EXPECTED["j_program"],
        "j_cv_results_sha256": EXPECTED["j_cv_results"],
        "j_verification_sha256": EXPECTED["j_verification"],
        "m_decomposition_sha256": EXPECTED["m_decomposition"],
        "j_outer_model_sha256": J_MODEL_SHA,
        "j_outer_result_sha256": J_RESULT_SHA,
    }
    if seal.get("schema") != "public-full688-rank16-cross-unit-covariance-cv-seal-v1":
        raise AssertionError("seal schema")
    if seal.get("status") != "frozen-before-any-n-metrics":
        raise AssertionError("seal status")
    for key, value in seal_expected.items():
        if seal.get(key) != value:
            raise AssertionError(("seal", key))
    if seal["launch_state_at_seal"] != {
        "smoke_receipt_present": False,
        "training_release_present": False,
        "fold_artifacts_present": False,
        "cv_results_present": False,
        "n_metrics_existed": False,
    }:
        raise AssertionError("launch state at seal")
    release_expected = {
        "status": "authorized-after-sealed-smoke-by-root",
        "root_release": True,
        "preregistration_sha256": EXPECTED["preregistration"],
        "program_sha256": EXPECTED["program"],
        "run_manifest_sha256": EXPECTED["run_manifest"],
        "seal_sha256": EXPECTED["seal"],
        "smoke_receipt_sha256": EXPECTED["smoke_receipt"],
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "root_release": True,
    }
    for key, value in release_expected.items():
        if release.get(key) != value:
            raise AssertionError(("release", key))
    return seal, release


def validate_feature_manifests(dataset: PublicResidualDataset) -> str:
    manifest = read_json(FEATURE_ROOT / "manifest.json")
    if manifest.get("status") != "SEALED_TARGET_BLIND_FEATURE_STORE":
        raise AssertionError("feature manifest status")
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    if len(entries) != 688:
        raise AssertionError("feature manifest coverage")
    ids_digest = hashlib.sha256(
        ("\n".join(str(int(value)) for value in dataset.mlp_ids) + "\n").encode(
            "ascii"
        )
    ).hexdigest()
    if ids_digest != manifest["roles"]["observer_fit"]["ordered_mlp_ids_sha256"]:
        raise AssertionError("ordered observer_fit IDs")
    canonical_rows: list[dict[str, int | str]] = []
    for rank, entry in enumerate(entries):
        expected = {
            "role_rank": rank,
            "mlp_id": int(dataset.mlp_ids[rank]),
        }
        for key, value in expected.items():
            if entry.get(key) != value:
                raise AssertionError(("feature entry", rank, key))
        path = FEATURE_ROOT / str(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise AssertionError(("feature row manifest", rank))
        canonical_rows.append(
            {
                "role_rank": rank,
                "mlp_id": int(dataset.mlp_ids[rank]),
                "sha256": str(entry["sha256"]),
            }
        )
    return canonical_sha256(canonical_rows)


def validate_inner_firewall(
    record: dict[str, object], outer_fold: int, mlp_ids: np.ndarray
) -> None:
    outer_train = set(int(value) for value in record["train_role_ranks"])
    outer_valid = set(int(value) for value in record["valid_role_ranks"])
    seen: set[int] = set()
    receipts = record["inner_stacking"]["receipts"]
    if len(receipts) != 3 or record["inner_stacking"]["outer_validation_rows_used"] != 0:
        raise AssertionError((outer_fold, "inner receipt count/firewall"))
    for inner_fold, receipt in enumerate(receipts):
        if receipt["inner_fold"] != inner_fold:
            raise AssertionError((outer_fold, inner_fold, "identity"))
        valid = set(int(value) for value in receipt["valid_role_ranks"])
        expected = {
            rank
            for rank in outer_train
            if inner_fold_of(outer_fold, int(mlp_ids[rank])) == inner_fold
        }
        if valid != expected or seen.intersection(valid):
            raise AssertionError((outer_fold, inner_fold, "partition"))
        if valid.intersection(outer_valid):
            raise AssertionError((outer_fold, inner_fold, "outer leakage"))
        if receipt["train_networks"] != len(outer_train - valid):
            raise AssertionError((outer_fold, inner_fold, "train count"))
        if receipt["valid_networks"] != len(valid):
            raise AssertionError((outer_fold, inner_fold, "valid count"))
        if receipt["outer_validation_rows_in_train"] != 0:
            raise AssertionError((outer_fold, inner_fold, "leakage receipt"))
        if receipt["fixed_final_epoch"] != 6:
            raise AssertionError((outer_fold, inner_fold, "checkpoint"))
        seen.update(valid)
    if seen != outer_train:
        raise AssertionError((outer_fold, "inner coverage"))
    trace = record["inner_stacking"]["trace"]
    if len(trace) != 18:
        raise AssertionError((outer_fold, "inner trace length"))
    for inner_fold in range(3):
        rows = [row for row in trace if row["inner_fold"] == inner_fold]
        if [row["epoch"] for row in rows] != list(range(1, 7)):
            raise AssertionError((outer_fold, inner_fold, "epochs"))
        if [row["exposure"] for row in rows] != J_EXPOSURE:
            raise AssertionError((outer_fold, inner_fold, "exposure"))


def load_batch(
    dataset: PublicResidualDataset,
    ranks: np.ndarray,
    error: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [dataset.load(int(rank)) for rank in ranks]
    for rank, row in zip(ranks, rows, strict=True):
        independently_derived = np.asarray(row.target) - np.asarray(row.host)
        if not np.array_equal(independently_derived, np.asarray(error[int(rank)])):
            raise AssertionError((int(rank), "sealed target-host error mismatch"))
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


def score_fold(
    fold: int,
    dataset: PublicResidualDataset,
    error: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    folds: np.ndarray,
    seal: dict[str, object],
    cv: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, int | float]]]:
    result_path = HERE / "folds" / f"fold_{fold}.json"
    model_path = HERE / "folds" / f"fold_{fold}.pt"
    j_result_path = J_DIR / "folds" / f"fold_{fold}.json"
    j_model_path = J_DIR / "folds" / f"fold_{fold}.pt"
    if sha256(result_path) != N_RESULT_SHA[fold]:
        raise AssertionError((fold, "N result hash"))
    if sha256(model_path) != N_MODEL_SHA[fold]:
        raise AssertionError((fold, "N model hash"))
    if sha256(j_result_path) != J_RESULT_SHA[fold]:
        raise AssertionError((fold, "J result hash"))
    if sha256(j_model_path) != J_MODEL_SHA[fold]:
        raise AssertionError((fold, "J model hash"))
    record = read_json(result_path)
    summary = cv["fold_results"][fold]
    valid = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    if len(valid) != FOLD_SIZES[fold]:
        raise AssertionError((fold, "size"))
    expected_header = {
        "schema": "public-full688-rank16-cross-unit-covariance-fold-v1",
        "status": "complete",
        "node": "P2R0F2N",
        "fold": fold,
        "valid_networks": len(valid),
        "train_networks": len(train),
        "valid_role_ranks": valid.tolist(),
        "train_role_ranks": train.tolist(),
        "valid_mlp_ids": dataset.mlp_ids[valid].astype(int).tolist(),
        "model_sha256": N_MODEL_SHA[fold],
        "outer_j_model_sha256": J_MODEL_SHA[fold],
        "preregistration_sha256": EXPECTED["preregistration"],
        "program_sha256": EXPECTED["program"],
        "run_manifest_sha256": EXPECTED["run_manifest"],
        "seal_sha256": EXPECTED["seal"],
        "training_release_sha256": EXPECTED["training_release"],
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
    }
    for key, value in expected_header.items():
        if record.get(key) != value:
            raise AssertionError((fold, key))
    if record["readiness"]["role"] != "observer_fit":
        raise AssertionError((fold, "role"))
    if record["readiness"]["data_derived_tail_ranks"] != TAIL_RANKS:
        raise AssertionError((fold, "tail roster"))
    if record["readiness"]["repaired_tail_ranks"] != TAIL_RANKS:
        raise AssertionError((fold, "tail repairs"))
    if record["readiness"]["unpatched_tail_ranks"] != []:
        raise AssertionError((fold, "unpatched tail"))
    if record["training"]["epochs"] != 8:
        raise AssertionError((fold, "N epochs"))
    if [row["epoch"] for row in record["training"]["trace"]] != list(range(1, 9)):
        raise AssertionError((fold, "N fixed final trace"))
    validate_inner_firewall(record, fold, dataset.mlp_ids)

    n_artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    j_artifact = torch.load(j_model_path, map_location="cpu", weights_only=True)
    n_header = {
        "schema": "public-full688-rank16-cross-unit-covariance-fold-model-v1",
        "node": "P2R0F2N",
        "fold": fold,
        "preregistration_sha256": EXPECTED["preregistration"],
        "program_sha256": EXPECTED["program"],
        "run_manifest_sha256": EXPECTED["run_manifest"],
        "seal_sha256": EXPECTED["seal"],
        "training_release_sha256": EXPECTED["training_release"],
        "outer_j_model_sha256": J_MODEL_SHA[fold],
    }
    for key, value in n_header.items():
        if n_artifact.get(key) != value:
            raise AssertionError((fold, "N PT", key))
    if j_artifact.get("schema") != "public-full688-transport-energy-coherence-fold-model-v1":
        raise AssertionError((fold, "J PT schema"))
    if j_artifact.get("node") != "P2R0F2J" or j_artifact.get("fold") != fold:
        raise AssertionError((fold, "J PT identity"))
    assert_state_finite(n_artifact["state_dict"])
    assert_state_finite(j_artifact["state_dict"])

    centers, scales, layer_rms = moments_for(train, error, sums, sums2)
    for state in (n_artifact["state_dict"], j_artifact["state_dict"]):
        if not np.array_equal(state["centers"].numpy(), centers):
            raise AssertionError((fold, "centers"))
        if not np.array_equal(state["scales"].numpy(), scales):
            raise AssertionError((fold, "scales"))
        if not np.array_equal(state["layer_rms"].numpy(), layer_rms):
            raise AssertionError((fold, "layer_rms"))

    j_model = FrozenJ()
    j_model.load_state_dict(j_artifact["state_dict"], strict=True)
    n_model = FrozenN()
    n_model.load_state_dict(n_artifact["state_dict"], strict=True)
    if parameter_count(j_model) != 25_600 or parameter_count(n_model) != 86_188:
        raise AssertionError((fold, parameter_count(j_model), parameter_count(n_model)))
    j_model.eval()
    n_model.eval()

    n_sse = np.zeros(DEPTH, dtype=np.float64)
    j_sse = np.zeros(DEPTH, dtype=np.float64)
    per_network: list[dict[str, int | float]] = []
    maximum_delta = 0.0
    all_finite = True
    with torch.no_grad():
        for offset in range(0, len(valid), 2):
            group = valid[offset : offset + 2]
            raw, sigma, weights, truth = load_batch(dataset, group, error)
            j_prediction, j_states, j_energies = j_model(raw, sigma, weights)
            delta = n_model(raw, sigma, weights, j_states, j_energies)
            prediction = j_prediction + delta
            all_finite = all_finite and bool(torch.isfinite(prediction).all())
            maximum_delta = max(maximum_delta, float(delta.abs().max()))
            n_squared = (prediction - truth).double().square()
            j_squared = (j_prediction - truth).double().square()
            n_sse += n_squared.sum(dim=(0, 2)).numpy()
            j_sse += j_squared.sum(dim=(0, 2)).numpy()
            n_final = n_squared[:, -1].mean(dim=1).numpy()
            j_final = j_squared[:, -1].mean(dim=1).numpy()
            for rank, n_value, j_value in zip(group, n_final, j_final, strict=True):
                per_network.append(
                    {
                        "role_rank": int(rank),
                        "mlp_id": int(dataset.mlp_ids[int(rank)]),
                        "fold": fold,
                        "autonomous_final_mse": float(n_value),
                        "frozen_j_final_mse": float(j_value),
                    }
                )
    denominator = len(valid) * WIDTH
    n_layer = n_sse / denominator
    j_layer = j_sse / denominator
    host_layer = np.mean(
        np.square(error[valid].astype(np.float64)), axis=(0, 2), dtype=np.float64
    )
    if not all_finite or record["autonomous"]["finite_full_rollout"] is not True:
        raise AssertionError((fold, "finite"))
    for actual, saved in zip(n_layer, record["autonomous"]["layer_mse"], strict=True):
        close(actual, saved)
    for actual, saved in zip(
        j_layer, record["autonomous"]["frozen_j_layer_mse"], strict=True
    ):
        close(actual, saved)
    for actual, saved in zip(host_layer, record["host"]["layer_mse"], strict=True):
        close(actual, saved)
    close(n_layer[-1], record["autonomous"]["final_mse"])
    close(j_layer[-1], record["autonomous"]["frozen_j_final_mse"])
    close(host_layer[-1], record["host"]["final_mse"])
    close(maximum_delta, record["autonomous"]["maximum_absolute_added_delta"])
    saved_rows = record["autonomous"]["per_network"]
    if len(saved_rows) != len(per_network):
        raise AssertionError((fold, "row count"))
    for replay, saved in zip(per_network, saved_rows, strict=True):
        if replay["role_rank"] != saved["role_rank"]:
            raise AssertionError((fold, "row rank"))
        if replay["mlp_id"] != saved["mlp_id"]:
            raise AssertionError((fold, "row mlp_id"))
        close(replay["autonomous_final_mse"], saved["final_mse"])
    fold_max = max(row["autonomous_final_mse"] for row in per_network)
    expected_summary = {
        "fold": fold,
        "valid_networks": len(valid),
        "result_sha256": N_RESULT_SHA[fold],
        "model_sha256": N_MODEL_SHA[fold],
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise AssertionError((fold, "CV summary", key))
    close(summary["host_final_mse"], host_layer[-1])
    close(summary["autonomous_final_mse"], n_layer[-1])
    close(summary["frozen_j_final_mse"], j_layer[-1])
    close(summary["maximum_per_network_final_mse"], fold_max)
    return (
        {
            "fold": fold,
            "networks": len(valid),
            "host_layer_mse": host_layer.tolist(),
            "frozen_j_layer_mse": j_layer.tolist(),
            "autonomous_layer_mse": n_layer.tolist(),
            "host_final_mse": float(host_layer[-1]),
            "frozen_j_final_mse": float(j_layer[-1]),
            "autonomous_final_mse": float(n_layer[-1]),
            "maximum_per_network_final_mse": float(fold_max),
            "maximum_absolute_added_delta": float(maximum_delta),
        },
        per_network,
    )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED & 0xFFFF_FFFF)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seal, release = validate_static_bindings()
    cv = read_json(HERE / "CV_RESULTS.json")
    cv_header = {
        "schema": "public-full688-rank16-cross-unit-covariance-cv-v1",
        "status": "complete",
        "node": "P2R0F2N",
        "folds": 4,
        "group_key": "mlp_id",
        "observer_fit_networks": 688,
        "observer_selection_targets_opened": 0,
        "public_name_joins": 0,
        "network_access": 0,
        "preregistration_sha256": EXPECTED["preregistration"],
        "program_sha256": EXPECTED["program"],
        "run_manifest_sha256": EXPECTED["run_manifest"],
        "seal_sha256": EXPECTED["seal"],
        "training_release_sha256": EXPECTED["training_release"],
    }
    for key, value in cv_header.items():
        if cv.get(key) != value:
            raise AssertionError(("CV", key))
    if len(cv["fold_results"]) != FOLDS:
        raise AssertionError("CV fold count")

    dataset = PublicResidualDataset(role="observer_fit")
    if len(dataset.mlp_ids) != 688 or dataset.targets.shape != (688, DEPTH, WIDTH):
        raise AssertionError("observer_fit target schema")
    row_manifest_digest = validate_feature_manifests(dataset)
    error = np.load(D_DIR / "cache/patched_error.npy", mmap_mode="r", allow_pickle=False)
    stats = np.load(D_DIR / "cache/raw_sufficient_stats.npz", allow_pickle=False)
    sums = stats["sum"]
    sums2 = stats["sum2"]
    if error.dtype != np.float32 or error.shape != (688, DEPTH, WIDTH):
        raise AssertionError("shared error schema")
    if sums.dtype != np.float64 or sums.shape != (688, DEPTH, LOCAL_FEATURES - 1):
        raise AssertionError("shared sum schema")
    if sums2.dtype != np.float64 or sums2.shape != sums.shape:
        raise AssertionError("shared sum2 schema")
    folds = np.asarray([fold_of(int(value)) for value in dataset.mlp_ids], dtype=np.int64)
    if [int(np.sum(folds == fold)) for fold in range(FOLDS)] != FOLD_SIZES:
        raise AssertionError("outer fold assignment")

    fold_metrics: list[dict[str, object]] = []
    rows: list[dict[str, int | float]] = []
    for fold in range(FOLDS):
        metrics, fold_rows = score_fold(
            fold, dataset, error, sums, sums2, folds, seal, cv
        )
        fold_metrics.append(metrics)
        rows.extend(fold_rows)
    rows.sort(key=lambda row: int(row["role_rank"]))
    if [int(row["role_rank"]) for row in rows] != list(range(688)):
        raise AssertionError("OOF row coverage")
    if len({int(row["mlp_id"]) for row in rows}) != 688:
        raise AssertionError("OOF group uniqueness")

    total = sum(int(item["networks"]) for item in fold_metrics)
    if total != 688:
        raise AssertionError("OOF total")
    host_layer = np.average(
        np.asarray([item["host_layer_mse"] for item in fold_metrics]),
        axis=0,
        weights=np.asarray([item["networks"] for item in fold_metrics]),
    )
    j_layer = np.average(
        np.asarray([item["frozen_j_layer_mse"] for item in fold_metrics]),
        axis=0,
        weights=np.asarray([item["networks"] for item in fold_metrics]),
    )
    n_layer = np.average(
        np.asarray([item["autonomous_layer_mse"] for item in fold_metrics]),
        axis=0,
        weights=np.asarray([item["networks"] for item in fold_metrics]),
    )
    for actual, saved in zip(host_layer, cv["oof"]["host_layer_mse"], strict=True):
        close(actual, saved)
    for actual, saved in zip(n_layer, cv["oof"]["autonomous_layer_mse"], strict=True):
        close(actual, saved)
    close(host_layer[-1], cv["oof"]["host_final_mse"])
    close(n_layer[-1], cv["oof"]["autonomous_final_mse"])
    close(n_layer[-1] / host_layer[-1], cv["oof"]["autonomous_ratio_to_host"])

    worst = max(rows, key=lambda row: float(row["autonomous_final_mse"]))
    close(worst["autonomous_final_mse"], cv["oof"]["maximum_per_network_final_mse"])
    if worst["role_rank"] != cv["oof"]["maximum_role_rank"]:
        raise AssertionError("worst role rank")
    if worst["mlp_id"] != cv["oof"]["maximum_mlp_id"]:
        raise AssertionError("worst mlp id")

    recomputed_gate = {
        "finite_all_688_outer_oof_rollouts": True,
        "oof_final_mse_below_1e_8": bool(n_layer[-1] < 1.0e-8),
        "every_outer_fold_below_1_1e_8": bool(
            all(float(item["autonomous_final_mse"]) < 1.1e-8 for item in fold_metrics)
        ),
        "maximum_per_network_below_1e_7": bool(
            float(worst["autonomous_final_mse"]) < 1.0e-7
        ),
        "observer_selection_targets_opened_zero": True,
    }
    recomputed_gate["conjunctive_pass"] = bool(all(recomputed_gate.values()))
    if cv["decision_gate"] != recomputed_gate:
        raise AssertionError((cv["decision_gate"], recomputed_gate))

    cost = cv["cost"]
    if cost["added_covariance_observer_upper"] != 1_226_926_080:
        raise AssertionError("N cost")
    if cost["j_plus_covariance_observer_correction_upper"] != 2_278_582_272:
        raise AssertionError("J+N cost")
    if cost["native_total_upper"] != 2_133_970_719_993:
        raise AssertionError("native total cost")
    if cost["headroom_below_exact_2pow41"] != (1 << 41) - 2_133_970_719_993:
        raise AssertionError("headroom")
    if not cost["within_exact_2pow41"] or not cost["within_phase2_budget"]:
        raise AssertionError("cost admission")

    j_cv = read_json(J_DIR / "CV_RESULTS.json")
    m_cv = read_json(M_DIR / "CV_RESULTS.json")
    m_decomposition = read_json(
        M_DIR / "M_VS_J_OOF_DECOMPOSITION_AND_ONE_SUCCESSOR.json"
    )
    if m_decomposition["bindings"]["j_cv_results_sha256"] != EXPECTED["j_cv_results"]:
        raise AssertionError("M decomposition J binding")
    if m_decomposition["bindings"]["m_cv_results_sha256"] != EXPECTED["m_cv_results"]:
        raise AssertionError("M decomposition M binding")
    j_frozen = float(j_cv["oof"]["autonomous_final_mse"])
    m_frozen = float(m_cv["oof"]["autonomous_final_mse"])
    close(j_layer[-1], j_frozen)
    close(m_decomposition["headline"]["j_oof_final_mse"], j_frozen)
    close(m_decomposition["headline"]["m_oof_final_mse"], m_frozen)

    n_final = float(n_layer[-1])
    row_wins = sum(
        float(row["autonomous_final_mse"]) < float(row["frozen_j_final_mse"])
        for row in rows
    )
    row_ties = sum(
        float(row["autonomous_final_mse"]) == float(row["frozen_j_final_mse"])
        for row in rows
    )
    row_losses = 688 - row_wins - row_ties
    row_digest = canonical_sha256(rows)
    verification = {
        "schema": "public-full688-rank16-cross-unit-covariance-literal-verification-v1",
        "status": "PASS",
        "scope": (
            "Literal batch-two replay of all 688 observer_fit rows from four frozen N "
            "and four seal-bound outer-J checkpoints; exact host/target cache, receipt, "
            "aggregation, gate, nested-firewall, and comparison verification"
        ),
        "bindings": {
            **EXPECTED,
            "verifier_sha256": sha256(Path(__file__)),
            "n_fold_result_sha256": N_RESULT_SHA,
            "n_fold_model_sha256": N_MODEL_SHA,
            "j_outer_result_sha256": J_RESULT_SHA,
            "j_outer_model_sha256": J_MODEL_SHA,
            "observer_fit_row_manifest_binding_digest": row_manifest_digest,
            "recomputed_row_metrics_sha256": row_digest,
        },
        "firewalls": {
            "role": "observer_fit",
            "observer_fit_rows_replayed": 688,
            "observer_selection_targets_opened": 0,
            "public_name_joins": 0,
            "network_access": 0,
            "training_steps": 0,
            "graph_edits": 0,
        },
        "checks": {
            "all_static_and_terminal_hash_bindings_exact": True,
            "all_688_feature_row_manifests_hash_bound": True,
            "sealed_error_bit_equal_to_observer_fit_target_minus_host_for_all_rows": True,
            "outer_validation_sets_disjoint_and_cover_688_exactly_once": True,
            "each_outer_training_set_is_validation_complement": True,
            "all_12_inner_validation_sets_partition_their_outer_training_complements": True,
            "all_outer_validation_rows_excluded_from_all_inner_j_training_sets": True,
            "all_inner_j_and_n_checkpoints_fixed_final_epoch": True,
            "all_checkpoint_tensors_float32_and_finite": True,
            "all_checkpoint_moment_buffers_exactly_recomputed": True,
            "j_forward_equation_independently_reconstructed": True,
            "n_forward_equation_independently_reconstructed": True,
            "all_688_saved_row_final_metrics_recomputed": True,
            "all_four_fold_layer_and_final_metrics_recomputed": True,
            "all_16_oof_layer_metrics_recomputed": True,
            "all_gate_booleans_recomputed": True,
            "cost_arithmetic_recomputed": True,
        },
        "metrics": {
            "oof_networks": 688,
            "host_layer_mse": host_layer.tolist(),
            "frozen_j_layer_mse": j_layer.tolist(),
            "rank16_n_layer_mse": n_layer.tolist(),
            "host_final_mse": float(host_layer[-1]),
            "frozen_j_final_mse": j_frozen,
            "frozen_m_final_mse": m_frozen,
            "rank16_n_final_mse": n_final,
            "n_minus_j": n_final - j_frozen,
            "n_fractional_improvement_vs_j": (j_frozen - n_final) / j_frozen,
            "n_minus_m": n_final - m_frozen,
            "n_fractional_improvement_vs_m": (m_frozen - n_final) / m_frozen,
            "n_factor_above_1e_8": n_final / 1.0e-8,
            "fractional_reduction_still_required_to_1e_8": 1.0 - 1.0e-8 / n_final,
            "row_wins_vs_j": row_wins,
            "row_ties_vs_j": row_ties,
            "row_losses_vs_j": row_losses,
            "maximum_role_rank": int(worst["role_rank"]),
            "maximum_mlp_id": int(worst["mlp_id"]),
            "maximum_per_network_final_mse": float(worst["autonomous_final_mse"]),
            "fold_metrics": fold_metrics,
        },
        "decision_gate": recomputed_gate,
        "cost": {
            "native_total_upper": cost["native_total_upper"],
            "exact_scoring_budget": cost["exact_scoring_budget"],
            "headroom_below_exact_2pow41": cost["headroom_below_exact_2pow41"],
            "phase2_budget": cost["phase2_budget"],
            "within_exact_2pow41": cost["within_exact_2pow41"],
            "within_phase2_budget": cost["within_phase2_budget"],
            "literal_meter_not_authorized_because_accuracy_gate_failed": True,
        },
    }
    print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
