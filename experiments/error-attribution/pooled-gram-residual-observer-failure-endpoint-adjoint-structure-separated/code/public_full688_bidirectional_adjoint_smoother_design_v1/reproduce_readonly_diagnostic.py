#!/Users/matt/anaconda3/bin/python3
"""Reproduce the bounded P2R0F2M endpoint-adjoint capacity diagnostic.

This program is deliberately read-only.  It replays the already-frozen J OOF
models on a target-independent prefix panel plus the four previously declared
hard role ranks.  Targets are used only after the target-free J prediction and
target-free Jacobian basis exist, to measure residual alignment/capacity.  It
does not train, select a checkpoint, inspect any selection role, or write an
artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_DIR = P2 / "public_full688_transport_energy_coherence_cv_v1"
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
ERROR_CACHE = (
    P2
    / "public_full688_dynamic_multistate_residual_cv_v1/cache/patched_error.npy"
)
EXPECTED_RESULT = HERE / "DIAGNOSTIC_RESULT.json"

PANEL_PREFIX = tuple(range(64))
HARD_RANKS = (201, 310, 599, 643)
PANEL = PANEL_PREFIX + HARD_RANKS
SEED = 1_753_358_493
MODES = 8

SOURCE_BINDINGS = {
    J_DIR / "CV_RESULTS.json":
        "70a0917258a2d656ab9593a67b3a04483af153d8ef59162be9570a6a62b9b5b4",
    J_DIR / "run_experiment.py":
        "1aedf41e6733ffb2a62a9881dc28945f97a4233372b272f6df6204cd525b8810",
    J_DIR / "folds/fold_0.pt":
        "5bf8a65a7a73ffc4ce06fa493c4e720ff0d3f06edb3f1186e7adf6bcf26396a2",
    J_DIR / "folds/fold_1.pt":
        "c26dbe6cd37ef96c268d6dc144d25d15793ea69b8dae135b6844c6809c21ec26",
    J_DIR / "folds/fold_2.pt":
        "8c1d4abf025b62e29391c2f59302ba204a83779f6eb11d2f11328295a23c8759",
    J_DIR / "folds/fold_3.pt":
        "760e70dd32b16f36fe2809f6bbe52669ec5dd0293b909995fdb743ce8e737cc3",
    ERROR_CACHE:
        "1414515cfbb098e731e4f00d3f4ba7a310f17ac96e886846cd08aaaa0a679ca1",
    (
        P2
        / "learned_residual_observer_design/v4_feature_store_v3"
        / "sealed_target_blind_feature_store_v3/candidate_manifest.json"
    ):
        "880b7578e97de3b2a20e2f4ef68dbd1c9d1d24c9e3705b5bab04e0600afca716",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bindings() -> None:
    for path, expected in SOURCE_BINDINGS.items():
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"source binding changed: {path}: {actual}")
    for path in (HERE, J_DIR, FEATURE_ROOT, ERROR_CACHE):
        if "observer_selection" in str(path):
            raise AssertionError("forbidden role path")


def orthonormal_rows(values: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(values.T, mode="reduced")
    return np.asarray(q.T, dtype=np.float32)


def import_j_module():
    spec = importlib.util.spec_from_file_location(
        "p2r0f2j_readonly_replay", J_DIR / "run_experiment.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot import frozen J source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(values: np.ndarray, hard: np.ndarray) -> dict[str, object]:
    ordinary = values[~hard]
    difficult = values[hard]
    return {
        "ordinary_mean": float(np.mean(ordinary)),
        "ordinary_median": float(np.median(ordinary)),
        "hard_mean": float(np.mean(difficult)),
        "hard_values": [float(value) for value in difficult],
    }


def main() -> None:
    verify_bindings()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    jrun = import_j_module()
    error = np.load(ERROR_CACHE, mmap_mode="r", allow_pickle=False)
    if error.dtype != np.float32 or error.shape != (688, 16, 1024):
        raise AssertionError("error cache schema changed")

    rank_fold: dict[int, int] = {}
    j_mse = np.empty(688, dtype=np.float64)
    for fold in range(4):
        record = json.loads(
            (J_DIR / f"folds/fold_{fold}.json").read_text(encoding="utf-8")
        )
        for row in record["autonomous"]["per_network"]:
            rank = int(row["role_rank"])
            rank_fold[rank] = fold
            j_mse[rank] = float(row["final_mse"])

    models = {}
    for fold in range(4):
        artifact = torch.load(
            J_DIR / f"folds/fold_{fold}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = artifact["state_dict"]
        model = jrun.TransportEnergyMultiStateResidual(
            state["centers"].numpy(),
            state["scales"].numpy(),
            state["layer_rms"].numpy(),
        )
        model.load_state_dict(state)
        model.eval()
        models[fold] = model

    rng = np.random.default_rng(SEED)
    omega = np.where(
        rng.integers(0, 2, size=(MODES, 1024), dtype=np.int8) > 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    omega = orthonormal_rows(omega)

    rows: list[dict[str, float | int]] = []
    for rank in PANEL:
        feature_dir = FEATURE_ROOT / f"{rank:04d}"
        raw = np.array(
            np.load(feature_dir / "cp1792_raw_locals.npy", mmap_mode="r"),
            copy=True,
        )
        sigma = np.array(
            np.load(feature_dir / "cp1792_sigma.npy", mmap_mode="r"),
            copy=True,
        )
        weights = np.array(
            np.load(feature_dir / "weights.npy", mmap_mode="r"), copy=True
        )
        if raw.shape != (16, 1024, 20) or sigma.shape != (16, 1024):
            raise AssertionError(f"rank {rank}: tape schema changed")
        if weights.shape != (16, 1024, 1024):
            raise AssertionError(f"rank {rank}: weight schema changed")

        with torch.no_grad():
            prediction = models[rank_fold[rank]](
                torch.from_numpy(raw[None]),
                torch.from_numpy(sigma[None]),
                torch.from_numpy(weights[None]),
                exposure=1.0,
            )[0].numpy()
        residual = prediction - np.asarray(error[rank], dtype=np.float32)

        def forward(values: np.ndarray) -> np.ndarray:
            result = values
            for layer in range(8, 16):
                result = (result @ weights[layer]) * raw[layer, :, 5][None, :]
            return result

        def backward(values: np.ndarray) -> np.ndarray:
            result = values
            for layer in range(15, 7, -1):
                result = (
                    result * raw[layer, :, 5][None, :]
                ) @ weights[layer].T
            return result

        # One fixed randomized subspace iteration for the dominant left space.
        first_left = orthonormal_rows(forward(omega))
        right = orthonormal_rows(backward(first_left))
        left_raw = forward(right)
        dominant_left = orthonormal_rows(left_raw)
        gain2 = np.sum(left_raw * left_raw, axis=1)

        endpoint = residual[15]
        endpoint_energy = float(np.dot(endpoint, endpoint))
        dominant_coefficients = dominant_left @ endpoint
        random_coefficients = omega @ endpoint
        dominant_fraction = float(
            np.dot(dominant_coefficients, dominant_coefficients) / endpoint_energy
        )
        random_fraction = float(
            np.dot(random_coefficients, random_coefficients) / endpoint_energy
        )
        pulled_back = backward(endpoint[None, :])[0]
        rows.append(
            {
                "role_rank": rank,
                "j_final_mse": float(j_mse[rank]),
                "dominant_top8_residual_fraction": dominant_fraction,
                "fixed_random8_residual_fraction": random_fraction,
                "estimated_top_gain2": float(np.max(gain2)),
                "endpoint_residual_pullback_norm": float(np.linalg.norm(pulled_back)),
            }
        )

    ranks = np.asarray([row["role_rank"] for row in rows], dtype=np.int64)
    hard = np.isin(ranks, np.asarray(HARD_RANKS, dtype=np.int64))
    top_fraction = np.asarray(
        [row["dominant_top8_residual_fraction"] for row in rows], dtype=np.float64
    )
    random_fraction = np.asarray(
        [row["fixed_random8_residual_fraction"] for row in rows], dtype=np.float64
    )
    top_gain2 = np.asarray(
        [row["estimated_top_gain2"] for row in rows], dtype=np.float64
    )
    pullback_norm = np.asarray(
        [row["endpoint_residual_pullback_norm"] for row in rows], dtype=np.float64
    )
    panel_mse = np.asarray([row["j_final_mse"] for row in rows], dtype=np.float64)

    result = {
        "schema": "p2r0f2m-readonly-endpoint-adjoint-capacity-v1",
        "status": "complete-capacity-only-not-performance-estimate",
        "panel": {
            "selection_law": "observer_fit role ranks 0:63 plus the four J-declared >1e-7 ranks",
            "ordinary_role_ranks": list(PANEL_PREFIX),
            "hard_role_ranks": list(HARD_RANKS),
            "networks": len(PANEL),
            "random_seed": SEED,
            "modes": MODES,
            "jacobian_horizon": "L7 to L15, gated transports at L8..L15",
            "subspace_iterations": 1,
        },
        "metrics": {
            "dominant_top8_residual_fraction": summarize(top_fraction, hard),
            "fixed_random8_residual_fraction": summarize(random_fraction, hard),
            "ordinary_dominant_to_random_enrichment": float(
                np.mean(top_fraction[~hard]) / np.mean(random_fraction[~hard])
            ),
            "hard_dominant_to_random_enrichment": float(
                np.mean(top_fraction[hard]) / np.mean(random_fraction[hard])
            ),
            "estimated_top_gain2": summarize(top_gain2, hard),
            "estimated_top_gain2_hard_to_ordinary_mean": float(
                np.mean(top_gain2[hard]) / np.mean(top_gain2[~hard])
            ),
            "estimated_top_gain2_corr_log_j_mse": float(
                np.corrcoef(top_gain2, np.log(panel_mse))[0, 1]
            ),
            "endpoint_residual_pullback_norm": summarize(pullback_norm, hard),
            "endpoint_residual_pullback_norm_hard_to_ordinary_mean": float(
                np.mean(pullback_norm[hard]) / np.mean(pullback_norm[~hard])
            ),
            "endpoint_residual_pullback_norm_corr_log_j_mse": float(
                np.corrcoef(pullback_norm, np.log(panel_mse))[0, 1]
            ),
            "random_subspace_dimension_fraction": MODES / 1024,
        },
        "interpretation": {
            "positive": "Spatial endpoint-adjoint modes are materially enriched for realized J residual energy, and a full residual pullback separates the hard panel.",
            "negative": "Scalar/top-gain and a fixed low-rank basis do not discriminate the hard rows strongly enough; they are rejected as the successor.",
            "design_implication": "Test a full-width learned backward-adjoint state whose sources are predicted only from native target-free raw20/sigma, never an oracle-seeded or fixed-mode correction.",
            "native_transfer_caveat": "CP1792 values establish capacity only. They are not Q24 parity targets; P2R0F2M accuracy requires a separately frozen native-Q24 transfer.",
        },
        "per_network": rows,
    }

    expected = json.loads(EXPECTED_RESULT.read_text(encoding="utf-8"))
    expected_metrics = expected["metrics"]
    for key in (
        "ordinary_dominant_to_random_enrichment",
        "hard_dominant_to_random_enrichment",
        "estimated_top_gain2_hard_to_ordinary_mean",
        "estimated_top_gain2_corr_log_j_mse",
        "endpoint_residual_pullback_norm_hard_to_ordinary_mean",
        "endpoint_residual_pullback_norm_corr_log_j_mse",
    ):
        if not np.isclose(
            result["metrics"][key], expected_metrics[key], rtol=2e-5, atol=1e-9
        ):
            raise AssertionError(
                f"diagnostic mismatch {key}: {result['metrics'][key]} != {expected_metrics[key]}"
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
