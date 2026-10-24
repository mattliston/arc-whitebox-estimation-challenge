#!/Users/matt/anaconda3/bin/python3
"""Read-only, group-held stacking screen over sealed observer_fit OOF models.

The source experiments did not persist unit-level predictions.  This script
reconstructs those predictions from their sealed fold checkpoints, verifies
the published OOF metrics, and then evaluates coefficient-learning rules only
with outer-fold-held predictions.  It never opens observer_selection.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import gc
from pathlib import Path
import sys
from typing import Any

import numpy as np

torch: Any = None


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PREDICTIONS_PATH = HERE / "OOF_PREDICTIONS.npy"
PREDICTIONS_METADATA_PATH = HERE / "OOF_PREDICTIONS_METADATA.npz"
RESULT_PATH = HERE / "RESULT.json"

CANDIDATES = {
    "D": {
        "directory": "public_full688_dynamic_multistate_residual_cv_v1",
        "class": "DynamicMultiStateResidual",
        "call": "dynamic",
    },
    "J": {
        "directory": "public_full688_transport_energy_coherence_cv_v1",
        "class": "TransportEnergyMultiStateResidual",
        "call": "standard",
    },
    "I": {
        "directory": "public_full688_typed_diagonal_error_variance_cv_v1",
        "class": "TypedDiagonalVarianceObserver",
        "call": "standard",
    },
    "K": {
        "directory": "public_full688_full_gram_pooled_context_cv_v1",
        "class": "FullGramPooledContextResidual",
        "call": "standard",
    },
    "M": {
        "directory": "public_full688_backward_adjoint_state_cv_v1",
        "class": "BackwardAdjointStateResidual",
        "call": "standard",
    },
}

EXPECTED_NETWORKS = 688
EXPECTED_DEPTH = 16
EXPECTED_WIDTH = 1024
NATIVE_Q24_HOST_TAPE_FLOPS = 2_131_692_137_721
PHASE2_BUDGET = 2_200_000_000_000
MATERIAL_BEAT_GATE = 3.107e-8
PREFERRED_GATE = 2.5e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(label: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"oof_ensemble_source_{label}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_inventory(*, load_modules: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    reference_valid: list[list[int]] | None = None
    reference_ids: list[list[int]] | None = None
    for label, spec in CANDIDATES.items():
        directory = P2 / str(spec["directory"])
        program = directory / "run_experiment.py"
        cv_path = directory / "CV_RESULTS.json"
        if load_modules:
            module = load_module(label, program)
            modules[label] = module
        cv = read_json(cv_path)
        folds: list[dict[str, Any]] = []
        valid_lists: list[list[int]] = []
        id_lists: list[list[int]] = []
        for fold in range(4):
            result_path = directory / "folds" / f"fold_{fold}.json"
            checkpoint_path = directory / "folds" / f"fold_{fold}.pt"
            result = read_json(result_path)
            if result.get("observer_selection_targets_opened") != 0:
                raise AssertionError(f"{label} fold {fold} opened observer_selection")
            if result.get("model_sha256") != sha256(checkpoint_path):
                raise AssertionError(f"{label} fold {fold} checkpoint hash mismatch")
            valid = [int(value) for value in result["valid_role_ranks"]]
            ids = [int(value) for value in result["valid_mlp_ids"]]
            valid_lists.append(valid)
            id_lists.append(ids)
            folds.append(
                {
                    "fold": fold,
                    "valid_networks": len(valid),
                    "valid_role_ranks_sha256": hashlib.sha256(
                        np.asarray(valid, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "valid_mlp_ids_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "fold_result_sha256": sha256(result_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
        if reference_valid is None:
            reference_valid, reference_ids = valid_lists, id_lists
        elif valid_lists != reference_valid or id_lists != reference_ids:
            raise AssertionError(f"{label} cohort/fold alignment differs")
        covered = sorted(value for values in valid_lists for value in values)
        if covered != list(range(EXPECTED_NETWORKS)):
            raise AssertionError(f"{label} OOF coverage is not disjoint 0:688")
        inventory[label] = {
            "directory": str(directory.relative_to(P2)),
            "program_sha256": sha256(program),
            "cv_results_sha256": sha256(cv_path),
            "published_oof_final_mse": float(cv["oof"]["autonomous_final_mse"]),
            "correction_flops_upper_estimate": int(
                cv["cost"]["correction_source_structure_upper_estimate"]
            ),
            "parameter_count": int(cv["cost"]["parameter_count"]),
            "folds": folds,
            "saved_unit_level_oof_predictions_before_this_screen": False,
            "saved_prediction_artifacts_before_this_screen": [],
            "reconstruction_source": "sealed fold checkpoints plus frozen observer_fit cache",
        }
    return inventory, modules


def instantiate(module: Any, class_name: str, train: np.ndarray, error: np.ndarray,
                sums: np.ndarray, sums2: np.ndarray) -> torch.nn.Module:
    centers, scales, layer_rms = module.moments_for(train, error, sums, sums2)
    return getattr(module, class_name)(centers, scales, layer_rms)


def predict(module: Any, model: torch.nn.Module, call_kind: str,
            raw: torch.Tensor, sigma: torch.Tensor,
            weights: torch.Tensor) -> torch.Tensor:
    if call_kind == "dynamic":
        value = model(raw, sigma, weights, truth=None, exposure=1.0)
    else:
        value = model(raw, sigma, weights, exposure=1.0)
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(type(value))
    return value


def reconstruct_predictions(inventory: dict[str, Any], modules: dict[str, Any]):
    # J owns the shared, frozen cache contract used by J/I/K/M; D's cache is the
    # same artifact and is hash-bound below.
    j = modules["J"]
    dataset = j.PublicResidualDataset(role="observer_fit")
    if dataset.total != EXPECTED_NETWORKS:
        raise AssertionError(dataset.total)
    error, sums, sums2, fold_ids = j.load_shared_cache(dataset)
    if error.shape != (EXPECTED_NETWORKS, EXPECTED_DEPTH, EXPECTED_WIDTH):
        raise AssertionError(error.shape)
    mlp_ids = np.asarray(dataset.mlp_ids, dtype=np.int64)
    predictions = np.lib.format.open_memmap(
        PREDICTIONS_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(len(CANDIDATES), EXPECTED_NETWORKS, EXPECTED_DEPTH, EXPECTED_WIDTH),
    )
    reproduction: dict[str, Any] = {}
    for candidate_index, (label, spec) in enumerate(CANDIDATES.items()):
        module = modules[label]
        candidate_layer_sse = np.zeros(EXPECTED_DEPTH, dtype=np.float64)
        candidate_fold_mse: list[float] = []
        for fold in range(4):
            valid = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            result_path = P2 / str(spec["directory"]) / "folds" / f"fold_{fold}.json"
            record = read_json(result_path)
            if valid.tolist() != [int(value) for value in record["valid_role_ranks"]]:
                raise AssertionError(f"{label} fold {fold} rank mismatch")
            model = instantiate(
                module, str(spec["class"]), train, error, sums, sums2
            )
            checkpoint_path = P2 / str(spec["directory"]) / "folds" / f"fold_{fold}.pt"
            artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(artifact["state_dict"], strict=True)
            model.eval()
            fold_sse = np.zeros(EXPECTED_DEPTH, dtype=np.float64)
            with torch.no_grad():
                for offset in range(0, len(valid), 2):
                    group = valid[offset : offset + 2]
                    raw, sigma, weights, truth = module.load_batch(
                        dataset, group, error
                    )
                    output = predict(
                        module, model, str(spec["call"]), raw, sigma, weights
                    )
                    if tuple(output.shape) != (len(group), EXPECTED_DEPTH, EXPECTED_WIDTH):
                        raise AssertionError((label, fold, tuple(output.shape)))
                    if not bool(torch.isfinite(output).all()):
                        raise FloatingPointError((label, fold))
                    output_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
                    predictions[candidate_index, group] = output_np
                    squared = (output.double() - truth.double()).square()
                    fold_sse += squared.sum(dim=(0, 2)).cpu().numpy()
            candidate_layer_sse += fold_sse
            fold_mse = float(fold_sse[-1] / (len(valid) * EXPECTED_WIDTH))
            published_fold = float(record["autonomous"]["final_mse"])
            if abs(fold_mse - published_fold) > 2e-15:
                raise AssertionError(
                    f"{label} fold {fold} reproduction {fold_mse} != {published_fold}"
                )
            candidate_fold_mse.append(fold_mse)
            del model, artifact
            gc.collect()
        layer_mse = candidate_layer_sse / (EXPECTED_NETWORKS * EXPECTED_WIDTH)
        final_mse = float(layer_mse[-1])
        published = float(inventory[label]["published_oof_final_mse"])
        if abs(final_mse - published) > 2e-15:
            raise AssertionError(f"{label} reproduction {final_mse} != {published}")
        reproduction[label] = {
            "reconstructed_oof_final_mse": final_mse,
            "published_oof_final_mse": published,
            "absolute_difference": abs(final_mse - published),
            "fold_final_mse": candidate_fold_mse,
            "layer_mse": layer_mse.tolist(),
        }
        print(json.dumps({"candidate": label, "oof_final_mse": final_mse}), flush=True)
        predictions.flush()
    np.savez(
        PREDICTIONS_METADATA_PATH,
        candidate_labels=np.asarray(list(CANDIDATES), dtype="U1"),
        role_ranks=np.arange(EXPECTED_NETWORKS, dtype=np.int64),
        mlp_ids=mlp_ids,
        fold_ids=np.asarray(fold_ids, dtype=np.int8),
    )
    return predictions, np.asarray(error), np.asarray(fold_ids), mlp_ids, reproduction


def load_reconstructed_predictions():
    payload = np.load(PREDICTIONS_METADATA_PATH, allow_pickle=False)
    labels = payload["candidate_labels"].tolist()
    if labels != list(CANDIDATES):
        raise AssertionError(labels)
    predictions = np.load(PREDICTIONS_PATH, mmap_mode="r", allow_pickle=False)
    if predictions.shape != (5, EXPECTED_NETWORKS, EXPECTED_DEPTH, EXPECTED_WIDTH):
        raise AssertionError(predictions.shape)
    truth_path = (
        P2 / "public_full688_dynamic_multistate_residual_cv_v1"
        / "cache" / "patched_error.npy"
    )
    error = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    fold_ids = payload["fold_ids"]
    if not np.array_equal(payload["fold_ids"], fold_ids):
        raise AssertionError("saved prediction fold IDs changed")
    return predictions, error, np.asarray(fold_ids), np.asarray(payload["mlp_ids"])


def recover_metadata_from_legacy_archive() -> None:
    """One-time low-memory migration after a completed compressed cache write."""
    legacy = HERE / "OOF_PREDICTIONS.npz"
    if PREDICTIONS_METADATA_PATH.is_file() or not legacy.is_file():
        return
    payload = np.load(legacy, allow_pickle=False)
    np.savez(
        PREDICTIONS_METADATA_PATH,
        candidate_labels=payload["candidate_labels"],
        role_ranks=payload["role_ranks"],
        mlp_ids=payload["mlp_ids"],
        fold_ids=payload["fold_ids"],
    )


def score_prediction(prediction: np.ndarray, truth: np.ndarray,
                     fold_ids: np.ndarray) -> dict[str, Any]:
    layer_sse = np.zeros(EXPECTED_DEPTH, dtype=np.float64)
    per_network = np.empty(EXPECTED_NETWORKS, dtype=np.float64)
    for start in range(0, EXPECTED_NETWORKS, 16):
        stop = min(start + 16, EXPECTED_NETWORKS)
        delta = (
            np.asarray(prediction[start:stop], dtype=np.float64)
            - np.asarray(truth[start:stop], dtype=np.float64)
        )
        layer_sse += np.einsum("nlu,nlu->l", delta, delta, optimize=True)
        per_network[start:stop] = np.einsum(
            "nu,nu->n", delta[:, -1], delta[:, -1], optimize=True
        ) / EXPECTED_WIDTH
    layer_mse = layer_sse / (EXPECTED_NETWORKS * EXPECTED_WIDTH)
    maximum_rank = int(np.argmax(per_network))
    return {
        "raw_final_mse": float(layer_mse[-1]),
        "all_layer_mse": float(layer_mse.mean()),
        "layer_mse": layer_mse.tolist(),
        "fold_final_mse": [
            float(per_network[fold_ids == fold].mean(dtype=np.float64))
            for fold in range(4)
        ],
        "maximum_per_network_final_mse": float(per_network[maximum_rank]),
        "maximum_role_rank": maximum_rank,
    }


def score_fold_weighted_mixture(
    predictions: np.ndarray,
    truth: np.ndarray,
    fold_ids: np.ndarray,
    fold_weights: list[np.ndarray],
    *,
    per_layer: bool,
) -> dict[str, Any]:
    """Score final-layer cross-fitted output in tiny blocks."""
    per_network = np.empty(EXPECTED_NETWORKS, dtype=np.float64)
    for start in range(0, EXPECTED_NETWORKS, 4):
        stop = min(start + 4, EXPECTED_NETWORKS)
        block = np.asarray(predictions[:, start:stop, -1], dtype=np.float32)
        truth_block = np.asarray(truth[start:stop, -1], dtype=np.float64)
        for local, rank in enumerate(range(start, stop)):
            weights = np.asarray(fold_weights[int(fold_ids[rank])], dtype=np.float32)
            if per_layer:
                mixed = np.tensordot(weights[-1], block[:, local], axes=(0, 0))
            else:
                mixed = np.tensordot(weights, block[:, local], axes=(0, 0))
            delta = mixed.astype(np.float64) - truth_block[local]
            per_network[rank] = float(delta @ delta / EXPECTED_WIDTH)
    maximum_rank = int(np.argmax(per_network))
    return {
        "raw_final_mse": float(per_network.mean(dtype=np.float64)),
        "fold_final_mse": [
            float(per_network[fold_ids == fold].mean(dtype=np.float64))
            for fold in range(4)
        ],
        "maximum_per_network_final_mse": float(per_network[maximum_rank]),
        "maximum_role_rank": maximum_rank,
    }


def gram(predictions: np.ndarray, truth: np.ndarray,
         ranks: np.ndarray, layers: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # Stream Gram sufficient statistics so the 225 MB trajectory cache remains
    # memory-mapped and this screen stays subordinate to the primary jobs.
    candidates = predictions.shape[0]
    g = np.zeros((candidates, candidates), dtype=np.float64)
    b = np.zeros(candidates, dtype=np.float64)
    y2 = 0.0
    observations = 0
    for start in range(0, len(ranks), 16):
        group = ranks[start : start + 16]
        x = np.take(np.asarray(predictions[:, group]), layers, axis=2).astype(
            np.float64, copy=False
        )
        y = np.take(np.asarray(truth[group]), layers, axis=1).astype(
            np.float64, copy=False
        )
        flat = x.reshape(candidates, -1)
        target = y.reshape(-1)
        g += flat @ flat.T
        b += flat @ target
        y2 += float(target @ target)
        observations += target.size
    normalizer = float(observations)
    return g / normalizer, b / normalizer, y2 / normalizer


def convex_solution(g: np.ndarray, b: np.ndarray, y2: float) -> tuple[np.ndarray, float]:
    candidates = g.shape[0]
    best_w: np.ndarray | None = None
    best_objective = np.inf
    for count in range(1, candidates + 1):
        for subset_tuple in itertools.combinations(range(candidates), count):
            subset = np.asarray(subset_tuple, dtype=np.int64)
            gs = g[np.ix_(subset, subset)]
            bs = b[subset]
            ones = np.ones(count, dtype=np.float64)
            kkt = np.block([[gs, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            rhs = np.concatenate((bs, np.ones(1, dtype=np.float64)))
            solution = np.linalg.lstsq(kkt, rhs, rcond=1e-12)[0][:-1]
            if np.any(solution < -1e-10):
                continue
            solution = np.maximum(solution, 0.0)
            solution /= solution.sum()
            w = np.zeros(candidates, dtype=np.float64)
            w[subset] = solution
            objective = float(w @ g @ w - 2.0 * b @ w + y2)
            if objective < best_objective:
                best_objective = objective
                best_w = w
    if best_w is None:
        raise RuntimeError("convex solver found no feasible vertex")
    return best_w, best_objective


def unrestricted_solution(g: np.ndarray, b: np.ndarray, y2: float) -> tuple[np.ndarray, float]:
    w = np.linalg.lstsq(g, b, rcond=1e-12)[0]
    return w, float(w @ g @ w - 2.0 * b @ w + y2)


def mix_global(predictions: np.ndarray, weights: np.ndarray,
               ranks: np.ndarray | None = None) -> np.ndarray:
    if ranks is None:
        ranks = np.arange(predictions.shape[1])
    output = np.empty((len(ranks), *predictions.shape[2:]), dtype=np.float32)
    weights32 = weights.astype(np.float32)
    for start in range(0, len(ranks), 4):
        stop = min(start + 4, len(ranks))
        block = np.asarray(predictions[:, ranks[start:stop]], dtype=np.float32)
        output[start:stop] = np.tensordot(weights32, block, axes=(0, 0))
    return output


def mix_per_layer(predictions: np.ndarray, weights: np.ndarray,
                  ranks: np.ndarray | None = None) -> np.ndarray:
    if ranks is None:
        ranks = np.arange(predictions.shape[1])
    output = np.empty((len(ranks), *predictions.shape[2:]), dtype=np.float32)
    weights32 = weights.astype(np.float32)
    for start in range(0, len(ranks), 4):
        stop = min(start + 4, len(ranks))
        block = np.asarray(predictions[:, ranks[start:stop]], dtype=np.float32)
        output[start:stop] = np.einsum(
            "lc,cnlu->nlu", weights32, block, dtype=np.float32, optimize=False
        )
    return output


def deployment_cost(inventory: dict[str, Any], weights: np.ndarray,
                    per_layer: bool) -> dict[str, Any]:
    active = np.any(np.abs(weights) > 1e-8, axis=0) if per_layer else np.abs(weights) > 1e-8
    active_labels = [label for label, value in zip(CANDIDATES, active) if value]
    correction_flops = sum(
        int(inventory[label]["correction_flops_upper_estimate"])
        for label in active_labels
    )
    outputs = EXPECTED_DEPTH * EXPECTED_WIDTH
    blend_flops = len(active_labels) * outputs + max(0, len(active_labels) - 1) * outputs
    parameters = sum(int(inventory[label]["parameter_count"]) for label in active_labels)
    coefficient_count = int(weights.size)
    total = NATIVE_Q24_HOST_TAPE_FLOPS + correction_flops + blend_flops
    return {
        "active_candidates": active_labels,
        "candidate_model_parameters": parameters,
        "blend_coefficient_count": coefficient_count,
        "correction_flops_upper_estimate": correction_flops,
        "blend_flops_upper_estimate": blend_flops,
        "native_q24_host_tape_plus_ensemble_upper_estimate": total,
        "headroom_under_phase2_budget": PHASE2_BUDGET - total,
        "within_phase2_budget_by_conservative_upper_estimate": total < PHASE2_BUDGET,
        "caveat": "source-structure sum; literal integrated meter required before deployment",
    }


def crossfit_global_convex(predictions: np.ndarray, truth: np.ndarray,
                           fold_ids: np.ndarray):
    fold_weights: list[np.ndarray] = []
    final_layer = np.asarray([EXPECTED_DEPTH - 1], dtype=np.int64)
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        valid = np.flatnonzero(fold_ids == fold)
        g, b, y2 = gram(predictions, truth, train, final_layer)
        w, _ = convex_solution(g, b, y2)
        fold_weights.append(w)
    g, b, y2 = gram(predictions, truth, np.arange(EXPECTED_NETWORKS), final_layer)
    final_weights, _ = convex_solution(g, b, y2)
    return fold_weights, final_weights


def crossfit_per_layer_convex(predictions: np.ndarray, truth: np.ndarray,
                              fold_ids: np.ndarray):
    fold_weights: list[np.ndarray] = []
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        valid = np.flatnonzero(fold_ids == fold)
        weights = np.zeros((EXPECTED_DEPTH, len(CANDIDATES)), dtype=np.float64)
        for layer in range(EXPECTED_DEPTH):
            g, b, y2 = gram(
                predictions, truth, train, np.asarray([layer], dtype=np.int64)
            )
            weights[layer], _ = convex_solution(g, b, y2)
        fold_weights.append(weights)
    final_weights = np.zeros((EXPECTED_DEPTH, len(CANDIDATES)), dtype=np.float64)
    all_ranks = np.arange(EXPECTED_NETWORKS)
    for layer in range(EXPECTED_DEPTH):
        g, b, y2 = gram(
            predictions, truth, all_ranks, np.asarray([layer], dtype=np.int64)
        )
        final_weights[layer], _ = convex_solution(g, b, y2)
    return fold_weights, final_weights


def crossfit_best_subset_average(predictions: np.ndarray, truth: np.ndarray,
                                 fold_ids: np.ndarray):
    selected: list[list[str]] = []
    fold_weights: list[np.ndarray] = []
    final_layer = np.asarray([EXPECTED_DEPTH - 1], dtype=np.int64)
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        valid = np.flatnonzero(fold_ids == fold)
        best_subset: tuple[int, ...] | None = None
        best_mse = np.inf
        for count in range(1, len(CANDIDATES) + 1):
            for subset in itertools.combinations(range(len(CANDIDATES)), count):
                prediction = predictions[np.asarray(subset)][:, train, -1].mean(axis=0)
                mse = float(np.square(prediction.astype(np.float64) - truth[train, -1]).mean())
                if mse < best_mse:
                    best_mse, best_subset = mse, subset
        if best_subset is None:
            raise RuntimeError
        weights = np.zeros(len(CANDIDATES), dtype=np.float64)
        weights[np.asarray(best_subset)] = 1.0 / len(best_subset)
        fold_weights.append(weights)
        selected.append([list(CANDIDATES)[value] for value in best_subset])
    # Full-OOF deployment selection, not used to score the same rows.
    all_ranks = np.arange(EXPECTED_NETWORKS)
    best_subset = None
    best_mse = np.inf
    for count in range(1, len(CANDIDATES) + 1):
        for subset in itertools.combinations(range(len(CANDIDATES)), count):
            prediction = predictions[np.asarray(subset)][:, all_ranks, -1].mean(axis=0)
            mse = float(np.square(prediction.astype(np.float64) - truth[:, -1]).mean())
            if mse < best_mse:
                best_mse, best_subset = mse, subset
    assert best_subset is not None
    weights = np.zeros(len(CANDIDATES), dtype=np.float64)
    weights[np.asarray(best_subset)] = 1.0 / len(best_subset)
    return fold_weights, selected, weights


def oracle_linear_span(predictions: np.ndarray, truth: np.ndarray,
                       fold_ids: np.ndarray) -> dict[str, Any]:
    ranks = np.arange(EXPECTED_NETWORKS)
    layer_weights = np.zeros((EXPECTED_DEPTH, len(CANDIDATES)), dtype=np.float64)
    objectives = []
    for layer in range(EXPECTED_DEPTH):
        g, b, y2 = gram(predictions, truth, ranks, np.asarray([layer]))
        weights, objective = unrestricted_solution(g, b, y2)
        layer_weights[layer] = weights
        objectives.append(objective)

    # More optimistic per-unit span.  It fits and scores 5 coefficients on the
    # same 688 rows independently at each final-layer unit, so it is ceiling
    # evidence only and is not a valid deployment estimate.
    x = predictions[:, :, -1].astype(np.float64).transpose(2, 1, 0)
    y = truth[:, -1].astype(np.float64).T
    unit_output = np.empty_like(y)
    unit_weights = np.empty((EXPECTED_WIDTH, len(CANDIDATES)), dtype=np.float64)
    for unit in range(EXPECTED_WIDTH):
        unit_weights[unit] = np.linalg.lstsq(x[unit], y[unit], rcond=1e-12)[0]
        unit_output[unit] = x[unit] @ unit_weights[unit]
    per_network = np.square(unit_output.T - truth[:, -1].astype(np.float64)).mean(axis=1)
    maximum_rank = int(np.argmax(per_network))
    return {
        "label": "NONDEPLOYABLE same-row oracle linear-span ceiling",
        "global_per_layer_same_row": {
            **score_fold_weighted_mixture(
                predictions, truth, fold_ids,
                [layer_weights] * 4, per_layer=True
            ),
            "weights": layer_weights.tolist(),
            "quadratic_objectives": objectives,
        },
        "per_unit_final_layer_same_row": {
            "raw_final_mse": float(np.square(unit_output.T - truth[:, -1]).mean()),
            "fold_final_mse": [
                float(per_network[fold_ids == fold].mean()) for fold in range(4)
            ],
            "maximum_per_network_final_mse": float(per_network[maximum_rank]),
            "maximum_role_rank": maximum_rank,
            "coefficient_count": int(unit_weights.size),
        },
        "nondeployable_reason": (
            "coefficients were fit and evaluated on the same OOF rows; this is a "
            "span-capacity ceiling, not a held-out generalization result"
        ),
    }


def main() -> None:
    np.random.seed(1_753_358_438 & 0xFFFF_FFFF)
    HERE.mkdir(parents=True, exist_ok=True)
    recover_metadata_from_legacy_archive()
    cache_ready = PREDICTIONS_PATH.is_file() and PREDICTIONS_METADATA_PATH.is_file()
    inventory, modules = candidate_inventory(load_modules=not cache_ready)
    print(json.dumps({"stage": "inventory", "cache_ready": cache_ready}), flush=True)
    if cache_ready:
        predictions, truth, fold_ids, mlp_ids = load_reconstructed_predictions()
        print(json.dumps({"stage": "cache_loaded"}), flush=True)
        reproduction = {}
        for index, label in enumerate(CANDIDATES):
            metric = score_prediction(predictions[index], truth, fold_ids)
            published = float(inventory[label]["published_oof_final_mse"])
            if abs(metric["raw_final_mse"] - published) > 2e-15:
                raise AssertionError(f"cached {label} prediction mismatch")
            reproduction[label] = {
                "reconstructed_oof_final_mse": metric["raw_final_mse"],
                "published_oof_final_mse": published,
                "absolute_difference": abs(metric["raw_final_mse"] - published),
                "fold_final_mse": metric["fold_final_mse"],
                "layer_mse": metric["layer_mse"],
            }
            print(json.dumps({"stage": "reproduced", "candidate": label}), flush=True)
    else:
        global torch
        import torch as torch_module
        torch = torch_module
        torch.manual_seed(1_753_358_438)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        predictions, truth, fold_ids, mlp_ids, reproduction = reconstruct_predictions(
            inventory, modules
        )

    methods: dict[str, Any] = {}
    print(json.dumps({"stage": "screen_start"}), flush=True)
    average_weights = np.full(len(CANDIDATES), 1.0 / len(CANDIDATES))
    methods["fixed_equal_average_all_five"] = {
        "validation": "fixed rule; no coefficient fitting",
        **score_fold_weighted_mixture(
            predictions, truth, fold_ids, [average_weights] * 4, per_layer=False
        ),
        "deployment_weights": average_weights.tolist(),
        "deployment_cost": deployment_cost(inventory, average_weights, False),
    }

    subset_fold_weights, subsets, subset_weights = crossfit_best_subset_average(
        predictions, truth, fold_ids
    )
    methods["nested_best_subset_equal_average"] = {
        "validation": "outer-fold-held subset selection",
        **score_fold_weighted_mixture(
            predictions, truth, fold_ids, subset_fold_weights, per_layer=False
        ),
        "outer_fold_selected_subsets": subsets,
        "full_oof_deployment_weights_not_used_for_reported_score": subset_weights.tolist(),
        "deployment_cost": deployment_cost(inventory, subset_weights, False),
    }

    global_fold_weights, global_final_weights = crossfit_global_convex(
        predictions, truth, fold_ids
    )
    methods["nested_nonnegative_convex_global"] = {
        "validation": "outer-fold-held coefficient calibration; one weight per candidate shared over layers/units",
        **score_fold_weighted_mixture(
            predictions, truth, fold_ids, global_fold_weights, per_layer=False
        ),
        "outer_fold_weights": [value.tolist() for value in global_fold_weights],
        "full_oof_deployment_weights_not_used_for_reported_score": global_final_weights.tolist(),
        "deployment_cost": deployment_cost(inventory, global_final_weights, False),
    }

    layer_fold_weights, layer_final_weights = crossfit_per_layer_convex(
        predictions, truth, fold_ids
    )
    methods["nested_nonnegative_convex_per_layer_global_units"] = {
        "validation": "outer-fold-held coefficient calibration; weights shared over networks/units within each layer",
        **score_fold_weighted_mixture(
            predictions, truth, fold_ids, layer_fold_weights, per_layer=True
        ),
        "outer_fold_weights": [value.tolist() for value in layer_fold_weights],
        "full_oof_deployment_weights_not_used_for_reported_score": layer_final_weights.tolist(),
        "deployment_cost": deployment_cost(inventory, layer_final_weights, True),
    }

    candidate_metrics = {
        label: score_prediction(predictions[index], truth, fold_ids)
        for index, label in enumerate(CANDIDATES)
    }
    valid_methods = {
        name: record for name, record in methods.items()
        if record["validation"] != "same-row"
    }
    best_name, best_record = min(
        valid_methods.items(), key=lambda item: float(item[1]["raw_final_mse"])
    )
    materially_beats = float(best_record["raw_final_mse"]) < MATERIAL_BEAT_GATE
    preferred = float(best_record["raw_final_mse"]) < PREFERRED_GATE
    decision = {
        "best_held_method": best_name,
        "best_held_raw_final_mse": float(best_record["raw_final_mse"]),
        "material_gate": MATERIAL_BEAT_GATE,
        "materially_beats_3_107e_8": materially_beats,
        "preferred_gate": PREFERRED_GATE,
        "below_2_5e_8": preferred,
        "promote_deployable_form": materially_beats,
        "family_status": "promote" if materially_beats else "closed_measured",
        "reason": (
            "held stacking materially clears the specified gate"
            if materially_beats
            else "no outer-fold-held ensemble materially beats 3.107e-8"
        ),
    }
    result = {
        "schema": "public-observer-fit-oof-ensemble-screen-v1",
        "status": "complete",
        "scope": {
            "role": "observer_fit",
            "networks": EXPECTED_NETWORKS,
            "group_key": "mlp_id",
            "outer_folds": 4,
            "observer_selection_targets_opened": 0,
            "names_opened": 0,
            "board_opened": 0,
            "new_q24_targets_opened": 0,
        },
        "candidate_order": list(CANDIDATES),
        "inventory": inventory,
        "alignment": {
            "all_candidates_same_role_rank_partition": True,
            "all_candidates_same_mlp_id_partition": True,
            "coverage": "688 disjoint OOF rows exactly once per candidate",
            "fold_counts": [int(np.sum(fold_ids == fold)) for fold in range(4)],
            "mlp_ids_sha256": hashlib.sha256(
                np.asarray(mlp_ids, dtype="<i8").tobytes()
            ).hexdigest(),
        },
        "prediction_artifact": {
            "path": PREDICTIONS_PATH.name,
            "sha256": sha256(PREDICTIONS_PATH),
            "contains": "candidate residual OOF trajectories only; no target array",
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
        },
        "reproduction": reproduction,
        "candidate_metrics": candidate_metrics,
        "held_methods": methods,
        "oracle": oracle_linear_span(predictions, truth, fold_ids),
        "decision": decision,
    }
    stable_json_write(RESULT_PATH, result)
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
