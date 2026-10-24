#!/usr/bin/env python3
"""Open rank-0 q truth once after source freeze and run grouped crossfit ridge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "SOURCE.npz"
SOURCE_RECEIPT = HERE / "SOURCE_RECEIPT.json"
REFERENCE = P2 / "public_q24_cp3_age_retirement_discriminator_v1/runs/rank0000_full.npz"
RESULT = HERE / "RESULT.json"
ALPHAS = (1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_predict(x_train, y_train, x_test, alpha):
    center = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    xt = (x_train - center) / scale
    xv = (x_test - center) / scale
    y_center = float(np.mean(y_train))
    yc = y_train - y_center
    gram = xt.T @ xt
    gram.flat[:: gram.shape[0] + 1] += alpha
    beta = np.linalg.solve(gram, xt.T @ yc)
    return y_center + xv @ beta


def choose_alpha(x, y, coordinate_fold, outer_fold):
    eligible = coordinate_fold != outer_fold
    losses = []
    for alpha in ALPHAS:
        sse = 0.0
        count = 0
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            valid = coordinate_fold == inner_fold
            train = eligible & ~valid
            prediction = fit_predict(x[train], y[train], x[valid], alpha)
            residual = prediction - y[valid]
            sse += float(residual @ residual)
            count += int(np.sum(valid))
        losses.append(sse / count)
    return ALPHAS[int(np.argmin(losses))], losses


def crossfit_layer(x, y, coordinate_fold):
    prediction = np.empty_like(y)
    selected = []
    inner_losses = []
    for outer_fold in range(5):
        alpha, losses = choose_alpha(x, y, coordinate_fold, outer_fold)
        train = coordinate_fold != outer_fold
        test = ~train
        prediction[test] = fit_predict(x[train], y[train], x[test], alpha)
        selected.append(alpha)
        inner_losses.append(losses)
    return prediction, selected, inner_losses


def mse(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.mean(value * value))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    receipt = json.loads(SOURCE_RECEIPT.read_text())
    if receipt.get("status") != "COMPLETE_TARGET_BLIND_SOURCE":
        raise RuntimeError(receipt.get("status"))
    if receipt["bindings"]["source_sha256"] != sha256(SOURCE):
        raise AssertionError("source binding changed")
    with np.load(SOURCE, allow_pickle=False) as source:
        baseline = np.asarray(source["baseline"], dtype=np.float64)
        cp3 = np.asarray(source["cp3_action"], dtype=np.float64)
        qj = np.asarray(source["autonomous_qj"], dtype=np.float64)
        folds = np.asarray(source["coordinate_fold"], dtype=np.int64)
        layers = np.asarray(source["layers"], dtype=np.int64)
    # First and only target access, after the source hash/folds/features are frozen.
    with np.load(REFERENCE, allow_pickle=False) as reference:
        q_truth = np.asarray(reference["q_truth"], dtype=np.float64)[layers]
    gap = q_truth - qj
    baseline_predictions = np.empty_like(gap)
    augmented_predictions = np.empty_like(gap)
    layer_records = []
    for local, layer in enumerate(layers):
        xb = baseline[local]
        xa = np.concatenate((xb, cp3[local, :, None]), axis=1)
        pb, alpha_b, losses_b = crossfit_layer(xb, gap[local], folds)
        pa, alpha_a, losses_a = crossfit_layer(xa, gap[local], folds)
        baseline_predictions[local] = pb
        augmented_predictions[local] = pa
        base_mse = mse(pb - gap[local])
        aug_mse = mse(pa - gap[local])
        layer_records.append(
            {
                "layer_zero_based": int(layer),
                "gap_mse": mse(gap[local]),
                "cp3_gap_correlation": corr(cp3[local], gap[local]),
                "baseline_crossfit_mse": base_mse,
                "augmented_crossfit_mse": aug_mse,
                "relative_mse_reduction": (base_mse - aug_mse) / base_mse,
                "baseline_selected_alphas": alpha_b,
                "augmented_selected_alphas": alpha_a,
                "baseline_inner_losses": losses_b,
                "augmented_inner_losses": losses_a,
            }
        )
    base = mse(baseline_predictions - gap)
    augmented = mse(augmented_predictions - gap)
    result = {
        "schema": "q24-cp3-action-gap-rank0-screen-v1",
        "status": "COMPLETE_NONPROMOTABLE_RANK0",
        "source_feature": "standardized CP3 K3 action",
        "cp4_tested": False,
        "primary": {
            "gap_mse": mse(gap),
            "baseline_crossfit_mse": base,
            "augmented_crossfit_mse": augmented,
            "relative_mse_reduction": (base - augmented) / base,
            "material_incremental_signal": (base - augmented) / base >= 0.01,
        },
        "layers": layer_records,
        "method": {
            "outer_folds": 5,
            "coordinate_grouped_across_layers": True,
            "ridge_alpha_grid": list(ALPHAS),
            "per_layer_models": True,
        },
        "access": {
            "target_member_opened": "q_truth",
            "observer_selection_opened": False,
            "sampling": False,
            "training_networks": 1,
        },
        "bindings": {
            "source_receipt_sha256": sha256(SOURCE_RECEIPT),
            "source_sha256": sha256(SOURCE),
            "scorer_sha256": sha256(Path(__file__)),
            "reference_sha256": sha256(REFERENCE),
        },
    }
    temporary = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

