#!/usr/bin/env python3
"""Transported raw20 bank: dev128 -> inner32 ridge capacity screen.

All twenty target-blind native fields are carried independently through the
actual Phi*W dynamics from layer 2 onward.  Two fixed ridge consumers compete:
six incremental delta-q sources, and one direct endpoint residual.  Protected
rows remain unopened unless the better inner arm reduces the frozen
alpha=.75+q-head MSE by at least one percent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
Q_SOURCE = P2 / "public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py"
Q_MODEL = P2 / "public_q24_native_j_q10q15_residual_head_v1/Q_SUFFIX_HEAD.pt"
BASE_MODEL = P2 / "public_q24_native_j_checkpoint_interpolation_v1/INTERPOLATED_MODEL.pt"
BASE_RESULT = P2 / "public_q24_native_j_checkpoint_interpolation_v1/RESULT.json"
OUT = HERE / "RESULT.json"
ARRAYS = HERE / "SELECTED_RIDGE.npz"

SUFFIX = tuple(range(10, 16))
RIDGES = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
FEATURES = 43
MATERIAL_RATIO = 0.99
MATERIAL_REFERENCE_MSE = 3.061465197605418e-8
MATERIAL_THRESHOLD_MSE = MATERIAL_RATIO * MATERIAL_REFERENCE_MSE


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


def load_base(j):
    state = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state["centers"].numpy(), state["scales"].numpy(), state["layer_rms"].numpy()
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_q(q, base):
    state = torch.load(Q_MODEL, map_location="cpu", weights_only=False)["state_dict"]
    model = q.QSuffixHead(base.centers, base.scales, base.layer_rms)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def raw_standard(raw: np.ndarray, base) -> np.ndarray:
    tail = np.clip(
        (raw[..., 1:].astype(np.float64) - base.centers.numpy()[:, None, :])
        / base.scales.numpy()[:, None, :],
        -8.0, 8.0,
    )
    return np.ascontiguousarray(np.concatenate((raw[..., :1], tail), axis=-1), dtype=np.float32)


def transport_bank(q, dataset, locals_: np.ndarray, base) -> np.ndarray:
    result = np.empty((len(locals_), len(SUFFIX), 1024, 20), dtype=np.float32)
    for position, local in enumerate(locals_):
        row = dataset.load(int(local))
        raw = np.asarray(row.raw_locals)
        standardized = raw_standard(raw, base)
        weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
        state = np.zeros((1024, 20), dtype=np.float32)
        for layer in range(2, 16):
            if layer > 2:
                state = (
                    raw[layer, :, 5, None]
                    * (weights[layer].T @ state)
                    + standardized[layer]
                ).astype(np.float32)
            else:
                state = standardized[layer].copy()
            if layer in SUFFIX:
                result[position, SUFFIX.index(layer)] = state
        if (position + 1) % 16 == 0:
            print(f"transport {position + 1}/{len(locals_)}", flush=True)
    return result


def head_sources(q, head, dataset, prepared) -> tuple[np.ndarray, np.ndarray]:
    sources = np.empty_like(prepared["delta_q"])
    finals = np.empty((len(prepared["locals"]), 1024), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for position, local in enumerate(prepared["locals"]):
            row = dataset.load(int(local))
            raw = np.asarray(row.raw_locals)
            sigma = np.asarray(row.sigma)
            weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
            state = np.zeros(1024, dtype=np.float32)
            for slot, layer in enumerate(SUFFIX):
                source = head.source(
                    torch.from_numpy(np.asarray(raw[layer]).copy()),
                    torch.from_numpy(np.asarray(sigma[layer]).copy()),
                    torch.from_numpy(prepared["base_pred"][position, layer]),
                    torch.from_numpy(prepared["base_q"][position, slot]),
                    layer,
                ).cpu().numpy()
                sources[position, slot] = source
                state = (
                    raw[layer, :, 5].astype(np.float64)
                    * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                    + source.astype(np.float64)
                ).astype(np.float32)
            finals[position] = state
    return sources, finals


def bank_moments(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(bank.astype(np.float64), axis=(0, 2), dtype=np.float64)
    second = np.mean(bank.astype(np.float64) ** 2, axis=(0, 2), dtype=np.float64)
    scale = np.sqrt(np.maximum(second - center * center, 1e-12))
    return center.astype(np.float32), scale.astype(np.float32)


def designs(q, dataset, prepared, bank, center, scale, base) -> tuple[np.ndarray, np.ndarray]:
    n = len(prepared["locals"])
    x = np.empty((n, len(SUFFIX), 1024, FEATURES), dtype=np.float32)
    physical = np.empty((n, len(SUFFIX), 1024), dtype=np.float32)
    for position, local in enumerate(prepared["locals"]):
        row = dataset.load(int(local))
        raw = np.asarray(row.raw_locals)
        sigma = np.asarray(row.sigma)
        rawz = raw_standard(raw, base)
        for slot, layer in enumerate(SUFFIX):
            rms = float(base.layer_rms[layer])
            z = np.clip(
                (bank[position, slot].astype(np.float64) - center[slot]) / scale[slot],
                -8.0, 8.0,
            )
            a = np.clip(prepared["base_pred"][position, layer] / rms, -8.0, 8.0)
            pscale = np.maximum(sigma[layer] * rms, 2.0**-30).astype(np.float32)
            qj = np.clip(prepared["base_q"][position, slot] / pscale, -8.0, 8.0)
            x[position, slot, :, 0] = 1.0
            x[position, slot, :, 1:21] = rawz[layer]
            x[position, slot, :, 21] = a
            x[position, slot, :, 22] = qj
            x[position, slot, :, 23:43] = z
            physical[position, slot] = pscale
    return x, physical


def sufficient(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    flat_y = y.reshape(-1).astype(np.float64)
    count = float(len(flat_y))
    return (flat_x.T @ flat_x) / count, (flat_x.T @ flat_y) / count


def ridge_solution(gram: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    column = np.sqrt(np.maximum(np.diag(gram), 1e-20))
    normalized = gram / column[:, None] / column[None, :]
    system = normalized + ridge * np.eye(len(rhs))
    if ridge == 0.0:
        beta_normal = np.linalg.pinv(system, rcond=1e-12) @ (rhs / column)
    else:
        try:
            beta_normal = np.linalg.solve(system, rhs / column)
        except np.linalg.LinAlgError:
            beta_normal = np.linalg.lstsq(system, rhs / column, rcond=1e-12)[0]
    return beta_normal / column


def rollout_local(q, dataset, prepared, base_sources, x, physical, betas) -> float:
    sse = 0.0
    for position, local in enumerate(prepared["locals"]):
        row = dataset.load(int(local))
        raw = np.asarray(row.raw_locals)
        weights = np.load(row.weights_path, mmap_mode="r", allow_pickle=False)
        state = np.zeros(1024, dtype=np.float32)
        for slot, layer in enumerate(SUFFIX):
            increment = physical[position, slot] * (x[position, slot].astype(np.float64) @ betas[slot])
            source = base_sources[position, slot].astype(np.float64) + increment
            state = (
                raw[layer, :, 5].astype(np.float64)
                * (state.astype(np.float64) @ weights[layer].astype(np.float64))
                + source
            ).astype(np.float32)
        residual = prepared["residual_final"][position].astype(np.float64) - state.astype(np.float64)
        sse += float(np.dot(residual, residual))
    return sse / (len(prepared["locals"]) * 1024)


def direct_mse(prepared, base_final, x15, physical15, beta) -> float:
    correction = physical15.astype(np.float64) * np.einsum(
        "nwf,f->nw", x15.astype(np.float64), beta, optimize=True
    )
    residual = (
        prepared["residual_final"].astype(np.float64)
        - base_final.astype(np.float64) - correction
    )
    return float(np.mean(residual * residual))


def main() -> None:
    if OUT.exists() or ARRAYS.exists():
        raise FileExistsError("transport-bank output already exists")
    torch.manual_seed(0x54524157)
    np.random.seed(0x54524157)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    j = load_module("transport_raw20_exact_j", J_SOURCE)
    q = load_module("transport_raw20_q", Q_SOURCE)
    dataset = q.Dataset()
    target, host = q.targets_and_host()
    error = np.ascontiguousarray(target - host, dtype=np.float32)
    dev, inner, protected = q.split()
    base = load_base(j)
    head = load_q(q, base)

    prepared_dev = q.prepare(j, base, dataset, dev, error)
    prepared_inner = q.prepare(j, base, dataset, inner, error)
    head_dev, head_final_dev = head_sources(q, head, dataset, prepared_dev)
    head_inner, head_final_inner = head_sources(q, head, dataset, prepared_inner)
    bank_dev = transport_bank(q, dataset, dev, base)
    bank_inner = transport_bank(q, dataset, inner, base)
    center, scale = bank_moments(bank_dev)
    xdev, pdev = designs(q, dataset, prepared_dev, bank_dev, center, scale, base)
    xinner, pinner = designs(q, dataset, prepared_inner, bank_inner, center, scale, base)
    del bank_dev, bank_inner

    baseline_inner = float(np.mean(
        (prepared_inner["residual_final"].astype(np.float64) - head_final_inner.astype(np.float64)) ** 2
    ))
    local_stats = []
    for slot in range(len(SUFFIX)):
        target_standard = (
            prepared_dev["delta_q"][:, slot] - head_dev[:, slot]
        ) / pdev[:, slot]
        local_stats.append(sufficient(xdev[:, slot], target_standard))
    direct_target = (
        prepared_dev["residual_final"] - head_final_dev
    ) / pdev[:, -1]
    direct_stat = sufficient(xdev[:, -1], direct_target)

    local_records = []
    direct_records = []
    local_candidates = {}
    direct_candidates = {}
    for ridge in RIDGES:
        betas = np.stack([ridge_solution(g, r, ridge) for g, r in local_stats])
        value = rollout_local(q, dataset, prepared_inner, head_inner, xinner, pinner, betas)
        local_records.append({"ridge": ridge, "inner_final_mse": value})
        local_candidates[ridge] = betas
        beta_direct = ridge_solution(direct_stat[0], direct_stat[1], ridge)
        direct_value = direct_mse(
            prepared_inner, head_final_inner, xinner[:, -1], pinner[:, -1], beta_direct
        )
        direct_records.append({"ridge": ridge, "inner_final_mse": direct_value})
        direct_candidates[ridge] = beta_direct
        print(json.dumps({"ridge": ridge, "local": value, "direct": direct_value}), flush=True)

    local_best = min((row["inner_final_mse"], row["ridge"]) for row in local_records)
    direct_best = min((row["inner_final_mse"], row["ridge"]) for row in direct_records)
    best_mse, best_ridge, best_arm = min(
        (local_best[0], local_best[1], "layerwise_delta_q"),
        (direct_best[0], direct_best[1], "direct_endpoint"),
    )
    material_gate = bool(best_mse <= MATERIAL_THRESHOLD_MSE)
    protected_record = None
    selected_beta = (
        local_candidates[best_ridge] if best_arm == "layerwise_delta_q"
        else direct_candidates[best_ridge]
    )

    if material_gate:
        # The arm and ridge are now frozen; protected is prepared exactly once.
        prepared_protected = q.prepare(j, base, dataset, protected, error)
        head_protected, head_final_protected = head_sources(q, head, dataset, prepared_protected)
        bank_protected = transport_bank(q, dataset, protected, base)
        xprotected, pprotected = designs(
            q, dataset, prepared_protected, bank_protected, center, scale, base
        )
        baseline_protected = float(np.mean(
            (prepared_protected["residual_final"].astype(np.float64) - head_final_protected.astype(np.float64)) ** 2
        ))
        if best_arm == "layerwise_delta_q":
            candidate_protected = rollout_local(
                q, dataset, prepared_protected, head_protected,
                xprotected, pprotected, selected_beta,
            )
        else:
            candidate_protected = direct_mse(
                prepared_protected, head_final_protected,
                xprotected[:, -1], pprotected[:, -1], selected_beta,
            )
        protected_record = {
            "opened": True,
            "baseline_alpha075_plus_q_head_mse": baseline_protected,
            "candidate_mse": candidate_protected,
            "ratio": candidate_protected / baseline_protected,
        }
    else:
        protected_record = {
            "opened": False,
            "reason": "best inner arm did not clear the frozen 0.99*3.061465197605418e-8 threshold",
        }

    np.savez(
        ARRAYS, bank_center=center, bank_scale=scale,
        selected_beta=selected_beta, selected_arm=np.asarray(best_arm),
        selected_ridge=np.asarray(best_ridge),
    )
    n = 1024
    bank_transitions = 13
    bank_transport = bank_transitions * 20 * ((2 * n - 1) * n + n)
    bank_source = 14 * 20 * n
    ridge_read = (6 if best_arm == "layerwise_delta_q" else 1) * (2 * FEATURES - 1) * n
    result = {
        "schema": "q24-native-j-transported-raw20-bank-screen-v1",
        "status": "complete",
        "observer_fit_only": True,
        "observer_selection_targets_opened": 0,
        "board_submissions": 0,
        "hypothesis": "retain all 20 standardized native fields as independent Phi*W states instead of compressing history into J's four latent channels",
        "state": {
            "channels": 20,
            "source_layers_zero_based": list(range(2, 16)),
            "read_layers_zero_based": list(SUFFIX),
            "recurrence": "Z_l=Phi_l*(Z_(l-1)@W_l)+standardized_raw20_l",
            "consumer_features": "bias+current raw20+A+qJ+standardized Z (43 total)",
        },
        "split": json.loads(P2.joinpath("public_q24_native_j_autonomous_continuation_v1/RESULT.json").read_text())["split"],
        "gate": {
            "protected_open_ratio_at_most": MATERIAL_RATIO,
            "frozen_reference_alpha075_plus_q_head_protected_mse": MATERIAL_REFERENCE_MSE,
            "inner_baseline_alpha075_plus_q_head_mse": baseline_inner,
            "threshold_mse": MATERIAL_THRESHOLD_MSE,
            "pass": material_gate,
        },
        "screens": {
            "ridge_grid": list(RIDGES),
            "layerwise_delta_q": local_records,
            "direct_endpoint": direct_records,
            "selected_arm": best_arm,
            "selected_ridge": best_ridge,
            "selected_inner_mse": best_mse,
            "selected_inner_ratio": best_mse / baseline_inner,
        },
        "protected_once": protected_record,
        "deployability": {
            "incremental_bank_transport_flops": int(bank_transport),
            "incremental_bank_source_add_flops": int(bank_source),
            "incremental_selected_ridge_read_flops": int(ridge_read),
            "incremental_total_flops": int(bank_transport + bank_source + ridge_read),
            "note": "does not include the already-banked 130.935M q-head upper bound",
        },
        "bindings": {
            "program_sha256": sha256(Path(__file__)),
            "base_model_sha256": sha256(BASE_MODEL),
            "q_head_sha256": sha256(Q_MODEL),
            "base_result_sha256": sha256(BASE_RESULT),
            "selected_arrays_sha256": sha256(ARRAYS),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": result["screens"], "protected": protected_record}, sort_keys=True))


if __name__ == "__main__":
    main()
