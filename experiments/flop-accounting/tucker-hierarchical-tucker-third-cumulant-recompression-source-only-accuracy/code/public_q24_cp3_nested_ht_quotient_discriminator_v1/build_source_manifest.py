#!/usr/bin/env python3
"""Build the frozen rank-0 r128/k32 quotient nested-HT source manifest.

The runner is target blind.  It reads only the quarantined CP3 factor tape,
analytic host quantities, and the bound weights.  It is intentionally heavy
and must not be launched without the explicit `--execute-source-only` flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from nested_ht_state import (
    canonicalize_columns,
    project_cp_quotient_core,
    read_k21,
    read_k3,
)


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
QUARANTINE = P2 / "public_q24_cp3_full_snapshot_rank0_v1/runs/row_0000"
QUARANTINE_MANIFEST = QUARANTINE / "QUARANTINE_MANIFEST.json"
FAILED_PARITY = QUARANTINE / "FAILED_BYTE_PARITY.json"
WEIGHTS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000/weights.npy"
)
PREREGISTER = HERE / "PREREGISTER.md"
STATE_SOURCE = HERE / "nested_ht_state.py"

WIDTH = 1024
DEPTH = 16
RANK = 128
ACTION_RANK = 32
ACTION_PROBES = 64
STATE_PROBES = 128
SEED = 2026082612832
CP_BATCH = 256

EXPECTED_QUARANTINE_SHA256 = (
    "ef187af733a2f8ae46b8ad3d85e3b17d8a41722063588f0d51e0fab46074ae3c"
)
EXPECTED_FAILED_PARITY_SHA256 = (
    "b27cafad2487414fd508dfb5a229f09c14c311b2a61755165525dd05de572cd6"
)
EXPECTED_WEIGHTS_SHA256 = (
    "399488100ef907d6a76fa27e911285809bf5d53405bc59290c84396a7e4c8c97"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def mse(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.mean(array * array))


def layer_array(layer: int, name: str, *, mmap: bool = True) -> np.ndarray:
    return np.load(
        QUARANTINE / f"layer_{layer:02d}" / f"{name}.npy",
        mmap_mode="r" if mmap else None,
        allow_pickle=False,
    )


def rademacher(rng: np.random.Generator, columns: int) -> np.ndarray:
    value = rng.integers(0, 2, size=(WIDTH, columns), dtype=np.int8)
    return np.asarray(2 * value - 1, dtype=np.float64) / np.sqrt(WIDTH)


def normalize_columns(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    scale = np.linalg.norm(result, axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1.0e-300):
        raise AssertionError((float(np.min(scale)), float(np.max(scale))))
    return result / scale[None, :]


def top_left_vectors(value: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    vectors, singular, _right = np.linalg.svd(
        np.asarray(value, dtype=np.float64), full_matrices=False
    )
    if vectors.shape[1] < count or singular[count - 1] <= 0.0:
        raise AssertionError((vectors.shape, singular[: min(count, singular.size)]))
    selected = canonicalize_columns(vectors[:, :count])
    return selected, singular


def reconstruct_post_cp(layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam_pre = np.asarray(layer_array(layer, "lambda_pre"), dtype=np.float32)
    pushed_p = np.asarray(layer_array(layer, "pushed_p"), dtype=np.float32)
    pushed_q = np.asarray(layer_array(layer, "pushed_q"), dtype=np.float32)
    gate = np.asarray(layer_array(layer, "gate32"), dtype=np.float32)
    covariance = np.asarray(layer_array(layer, "pre_covariance"), dtype=np.float32)
    birth_lambda = np.asarray(layer_array(layer, "birth_lambda32"), dtype=np.float32)
    prior_rank = int(lam_pre.shape[0])
    rank = prior_rank + WIDTH
    lam = np.empty(rank, dtype=np.float32)
    p = np.empty((WIDTH, rank), dtype=np.float32)
    q = np.empty((WIDTH, rank), dtype=np.float32)
    lam[:prior_rank] = lam_pre
    lam[prior_rank:] = birth_lambda
    p[:, :prior_rank] = gate[:, None] * pushed_p
    q[:, :prior_rank] = gate[:, None] * pushed_q
    p[:, prior_rank:] = gate[:, None] * covariance
    q[:, prior_rank:] = np.eye(WIDTH, dtype=np.float32)
    return lam, p, q


def reverse_endpoint_adjoint(weights: np.ndarray, probes: np.ndarray) -> dict[int, np.ndarray]:
    adjoint: dict[int, np.ndarray] = {15: normalize_columns(probes)}
    for layer in range(14, 0, -1):
        next_layer = layer + 1
        gate = np.asarray(layer_array(next_layer, "gate64"), dtype=np.float64)
        value = np.asarray(weights[next_layer], dtype=np.float64) @ (
            gate[:, None] * adjoint[next_layer]
        )
        adjoint[layer] = normalize_columns(value)
    return adjoint


def quotient_action_sketch(
    lam: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    pushed_p: np.ndarray,
    pushed_q: np.ndarray,
    repeated: np.ndarray,
    weight: np.ndarray,
    observation: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.zeros((WIDTH, observation.shape[1]), dtype=np.float64)
    d32 = np.asarray(observation, dtype=np.float32)
    for start in range(0, lam.shape[0], CP_BATCH):
        stop = min(start + CP_BATCH, lam.shape[0])
        p_part = p[:, start:stop]
        q_part = q[:, start:stop]
        pushed_p_part = pushed_p[:, start:stop].T
        pushed_q_part = pushed_q[:, start:stop].T
        scale = lam[start:stop, None]
        first = (scale * (pushed_p_part * pushed_q_part)) @ d32
        second = (scale * (pushed_p_part * pushed_p_part)) @ d32
        source += 2.0 * np.asarray(p_part @ first, dtype=np.float64)
        source += np.asarray(q_part @ second, dtype=np.float64)
    h = np.asarray(repeated, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    d = np.asarray(observation, dtype=np.float64)
    diagonal_map = (w * w) @ d
    lifted = (
        h.T @ diagonal_map
        + 2.0 * (w * (h @ w)) @ d
        - 2.0 * np.diag(h)[:, None] * diagonal_map
    )
    quotient = source - lifted
    return quotient, {
        "raw_sketch_mse": mse(source),
        "lift_sketch_mse": mse(lifted),
        "quotient_sketch_mse": mse(quotient),
    }


def quotient_state_sketch(
    lam: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    repeated: np.ndarray,
    probe_a: np.ndarray,
    probe_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.zeros((WIDTH, probe_a.shape[1]), dtype=np.float64)
    a32 = np.asarray(probe_a, dtype=np.float32)
    b32 = np.asarray(probe_b, dtype=np.float32)
    for start in range(0, lam.shape[0], CP_BATCH):
        stop = min(start + CP_BATCH, lam.shape[0])
        p_part = p[:, start:stop]
        q_part = q[:, start:stop]
        scale = lam[start:stop, None]
        pa = p_part.T @ a32
        pb = p_part.T @ b32
        qa = q_part.T @ a32
        qb = q_part.T @ b32
        source += np.asarray(p_part @ (scale * pa * qb), dtype=np.float64)
        source += np.asarray(p_part @ (scale * qa * pb), dtype=np.float64)
        source += np.asarray(q_part @ (scale * pa * pb), dtype=np.float64)
    h = np.asarray(repeated, dtype=np.float64)
    a = np.asarray(probe_a, dtype=np.float64)
    b = np.asarray(probe_b, dtype=np.float64)
    product = a * b
    lifted = (
        a * (h @ b)
        + b * (h @ a)
        + h.T @ product
        - 2.0 * np.diag(h)[:, None] * product
    )
    quotient = source - lifted
    return quotient, {
        "raw_sketch_mse": mse(source),
        "lift_sketch_mse": mse(lifted),
        "quotient_sketch_mse": mse(quotient),
    }


def build_basis(
    lam: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    pushed_p: np.ndarray,
    pushed_q: np.ndarray,
    repeated: np.ndarray,
    weight: np.ndarray,
    observation: np.ndarray,
    probe_a: np.ndarray,
    probe_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    action_y, action_metrics = quotient_action_sketch(
        lam, p, q, pushed_p, pushed_q, repeated, weight, observation
    )
    action, action_singular = top_left_vectors(action_y, ACTION_RANK)
    state_y, state_metrics = quotient_state_sketch(
        lam, p, q, repeated, probe_a, probe_b
    )
    # Two projections suppress numerical leakage back into the frozen action block.
    state_y -= action @ (action.T @ state_y)
    state_y -= action @ (action.T @ state_y)
    state, state_singular = top_left_vectors(state_y, RANK - ACTION_RANK)
    state -= action @ (action.T @ state)
    state, _ = np.linalg.qr(state, mode="reduced")
    state = canonicalize_columns(state[:, : RANK - ACTION_RANK])
    basis = np.concatenate((action, state), axis=1)
    gram_error = float(np.max(np.abs(basis.T @ basis - np.eye(RANK))))
    if gram_error > 2.0e-10:
        raise AssertionError(gram_error)
    return np.asarray(basis, dtype=np.float32), {
        "action": action_metrics,
        "state": state_metrics,
        "action_singular_top": [float(x) for x in action_singular[:ACTION_RANK]],
        "action_singular_tail": float(action_singular[ACTION_RANK - 1]),
        "state_singular_top": [
            float(x) for x in state_singular[: RANK - ACTION_RANK]
        ],
        "state_singular_tail": float(state_singular[RANK - ACTION_RANK - 1]),
        "gram_max_abs": gram_error,
    }


def pushed_star_k3(star: np.ndarray, weight: np.ndarray) -> np.ndarray:
    h = np.asarray(star, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    diagonal = np.diag(h).copy()
    directed = np.array(h, copy=True, order="C")
    np.fill_diagonal(directed, np.float32(0.0))
    transported = directed @ w
    w2 = w * w
    return np.asarray(
        np.sum(
            diagonal[:, None] * w2 * w
            + np.float32(3.0) * w2 * transported,
            axis=0,
            dtype=np.float32,
        ),
        dtype=np.float64,
    )


def pushed_star_k21(star: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Same float32 coordinate-star push as the frozen D1 implementation."""

    h = np.asarray(star, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    diagonal = np.diag(h).copy()
    directed = np.array(h, copy=True, order="C")
    np.fill_diagonal(directed, np.float32(0.0))
    transported = directed @ w
    w2 = w * w
    pushed_diagonal = np.sum(
        diagonal[:, None] * w2 * w
        + np.float32(3.0) * w2 * transported,
        axis=0,
        dtype=np.float32,
    )
    pushed = (
        w2.T @ (diagonal[:, None] * w)
        + np.float32(2.0) * (w * transported).T @ w
        + w2.T @ transported
    )
    pushed = np.asarray(pushed, dtype=np.float32)
    np.fill_diagonal(pushed, pushed_diagonal)
    return np.asarray(pushed, dtype=np.float64)


def direct_cp_k21(
    lam: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Frozen same-NumPy full CP contraction control."""

    lam32 = np.asarray(lam, dtype=np.float32)
    p32 = np.asarray(p, dtype=np.float32)
    q32 = np.asarray(q, dtype=np.float32)
    first = (p32 * p32 * lam32[None, :]) @ q32.T
    second = (p32 * q32 * lam32[None, :]) @ p32.T
    return np.asarray(first + np.float32(2.0) * second, dtype=np.float64)


def direct_cp_k3(
    lam: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    lam32 = np.asarray(lam, dtype=np.float32)
    p32 = np.asarray(p, dtype=np.float32)
    q32 = np.asarray(q, dtype=np.float32)
    return np.asarray(
        np.float32(3.0)
        * np.sum(
            lam32[None, :] * p32 * p32 * q32,
            axis=1,
            dtype=np.float32,
        ),
        dtype=np.float64,
    )


def full_pre_k3(next_layer: int) -> np.ndarray:
    return np.asarray(layer_array(next_layer, "k3"), dtype=np.float64)


def endpoint_action(local: np.ndarray, start_layer: int, weights: np.ndarray) -> np.ndarray:
    value = np.asarray(local, dtype=np.float64)
    for layer in range(start_layer + 1, DEPTH):
        gate = np.asarray(layer_array(layer, "gate64"), dtype=np.float64)
        value = (value @ np.asarray(weights[layer], dtype=np.float64)) * gate
    return value


def evaluate_core(
    u: np.ndarray,
    core: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    next_layer: int,
    direct_next_k21: np.ndarray | None,
    direct_next_k3: np.ndarray,
    canonical_k3: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | None]]:
    core32 = np.asarray(core, dtype=np.float32)
    current = read_k21(u, core32)
    star = np.asarray(np.asarray(target, dtype=np.float64) - current, dtype=np.float32)
    parity = current + star.astype(np.float64) - np.asarray(target, dtype=np.float64)
    pushed_u = np.asarray(np.asarray(weight, dtype=np.float32).T @ u, dtype=np.float32)
    if next_layer < DEPTH - 1:
        if direct_next_k21 is None:
            raise AssertionError(next_layer)
        candidate_next_k21 = read_k21(pushed_u, core32) + pushed_star_k21(
            star, weight
        )
        plane_delta = candidate_next_k21 - np.asarray(
            direct_next_k21, dtype=np.float64
        )
        # Q24 consumes the diagonal of the physically materialized full read.
        candidate_same_k3 = np.diag(candidate_next_k21).copy()
        plane_mse: float | None = mse(plane_delta)
        plane_rms: float | None = float(np.sqrt(plane_mse))
        plane_max: float | None = float(np.max(np.abs(plane_delta)))
    else:
        if direct_next_k21 is not None:
            raise AssertionError(next_layer)
        # Only the final layer uses the marginal-only carrier read.
        candidate_same_k3 = read_k3(pushed_u, core32) + pushed_star_k3(star, weight)
        plane_mse = None
        plane_rms = None
        plane_max = None
    delta = candidate_same_k3 - np.asarray(direct_next_k3, dtype=np.float64)
    # Reanchor only the isolated representation delta to the canonical D1 tape.
    candidate_k3 = np.asarray(canonical_k3, dtype=np.float64) + delta
    rho3 = np.asarray(layer_array(next_layer, "rho3"), dtype=np.float64)
    local_mean = rho3 * delta
    endpoint = endpoint_action(local_mean, next_layer, weights)
    return candidate_k3, {
        "reset_parity_rms": float(np.sqrt(mse(parity))),
        "reset_parity_max_abs": float(np.max(np.abs(parity))),
        "k3_delta_mse": mse(delta),
        "k3_delta_rms": float(np.sqrt(mse(delta))),
        "k3_delta_max_abs": float(np.max(np.abs(delta))),
        "next_full_plane_delta_mse": plane_mse,
        "next_full_plane_delta_rms": plane_rms,
        "next_full_plane_delta_max_abs": plane_max,
        "local_mean_mse": mse(local_mean),
        "endpoint_action_mse": mse(endpoint),
        "endpoint_action_max_abs": float(np.max(np.abs(endpoint))),
    }


def save_array(path: Path, value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": sha256(path),
        "array_sha256": array_sha256(array),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-source-only", action="store_true", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "runs/source_rank0000_r128_k32_v1",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path, expected in (
        (QUARANTINE_MANIFEST, EXPECTED_QUARANTINE_SHA256),
        (FAILED_PARITY, EXPECTED_FAILED_PARITY_SHA256),
        (WEIGHTS, EXPECTED_WEIGHTS_SHA256),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError((path, actual, expected))
    quarantine_manifest = json.loads(QUARANTINE_MANIFEST.read_text())
    if quarantine_manifest.get("status") != "QUARANTINED_NOT_PROMOTED":
        raise RuntimeError(quarantine_manifest.get("status"))
    weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
        raise AssertionError((weights.shape, weights.dtype))

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    running = {
        "schema": "q24-cp3-quotient-nested-ht-source-running-v1",
        "status": "RUNNING_SOURCE_ONLY_NONPROMOTABLE",
        "started_unix_ns": time.time_ns(),
    }
    (temporary / "RUNNING.json").write_text(
        json.dumps(running, indent=2, sort_keys=True) + "\n"
    )

    rng = np.random.default_rng(SEED)
    endpoint_probes = rademacher(rng, ACTION_PROBES)
    state_probe_a = {layer: rademacher(rng, STATE_PROBES) for layer in range(1, 15)}
    state_probe_b = {layer: rademacher(rng, STATE_PROBES) for layer in range(1, 15)}
    adjoint = reverse_endpoint_adjoint(weights, endpoint_probes)
    captured_k3 = np.zeros((DEPTH, WIDTH), dtype=np.float64)
    for layer in range(1, DEPTH):
        captured_k3[layer] = full_pre_k3(layer)
    quotient_k3 = captured_k3.copy()
    raw_k3 = captured_k3.copy()

    receipt_layers: dict[str, object] = {}
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        for layer in range(1, 15):
            next_layer = layer + 1
            layer_output = temporary / f"layer_{layer:02d}"
            layer_output.mkdir()
            lam, p, q = reconstruct_post_cp(layer)
            weight = np.asarray(weights[next_layer], dtype=np.float32)
            pushed_p = np.asarray(weight.T @ p, dtype=np.float32)
            pushed_q = np.asarray(weight.T @ q, dtype=np.float32)
            repeated = direct_cp_k21(lam, p, q)
            captured_repeated = np.asarray(
                layer_array(layer, "cp_post_k21_full"), dtype=np.float64
            )
            target = np.asarray(layer_array(layer, "k21_target"), dtype=np.float64)
            direct_star = np.asarray(target - repeated, dtype=np.float32)
            if next_layer < DEPTH - 1:
                direct_next_k21 = direct_cp_k21(
                    lam, pushed_p, pushed_q
                ) + pushed_star_k21(direct_star, weight)
                # Match the physical 29-full-read schedule exactly.
                direct_next_k3 = np.diag(direct_next_k21).copy()
            else:
                direct_next_k21 = None
                # Match the one final marginal-only read exactly.
                direct_next_k3 = direct_cp_k3(
                    lam, pushed_p, pushed_q
                ) + pushed_star_k3(direct_star, weight)
            captured_next_k3 = captured_k3[next_layer]
            observation = np.asarray(
                layer_array(next_layer, "rho3"), dtype=np.float64
            )[:, None] * adjoint[next_layer]
            observation = normalize_columns(observation)
            u, basis_metrics = build_basis(
                lam,
                p,
                q,
                pushed_p,
                pushed_q,
                repeated,
                weight,
                observation,
                state_probe_a[layer],
                state_probe_b[layer],
            )
            quotient_core, raw_core, lift_core = project_cp_quotient_core(
                lam, p, q, repeated, u, ACTION_RANK, batch=CP_BATCH
            )
            quotient_core32 = np.asarray(quotient_core, dtype=np.float32)
            raw_core32 = np.asarray(raw_core, dtype=np.float32)
            quotient_candidate, quotient_metrics = evaluate_core(
                u,
                quotient_core32,
                target,
                weight,
                next_layer,
                direct_next_k21,
                direct_next_k3,
                captured_next_k3,
                weights,
            )
            raw_candidate, raw_metrics = evaluate_core(
                u,
                raw_core32,
                target,
                weight,
                next_layer,
                direct_next_k21,
                direct_next_k3,
                captured_next_k3,
                weights,
            )
            quotient_k3[next_layer] = quotient_candidate
            raw_k3[next_layer] = raw_candidate
            arrays = {
                "u": save_array(layer_output / "u.npy", u),
                "core_quotient": save_array(
                    layer_output / "core_quotient.npy", quotient_core32
                ),
                "core_raw_control": save_array(
                    layer_output / "core_raw_control.npy", raw_core32
                ),
                "core_lift": save_array(
                    layer_output / "core_lift.npy", np.asarray(lift_core, dtype=np.float32)
                ),
                "quotient_candidate_k3_next": save_array(
                    layer_output / "quotient_candidate_k3_next.npy",
                    np.asarray(quotient_candidate, dtype=np.float64),
                ),
                "raw_candidate_k3_next": save_array(
                    layer_output / "raw_candidate_k3_next.npy",
                    np.asarray(raw_candidate, dtype=np.float64),
                ),
            }
            receipt_layers[f"layer_{layer:02d}"] = {
                "post_layer_zero_based": layer,
                "next_layer_zero_based": next_layer,
                "cp_rank": int(lam.shape[0]),
                "basis": basis_metrics,
                "same_numpy_control": {
                    "current_direct_vs_captured_k21_mse": mse(
                        repeated - captured_repeated
                    ),
                    "current_direct_vs_captured_k21_max_abs": float(
                        np.max(np.abs(repeated - captured_repeated))
                    ),
                    "next_direct_vs_canonical_k3_mse": mse(
                        direct_next_k3 - captured_next_k3
                    ),
                    "next_direct_vs_canonical_k3_max_abs": float(
                        np.max(np.abs(direct_next_k3 - captured_next_k3))
                    ),
                },
                "quotient": quotient_metrics,
                "raw_control": raw_metrics,
                "quotient_core_vs_raw_mse": mse(quotient_core - raw_core),
                "arrays": arrays,
            }
            del (
                lam,
                p,
                q,
                pushed_p,
                pushed_q,
                repeated,
                direct_next_k21,
                quotient_core,
                raw_core,
                lift_core,
            )

    top_arrays = {
        "captured_full_k3": save_array(
            temporary / "captured_full_k3.npy", captured_k3
        ),
        "quotient_k3": save_array(temporary / "quotient_k3.npy", quotient_k3),
        "raw_control_k3": save_array(
            temporary / "raw_control_k3.npy", raw_k3
        ),
    }
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "q24-cp3-quotient-nested-ht-source-manifest-v1",
        "status": "COMPLETE_SOURCE_ONLY_NONPROMOTABLE_QUARANTINE",
        "r": RANK,
        "k": ACTION_RANK,
        "seed": SEED,
        "action_probes": ACTION_PROBES,
        "state_probes": STATE_PROBES,
        "cp_batch": CP_BATCH,
        "score_gate": {
            "combined_flops": 551_371_243_519,
            "raw_mse_at_most": 3.060640051205329e-8,
            "leader_adjusted": 7.67408396767286e-9,
        },
        "bindings": {
            "preregister_sha256": sha256(PREREGISTER),
            "state_source_sha256": sha256(STATE_SOURCE),
            "runner_sha256": sha256(Path(__file__)),
            "quarantine_manifest_sha256": sha256(QUARANTINE_MANIFEST),
            "failed_parity_sha256": sha256(FAILED_PARITY),
            "weights_sha256": sha256(WEIGHTS),
        },
        "access": {
            "target_or_error_arrays_opened": [],
            "observer_selection": False,
            "network": False,
            "training": False,
            "fitting": False,
            "submission": False,
        },
        "wall_seconds": elapsed,
        "threads": args.threads,
        "top_arrays": top_arrays,
        "layers": receipt_layers,
    }
    receipt_path = temporary / "SOURCE_MANIFEST.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (temporary / "RUNNING.json").unlink()
    os.replace(temporary, output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
