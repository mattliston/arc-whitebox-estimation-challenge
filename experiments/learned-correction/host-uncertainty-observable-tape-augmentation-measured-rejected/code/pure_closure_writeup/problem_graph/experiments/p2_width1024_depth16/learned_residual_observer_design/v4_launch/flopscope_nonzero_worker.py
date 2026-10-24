#!/usr/bin/env python3
"""Target-free Flopscope host/observer trace worker for V4 parity.

The worker accepts only four frozen weight fixtures: two procedural and two
target-blind mini rows.  It never projects a target column.  Outputs live in a
caller-created /private/tmp directory and are deleted by the parent verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DATA_GATE))

from checkpoint_converter import PARAMETER_ORDER, load_numpy_bundle  # noqa: E402


FLOP_BUDGET = 2**41
EXPECTED_FIXTURES = {
    "identity": {"kind": "procedural", "seed": 0},
    "official_seed20260823_mlp0": {
        "kind": "procedural",
        "seed": 1941843782,
        "aggregate_sha256": "86c25816f08c62f389ce885d31e4fa5576411aab2a61c6dc29e858f02ec07ce8",
    },
    "mini_ordinal_0": {
        "kind": "mini",
        "ordinal": 0,
        "seed": 6319981554997072999,
        "weights_sha256": "d524e662eca3d117e9ba164fc3ae51a790dd675c585914a895ec69725fd880d0",
    },
    "mini_ordinal_1": {
        "kind": "mini",
        "ordinal": 1,
        "seed": 2392109381927064353,
        "weights_sha256": "41f9a2a1d715abfc9f0ca535b515fb0aa0ae0edda0f022b3f76d61abfc69ae54",
    },
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERALIZED = _load("v4_frozen_generalized", V3 / "estimator_generalized.py")
BASE = GENERALIZED._BASE


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _raw_weights_hash(weights: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(weights)).cast("B")).hexdigest()


def _aggregate_weights(weights: np.ndarray) -> str:
    digest = hashlib.sha256()
    for layer0, weight in enumerate(weights):
        name = f"mlp0/layer{layer0}".encode("utf-8")
        value = _array_hash(weight).encode("ascii")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(value)
    return digest.hexdigest()


def _fixture(label: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    record = EXPECTED_FIXTURES[label]
    if label == "identity":
        weights = np.stack([np.eye(1024, dtype=np.float32) for _ in range(16)])
    elif label == "official_seed20260823_mlp0":
        stream = np.random.SeedSequence(20260823).spawn(3)[0]
        rng = np.random.default_rng(stream)
        weights = np.stack(
            [
                (rng.standard_normal((1024, 1024)) * math.sqrt(2.0 / 1024)).astype(np.float32)
                for _ in range(16)
            ]
        )
    else:
        import target_blind_mini_v2

        wanted = int(record["ordinal"])
        selected = None
        for row in target_blind_mini_v2.iter_target_blind_mini():
            if row.ordinal == wanted:
                selected = row
                break
        if selected is None:
            raise AssertionError("target-blind mini fixture is absent")
        if selected.decoded_columns != ("mlp_id", "mlp_name", "mlp_seed", "weights"):
            raise AssertionError("mini loader projection changed")
        if selected.mlp_seed != int(record["seed"]):
            raise AssertionError("mini fixture seed changed")
        weights = np.array(selected.weights, dtype=np.float32, order="C", copy=True)
        if _raw_weights_hash(weights) != record["weights_sha256"]:
            raise AssertionError("mini fixture weight hash changed")
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.shape != (16, 1024, 1024) or not np.isfinite(weights).all():
        raise AssertionError("fixture weights changed shape/dtype/finiteness")
    aggregate = _aggregate_weights(weights)
    if record.get("aggregate_sha256") and aggregate != record["aggregate_sha256"]:
        raise AssertionError("procedural fixture aggregate hash changed")
    return weights, int(record["seed"]), {
        "label": label,
        "kind": record["kind"],
        "weights_raw_sha256": _raw_weights_hash(weights),
        "weights_aggregate_sha256": aggregate,
        "target_columns_decoded": 0,
    }


def _np(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = value.view(np.ndarray) if hasattr(value, "view") else np.asarray(value)
    return np.array(array, dtype=dtype, order="C", copy=True)


def _state_hash(layer0: int, cap: int, cp_lambda: Any, cp_p: Any, cp_q: Any, retained: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"entering-cp-v1|layer={layer0}|cap={cap}".encode("ascii"))
    for name, value in (("lambda", cp_lambda), ("p", cp_p), ("q", cp_q), ("retained", retained)):
        array = np.ascontiguousarray(_np(value))
        metadata = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        # ``memoryview.cast`` rejects NumPy arrays with a zero-sized axis on
        # Python 3.11; ``tobytes`` has the required canonical empty payload.
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _host_tape(weights: np.ndarray, cap: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    BASE._CP_CAP = int(cap)
    width = 1024
    analytic_identity = fnp.eye(width, dtype=fnp.float64)
    cp_identity = fnp.eye(width, dtype=fnp.float32)
    mean = fnp.zeros(width, dtype=fnp.float64)
    covariance = analytic_identity
    cp_lambda = fnp.zeros(0, dtype=fnp.float32)
    cp_p = fnp.zeros((width, 0), dtype=fnp.float32)
    cp_q = fnp.zeros((width, 0), dtype=fnp.float32)
    retained = fnp.asarray(1.0, dtype=fnp.float64)
    cp_diag = fnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=fnp.float64)
    raw32_rows: list[np.ndarray] = []
    raw64_rows: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    host_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    cp_diag_rows: list[np.ndarray] = []
    entering_hashes: list[str] = []

    for layer0, weights_raw in enumerate(weights):
        entering_hashes.append(_state_hash(layer0, cap, cp_lambda, cp_p, cp_q, retained))
        cp_diag_rows.append(_np(cp_diag, dtype=np.float64))
        weights32 = fnp.asarray(weights_raw, dtype=fnp.float32)
        weights64 = fnp.asarray(weights_raw, dtype=fnp.float64)
        pre_mean = fnp.transpose(weights64) @ mean
        pre_cov = fnp.einsum("ij,ia,jb->ab", covariance, weights64, weights64)
        pre_var = fnp.maximum(fnp.diag(pre_cov), BASE._VAR_FLOOR)
        sigma = fnp.sqrt(pre_var)
        alpha = pre_mean / sigma
        pdf = fnp.exp(-0.5 * alpha * alpha) / BASE._SQRT_2PI
        cdf = BASE.flops.stats.norm.cdf(alpha)
        coefficients = BASE._hermite_coefficients(alpha, pdf, cdf)
        gaussian_mean = sigma * coefficients[0]
        if int(cp_lambda.shape[0]) == 0:
            pushed_p, pushed_q = cp_p, cp_q
            k3 = fnp.zeros(width, dtype=fnp.float32)
        else:
            pushed_p = fnp.transpose(weights32) @ cp_p
            pushed_q = fnp.transpose(weights32) @ cp_q
            k3 = BASE._cp_k3(cp_lambda, pushed_p, pushed_q)
        reported = fnp.maximum(
            gaussian_mean
            + (fnp.asarray(k3, dtype=fnp.float64) / 6.0)
            * (-(alpha * pdf) / pre_var),
            0.0,
        )
        correlation = pre_cov / fnp.outer(sigma, sigma)
        exact_var = fnp.maximum(
            pre_var * ((1.0 + alpha * alpha) * cdf + alpha * pdf)
            - gaussian_mean * gaussian_mean,
            BASE._VAR_FLOOR,
        )
        edges = BASE._edge_matrices(weights32, width)
        raw = BASE._raw_features(
            layer0,
            width,
            int(cp_lambda.shape[0]),
            retained,
            alpha,
            sigma,
            cdf,
            pdf,
            coefficients,
            gaussian_mean,
            exact_var,
            pre_var,
            k3,
            reported,
            correlation,
            edges,
        )
        raw64 = _np(raw, dtype=np.float64)
        raw64_rows.append(raw64)
        raw32_rows.append(np.ascontiguousarray(raw64, dtype=np.float32))
        edge_rows.append(np.stack([_np(edge, dtype=np.float32) for edge in edges]))
        host_rows.append(_np(fnp.asarray(reported, dtype=fnp.float32), dtype=np.float32))
        sigma_rows.append(_np(fnp.asarray(sigma, dtype=fnp.float32), dtype=np.float32))

        covariance = fnp.zeros((width, width), dtype=fnp.float64)
        correlation_power = correlation
        for order in range(1, BASE._HERMITE_ORDER + 1):
            coefficient = coefficients[order]
            covariance = covariance + fnp.outer(coefficient, coefficient) * correlation_power
            correlation_power = correlation_power * correlation
        covariance = covariance * fnp.outer(sigma, sigma)
        covariance = covariance + fnp.diag(fnp.maximum(exact_var - fnp.diag(covariance), 0.0))
        mean = gaussian_mean
        if layer0 + 1 < 16:
            cp_lambda, cp_p, cp_q, retained, cp_diag = BASE._cp_grow(
                cp_lambda,
                pushed_p,
                pushed_q,
                fnp.asarray(cdf, dtype=fnp.float32),
                fnp.asarray(pre_cov, dtype=fnp.float32),
                fnp.asarray(sigma, dtype=fnp.float32),
                fnp.asarray(pdf, dtype=fnp.float32),
                cp_identity,
            )
    tape = {
        "raw32": np.stack(raw32_rows),
        "raw64": np.stack(raw64_rows),
        "edges": np.stack(edge_rows),
        "host": np.stack(host_rows),
        "sigma": np.stack(sigma_rows),
        "cp_diagnostic": np.stack(cp_diag_rows),
        "entering_cp_hashes": np.asarray(entering_hashes, dtype="U64"),
    }
    report = {
        "entering_cp_state_hashes": entering_hashes,
        "entering_cp_chain_sha256": hashlib.sha256("\n".join(entering_hashes).encode("ascii")).hexdigest(),
        "raw32_sha256": _array_hash(tape["raw32"]),
        "edges_sha256": _array_hash(tape["edges"]),
        "host_sha256": _array_hash(tape["host"]),
        "sigma_sha256": _array_hash(tape["sigma"]),
    }
    return tape, report


def _sigmoid(value: Any) -> Any:
    return 1.0 / (1.0 + fnp.exp(-value))


def _gate_locals(local: Any, gates: Any) -> Any:
    one = fnp.ones(20, dtype=fnp.float32)
    vector = fnp.stack(
        (
            one[0], gates[1], gates[1], one[3], one[4], one[5], one[6], one[7],
            gates[1], gates[1], gates[1], one[11], gates[0], gates[2], gates[2],
            gates[2], gates[3], gates[3], gates[3], gates[3],
        )
    )
    return local * vector[None, :]


def _messages(edges: Any, previous: Any, hidden: int) -> tuple[Any, Any, Any]:
    channels = hidden // 4
    groups, logs, rhos = [], [], []
    for power in range(4):
        prior = previous[:, power * channels : (power + 1) * channels]
        transported = fnp.transpose(edges[power]) @ prior
        rho = fnp.sqrt(fnp.sum(transported * transported) / (1024 * channels) + BASE._MESSAGE_EPS)
        groups.append(fnp.clip(transported / rho, -BASE._MESSAGE_CLIP, BASE._MESSAGE_CLIP))
        logs.append(fnp.clip(fnp.log(rho), -8.0, 8.0))
        rhos.append(rho)
    return fnp.concatenate(groups, axis=1), fnp.stack(logs), fnp.stack(rhos)


def _cell(local: Any, messages: Any, log_rms: Any, params: tuple[Any, ...], hidden: int) -> tuple[Any, Any, Any]:
    (w1, b1, w2, b2, wc, bc, wg, bg, wm, bm, source, _, _) = params
    gates = _sigmoid(source)
    gated_local = _gate_locals(local, gates)
    channels = hidden // 4
    groups, logs = [], []
    for power in range(4):
        gate = gates[4 + power]
        groups.append(messages[:, power * channels : (power + 1) * channels] * gate)
        logs.append(log_rms[power] * gate)
    broadcast = fnp.broadcast_to(fnp.stack(logs)[None, :], (1024, 4))
    cell_input = fnp.concatenate((gated_local, broadcast, fnp.concatenate(groups, axis=1)), axis=1)
    q1 = BASE._silu(cell_input @ fnp.transpose(w1) + b1)
    q2 = BASE._silu(q1 @ fnp.transpose(w2) + b2)
    candidate = fnp.tanh(q2 @ fnp.transpose(wc) + bc)
    update = _sigmoid(q2 @ fnp.transpose(wg) + bg)
    hidden_row = fnp.tanh(messages + update * (candidate - messages))
    unit = fnp.tanh(q2 @ fnp.transpose(wm) + bm)[:, 0]
    return cell_input, hidden_row, unit


def _observer_trace(
    tape: dict[str, np.ndarray],
    bundle_dir: Path,
    arm: str,
    fault_layer0: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, centers, scales, manifest = load_numpy_bundle(bundle_dir)
    hidden = int(manifest["hidden_channels"])
    params = tuple(fnp.asarray(arrays[name], dtype=fnp.float32) for name in PARAMETER_ORDER) + (
        fnp.asarray(centers, dtype=fnp.float64),
        fnp.asarray(scales, dtype=fnp.float64),
    )
    previous = fnp.zeros((1024, hidden), dtype=fnp.float32)
    active = True
    traces: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "standardized", "messages", "log_rms", "hidden", "unit",
            "physical", "predictions",
        )
    }
    latch: list[bool] = []
    for layer0 in range(16):
        host = fnp.asarray(tape["host"][layer0], dtype=fnp.float32)
        if not active:
            standardized = fnp.asarray(tape["raw32"][layer0], dtype=fnp.float32)
            messages = fnp.zeros((1024, hidden), dtype=fnp.float32)
            logs = fnp.zeros(4, dtype=fnp.float32)
            hidden_row = fnp.zeros_like(previous)
            unit = fnp.zeros(1024, dtype=fnp.float32)
            delta = fnp.zeros(1024, dtype=fnp.float32)
            prediction = fnp.maximum(host, fnp.asarray(0.0, dtype=fnp.float32))
        else:
            raw = fnp.asarray(tape["raw64"][layer0], dtype=fnp.float64)
            standardized = BASE._standardize(raw, layer0, params[-2], params[-1])
            if fault_layer0 == layer0:
                fault = _np(standardized, dtype=np.float32)
                fault[0, 0] = np.nan
                standardized = fnp.asarray(fault, dtype=fnp.float32)
            edges = tuple(fnp.asarray(value, dtype=fnp.float32) for value in tape["edges"][layer0])
            if layer0 == 0 or arm == "local_no_recurrence":
                messages = fnp.zeros((1024, hidden), dtype=fnp.float32)
                logs = fnp.zeros(4, dtype=fnp.float32)
                rhos = fnp.ones(4, dtype=fnp.float32)
            else:
                messages, logs, rhos = _messages(edges, previous, hidden)
            cell_input, hidden_row, unit = _cell(standardized, messages, logs, params, hidden)
            delta = (
                fnp.zeros(1024, dtype=fnp.float32)
                if layer0 == 0
                else fnp.asarray(tape["sigma"][layer0], dtype=fnp.float32) * (1.0 / 32.0) * unit
            )
            prediction = fnp.maximum(host + delta, fnp.asarray(0.0, dtype=fnp.float32))
            valid = BASE._one_scalar_latch(
                raw,
                standardized,
                edges,
                messages,
                logs,
                rhos,
                cell_input,
                hidden_row,
                unit,
                delta,
                prediction,
                fnp.asarray(tape["sigma"][layer0], dtype=fnp.float32),
                fnp.asarray(tape["cp_diagnostic"][layer0], dtype=fnp.float64),
                params,
            )
            if not valid:
                active = False
                hidden_row = fnp.zeros_like(previous)
                unit = fnp.zeros(1024, dtype=fnp.float32)
                delta = fnp.zeros(1024, dtype=fnp.float32)
                prediction = fnp.maximum(host, fnp.asarray(0.0, dtype=fnp.float32))
        traces["standardized"].append(_np(standardized, dtype=np.float32))
        traces["messages"].append(_np(messages, dtype=np.float32))
        traces["log_rms"].append(_np(logs, dtype=np.float32))
        traces["hidden"].append(_np(hidden_row, dtype=np.float32))
        traces["unit"].append(_np(unit, dtype=np.float32))
        traces["physical"].append(_np(delta, dtype=np.float32))
        traces["predictions"].append(_np(prediction, dtype=np.float32))
        latch.append(active)
        previous = hidden_row if arm == "recurrent" and active else fnp.zeros_like(previous)
    result = {name: np.stack(rows) for name, rows in traces.items()}
    result["latch"] = np.asarray(latch, dtype=np.bool_)
    return result, {
        "arm": arm,
        "fault_layer0": fault_layer0,
        "latch": latch,
        "prediction_sha256": _array_hash(result["predictions"]),
        "physical_sha256": _array_hash(result["physical"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=tuple(EXPECTED_FIXTURES), required=True)
    parser.add_argument("--cap", type=int, choices=(1536, 1792), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise SystemExit("output must be a new child of an existing temporary directory")
    if not args.output.parent.resolve().as_posix().startswith("/private/tmp/"):
        raise SystemExit("parity output must live below /private/tmp")
    args.output.mkdir(mode=0o700)
    flops.configure(symmetry_warnings=False)
    weights, seed, fixture = _fixture(args.fixture)
    with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as context:
        tape, host_report = _host_tape(weights, args.cap)
        clean_reports: dict[str, Any] = {}
        for arm in ("local_no_recurrence", "recurrent"):
            trace, report = _observer_trace(tape, args.bundle, arm, None)
            np.savez(args.output / f"flops-{arm}.npz", **trace)
            clean_reports[arm] = report
        fault_trace, fault_report = _observer_trace(tape, args.bundle, "recurrent", 2)
        np.savez(args.output / "flops-recurrent-fault-l3.npz", **fault_trace)
    np.savez(args.output / "host-tape.npz", **tape)
    payload = {
        "status": "PASS_TARGET_FREE_FLOPSCOPE_NONZERO_TRACE_WORKER",
        "fixture": fixture,
        "mlp_seed": seed,
        "cap": args.cap,
        "host": host_report,
        "clean": clean_reports,
        "fault": fault_report,
        "flops_used_host_plus_three_observer_traces": context.flops_used,
        "target_values_decoded": 0,
        "fits": 0,
    }
    (args.output / "worker.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
