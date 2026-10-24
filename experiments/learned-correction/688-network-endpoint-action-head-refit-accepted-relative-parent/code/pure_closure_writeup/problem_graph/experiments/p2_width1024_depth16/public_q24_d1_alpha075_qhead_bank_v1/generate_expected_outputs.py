#!/usr/bin/env python3
"""Generate target-free offline authorities for the D1 full gate.

This is intentionally a separate process from the integrated candidate gate,
so the latter retains the established cold/hot/hot [35,279,35] meter pattern.
It makes two unique closure calls, opens weights only, and applies an
independent NumPy spelling of the exact frozen alpha=.75 J and q-head laws.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "estimator.py"
PREREGISTRATION = HERE / "FULL_GATE_PREREGISTRATION.json"
PARITY = HERE / "TRANSPORT_PARITY_RESULT.json"
OUTPUT = HERE / "expected_outputs"
SOURCE_SHA256 = "a6e332c5372839409340b100484d81e423f23239849e0c9de11b72669f9367ea"
PREREGISTRATION_SHA256 = "e3b91a6b4e5dd6996b3cc1bd547d843676b01f4eda202b6fdfad14bcc6f7c1b9"
PARITY_SHA256 = "ca4f5f7e891d0908b41df33e0123bcc31c9b56ba4fd3a4fc0d5d304843a0756c"
FIXTURES = {
    35: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy",
        "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    ),
    279: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0279/weights.npy",
        "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298",
    ),
}
PHASE2_BUDGET = 2**41
MAX_WALL_SECONDS = 120.0
MAX_RSS_BYTES = 8 * (1 << 30)


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
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def load_module():
    spec = importlib.util.spec_from_file_location("d1_alpha075_expected_authority", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fixture(role_rank: int) -> tuple[MLP, np.ndarray]:
    path, expected = FIXTURES[role_rank]
    if sha256(path) != expected:
        raise AssertionError(f"fixture weights changed: {role_rank}")
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError(f"fixture schema changed: {role_rank}")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-target-free-weight-fixture",
    )
    mlp.validate()
    return mlp, weights


def parameters(payload: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        for name, value in payload.items()
    }


def silu(value: np.ndarray) -> np.ndarray:
    one = np.float32(1.0)
    return value / (one + np.exp(-value))


def offline_recurrence(
    raw: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    jpayload: dict[str, object],
    qpayload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy spelling of the frozen autonomous recurrences."""
    p = parameters(jpayload)
    q = parameters(qpayload)
    states = np.zeros((1024, 4), dtype=np.float32)
    energies = np.zeros_like(states)
    qstate = np.zeros(1024, dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    outputs: list[np.ndarray] = []
    qoutputs: list[np.ndarray] = []
    for layer in range(16):
        if layer < 2:
            states = np.zeros_like(states)
            energies = np.zeros_like(states)
            outputs.append(states[:, 0].copy())
            qoutputs.append(qstate.copy())
            continue
        if layer == 2:
            messages = np.zeros_like(states)
            energies = np.zeros_like(states)
        else:
            layer_weights = np.asarray(weights[layer], dtype=np.float32)
            messages = (states.T @ layer_weights).T
            energies = ((states * states).T @ (layer_weights * layer_weights)).T
            phi = raw[layer, :, 5, None]
            messages = phi * messages
            energies = (phi * phi) * energies
        tail = np.clip(
            (raw[layer, :, 1:] - p["centers"][layer]) / p["scales"][layer],
            np.float32(-8.0), np.float32(8.0),
        )
        rms = p["layer_rms"][layer]
        rms2 = rms * rms
        features = np.concatenate(
            (
                raw[layer, :, :1],
                tail,
                np.clip(messages / rms, np.float32(-8.0), np.float32(8.0)),
                np.clip(
                    messages / np.maximum(sigma[layer, :, None] * rms, np.float32(2.0**-30)),
                    np.float32(-8.0), np.float32(8.0),
                ),
                np.clip(
                    np.log1p(np.maximum(energies / rms2, np.float32(0.0))),
                    np.float32(0.0), np.float32(8.0),
                ),
                np.clip(
                    messages / np.sqrt(energies + np.float32(1.0e-4) * rms2),
                    np.float32(-8.0), np.float32(8.0),
                ),
            ),
            axis=1,
        )
        hidden = silu(features @ p["trunk0_weight"].T + p["trunk0_bias"])
        hidden = silu(hidden @ p["trunk2_weight"].T + p["trunk2_bias"])
        source_standard = hidden @ p["source_weight"][layer] + p["source_bias"][layer]
        source = (sigma[layer] * rms)[:, None] * source_standard
        gate_standard = hidden @ p["gate_weight"][layer] + p["gate_bias"][layer]
        dynamic_gain = np.float32(1.0) + np.float32(0.25) * np.tanh(
            gate_standard + p["transport_delta"][layer]
        )
        mixing = identity + np.float32(0.12) * np.tanh(p["mix_delta"][layer])
        states = (dynamic_gain * messages) @ mixing + source

        if layer >= 10:
            qmessage = raw[layer, :, 5] * (qstate @ layer_weights)
            qrms = q["layer_rms"][layer]
            qtail = np.clip(
                (raw[layer, :, 1:] - q["centers"][layer]) / q["scales"][layer],
                np.float32(-8.0), np.float32(8.0),
            )
            base_q = states[:, 0] - messages[:, 0]
            qfeatures = np.concatenate(
                (
                    raw[layer, :, :1],
                    qtail,
                    np.clip(states[:, 0] / qrms, np.float32(-8.0), np.float32(8.0))[:, None],
                    np.clip(
                        base_q / np.maximum(sigma[layer] * qrms, np.float32(2.0**-30)),
                        np.float32(-8.0), np.float32(8.0),
                    )[:, None],
                ),
                axis=1,
            )
            qhidden = silu(qfeatures @ q["trunk0_weight"].T + q["trunk0_bias"])
            qhidden = silu(qhidden @ q["trunk2_weight"].T + q["trunk2_bias"])
            slot = layer - 10
            qstandard = qhidden @ q["head_weight"][slot] + q["head_bias"][slot]
            qsource = sigma[layer] * qrms * qstandard
            qstate = qmessage + qsource
        outputs.append(states[:, 0].copy())
        qoutputs.append(qstate.copy())
    return (
        np.ascontiguousarray(np.stack(outputs), dtype=np.float32),
        np.ascontiguousarray(np.stack(qoutputs), dtype=np.float32),
    )


def write_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise AssertionError("candidate source changed")
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("full-gate preregistration changed")
    if sha256(PARITY) != PARITY_SHA256:
        raise AssertionError("transport parity evidence changed")
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if parity.get("status") != "PASS_TARGET_FREE_TRANSPORT_PARITY":
        raise AssertionError("transport parity status changed")
    module = load_module()
    temporary = HERE / f".expected_outputs.{os.getpid()}.tmp"
    temporary.mkdir(exist_ok=False)
    trials: list[dict[str, object]] = []
    expected: dict[int, np.ndarray] = {}
    with threadpool_limits(limits=2):
        for ordinal, role_rank in enumerate((35, 279)):
            mlp, weights = load_fixture(role_rank)
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"d1_alpha075_offline_authority_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                host_value, raw_value, sigma_value = module._predict(
                    mlp, return_native_tape=True
                )
            outer_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            operation_sum = sum(
                int(item["flop_cost"]) for item in summary["operations"].values()
            )
            host = np.ascontiguousarray(np.asarray(host_value), dtype=np.float32)
            raw = np.ascontiguousarray(np.asarray(raw_value), dtype=np.float32)
            sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.float32)
            jvalue, qvalue = offline_recurrence(
                raw, sigma, weights, module._TRANSPORT_PAYLOAD, module._QHEAD_PAYLOAD
            )
            output = np.ascontiguousarray(
                np.maximum(host + jvalue + qvalue, np.float32(0.0)), dtype=np.float32
            )
            if output.shape != (16, 1024) or output.dtype != np.float32:
                raise AssertionError("offline authority schema failed")
            if not np.isfinite(output).all():
                raise FloatingPointError("offline authority is nonfinite")
            if operation_sum != int(summary["flops_used"]):
                raise AssertionError("authority operation sum mismatch")
            if int(summary["flops_used"]) >= PHASE2_BUDGET:
                raise AssertionError("authority closure exceeded budget")
            if outer_wall >= MAX_WALL_SECONDS:
                raise TimeoutError("authority closure exceeded wall bound")
            if resident_bytes() >= MAX_RSS_BYTES:
                raise MemoryError("authority closure exceeded RSS bound")
            expected[role_rank] = output
            trials.append(
                {
                    "ordinal": ordinal,
                    "role_rank": role_rank,
                    "literal_meter": summary,
                    "literal_operation_sum": operation_sum,
                    "outer_wall_seconds": outer_wall,
                    "maximum_process_rss_bytes": resident_bytes(),
                    "output_array_sha256": array_sha256(output),
                }
            )
            print(json.dumps(trials[-1], sort_keys=True), flush=True)

    call_order = (35, 279, 35)
    files: list[dict[str, object]] = []
    for ordinal, role_rank in enumerate(call_order):
        path = temporary / f"expected_{ordinal}_rank{role_rank}.npy"
        write_array(path, expected[role_rank])
        files.append(
            {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "name": path.name,
                "file_sha256": sha256(path),
                "array_sha256": array_sha256(expected[role_rank]),
            }
        )
    if files[0]["array_sha256"] != files[2]["array_sha256"]:
        raise AssertionError("offline rank35 repeat authority mismatch")
    manifest = {
        "schema": "q24-d1-alpha075-qhead-offline-expected-outputs-v1",
        "status": "PASS_TARGET_FREE_EXPECTED_OUTPUT_GENERATION",
        "call_order": list(call_order),
        "trials": trials,
        "files": files,
        "bindings": {
            "candidate_source_sha256": sha256(SOURCE),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "transport_parity_result_sha256": sha256(PARITY),
            "generator_sha256": sha256(Path(__file__)),
            "fixture_weight_sha256": {
                str(rank): sha256(path) for rank, (path, _) in FIXTURES.items()
            },
        },
        "access_declarations": {
            "weights_only": True,
            "target_columns_opened": [],
            "scores_opened": [],
            "names_opened": [],
            "board_opened": False,
            "submission": False,
            "training": False,
            "network": False,
        },
    }
    manifest_path = temporary / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT)
    print(json.dumps({"status": manifest["status"], "manifest_sha256": sha256(OUTPUT / "MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
