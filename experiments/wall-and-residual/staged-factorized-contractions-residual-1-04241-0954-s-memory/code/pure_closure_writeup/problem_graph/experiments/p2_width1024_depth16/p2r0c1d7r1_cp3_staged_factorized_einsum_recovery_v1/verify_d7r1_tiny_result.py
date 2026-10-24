#!/usr/bin/env python3
"""Read-only verifier for the sealed D7R1 tiny-runtime PASS result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
RESULT = HERE / "D7R1_TINY_RUNTIME_RESULT.json"
MARKER = HERE / "D7R1_TINY_ATTEMPT_MARKER.json"
BIND_JOURNAL = HERE / "D7R1_BIND_INVENTORY_PREASSERT.json"
SEAL = HERE / "D7R1_TINY_RUNTIME_LAUNCH_SEAL.json"
EXPECTED = {
    RESULT: "24066dd50eb5fb00230cc6629b981cecf7593f7e459af50791b566455fb77427",
    MARKER: "7118e5c21f8b7494e5c0e89baf0a0c9eb4d08cda88bfcd7f227b6c249e06904d",
    BIND_JOURNAL: "f909348719a3a5474a2e5c47c59b20cc44ef88f10af632ab964ae5f149a698f5",
    SEAL: "22da348b8297ea65bf115ab6ff4090dffabbf33072df8a3731abf1cc68945148",
    HERE / "run_tiny_runtime_discriminator.py": "1222bdbc067ae28bdf90c0baced1687c5dbeb3624fe24b4f6c8f9a21f3a4a439",
    HERE / "staged_winograd_kernel.py": "81d14f2ca983c2820ab078d15a3a8a861427ac5a8b1b5e82828300256648bbe9",
    HERE / "ACCOUNTING_AND_RESOURCE_MANIFEST.json": "94ad1b2e89a58417bf91cb4fd844ac1d02674b58a2738b0218bf7cc63eb145ec",
    HERE / "DESIGN.md": "3ccdefd80d69817f1ad8519124f2f0def2f03e95a5c96385d921edfdc97b1b2e",
    HERE / "verify_design_static.py": "27a067afcee05558f2d066bfc7ee594752dce8fa62a5c7166353b97ebed9a816",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)
    assert not (HERE / "D7R1_RUNTIME_EXCEPTION_RECEIPT.json").exists()

    result = json.loads(RESULT.read_text())
    marker = json.loads(MARKER.read_text())
    journal = json.loads(BIND_JOURNAL.read_text())
    assert result["schema"] == "p2r0c1d7r1-tiny-runtime-discriminator-v1"
    assert result["status"] == "PASS_D7R1_TINY_RUNTIME_ADMISSION_PREPARATION_ONLY"
    assert result["node"] == marker["node"] == journal["node"] == "P2R0C1D7R1"
    assert result["disposition"] == "AUTHORIZE_DESIGN_PREPARATION_ONLY_NO_EXACT_SHAPE"
    assert result["residual_gate_pass"] is True
    assert result["residual_delta_gate_seconds"] == 0.01
    assert all(result["checks"].values())

    expected_bind = {
        "calls": {"asarray": 3},
        "costs": {"asarray": 168},
        "flops": 168,
    }
    assert result["coefficient_binding_meter"] == expected_bind
    assert journal["observed"] == journal["expected"] == expected_bind
    assert journal["status"] == "PERSISTED_BEFORE_BIND_ASSERTION"
    assert marker["status"] == "CLAIMED_PERMANENTLY"
    assert marker["launch_owner"] == "/root"
    assert marker["launch_seal_sha256"] == EXPECTED[SEAL]

    candidate = result["candidate_residual_wall_seconds"]
    direct = result["direct_residual_wall_seconds"]
    deltas = result["candidate_minus_direct_residual_seconds"]
    assert len(candidate) == len(direct) == len(deltas) == 8
    assert all(math.isfinite(value) for value in candidate + direct + deltas)
    for candidate_value, direct_value, delta in zip(candidate, direct, deltas):
        assert math.isclose(
            candidate_value - direct_value, delta, rel_tol=0.0, abs_tol=1e-18
        )
    assert statistics.median(candidate) == 0.0076775806955993176
    assert statistics.median(direct) == 0.000426954822614789
    assert statistics.median(deltas) == 0.007248477544635534 < 0.01

    assert result["candidate_roster_meter"] == {
        "calls": {"copyto": 58, "einsum": 696, "matmul": 58, "reshape": 174},
        "costs": {
            "copyto": 237568,
            "einsum": 99528000,
            "matmul": 15596896,
            "reshape": 712704,
        },
        "flops": 116075168,
    }
    assert result["direct_roster_meter"] == {
        "calls": {"matmul": 58},
        "costs": {"matmul": 30171136},
        "flops": 30171136,
    }
    gate = result["kernel_normalized_maximum_absolute_delta_vs_float64_gate"]
    assert gate == 1.0112751933930056e-4
    assert result["gapped_drift_vs_float64"]["normalized_maximum_absolute"] <= gate
    assert result["maximum_shape_drift_vs_float64"]["normalized_maximum_absolute"] <= gate
    assert result["rss"] == {
        "baseline_bytes": 44662784,
        "increment_bytes": 10371072,
        "peak_bytes": 55033856,
        "scope": "tiny process high-water only; exact-shape RSS remains unmeasured",
    }

    bindings = result["bindings"]
    assert bindings["attempt_marker_sha256"] == EXPECTED[MARKER]
    assert bindings["bind_preassert_sha256"] == EXPECTED[BIND_JOURNAL]
    assert bindings["launch_seal_sha256"] == EXPECTED[SEAL]
    assert bindings["runner_sha256"] == EXPECTED[HERE / "run_tiny_runtime_discriminator.py"]
    assert bindings["source_sha256"] == EXPECTED[HERE / "staged_winograd_kernel.py"]
    assert result["estimator_imported"] is False
    assert result["estimator_calls"] == result["fixture_calls"] == 0
    assert result["exact_shape_calls"] == result["common_driver_calls"] == 0
    assert result["targets_scores_names_board_opened"] is False
    assert result["production"] is result["submission"] is result["network"] is False
    print("PASS_D7R1_TINY_RESULT_SEALED_PREPARATION_ONLY")


if __name__ == "__main__":
    main()
