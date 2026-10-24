#!/usr/bin/env python3
"""Verify the read-only residual/synchronization protocol audit."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = HERE / "RESIDUAL_SYNC_PROTOCOL_AUDIT.json"
D4_SOURCE = ROOT / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
D4_RESULT = ROOT / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "D4_CANDIDATE_RESULT.json"
PACKED_DIR = ROOT / "public_q24_d4_lifetime_packed_full_gate_v1"
PACKED_RESULT = PACKED_DIR / "FULL_GATE_RESULT.json"
PACKED_CLOSURE = PACKED_DIR / "TERMINAL_CLOSURE_RECEIPT.json"
DATA_GATE = ROOT / "data_gate" / "pyproject.toml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_truth_calls(source: str) -> list[tuple[int, str, str]]:
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"float", "bool"}:
            continue
        text = ast.get_source_segment(source, node) or ""
        if "fnp" in text:
            calls.append((node.lineno, node.func.id, text))
    return sorted(calls)


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    d4 = json.loads(D4_RESULT.read_text())
    packed = json.loads(PACKED_RESULT.read_text())
    closure = json.loads(PACKED_CLOSURE.read_text())
    pins = audit["bindings"]
    assert pins == {
        "d4_source_sha256": sha256(D4_SOURCE),
        "d4_result_sha256": sha256(D4_RESULT),
        "packed_full_gate_result_sha256": sha256(PACKED_RESULT),
        "packed_terminal_closure_sha256": sha256(PACKED_CLOSURE),
        "data_gate_pyproject_sha256": sha256(DATA_GATE),
    }
    pyproject = DATA_GATE.read_text()
    for pin in ("flopscope==0.12.0", "flopscope-server==0.12.0", "whestbench==0.16.0"):
        assert pin in pyproject

    prior = d4["trials"][0]
    prior_calls = sum(item["calls"] for item in prior["literal_meter"]["operations"].values())
    observed = audit["same_parent_same_inventory_observation"]
    assert prior["candidate_flops"] == observed["prior_d4_call"]["flops"]
    assert prior_calls == observed["prior_d4_call"]["calls"]
    assert prior["literal_meter"]["residual_wall_time_s"] == observed["prior_d4_call"]["residual_seconds"]

    match = re.search(
        r'stdout=\'(?P<record>\{.*?\})\\n\'; stderr=', packed["reason"]
    )
    assert match is not None
    parent = json.loads(match.group("record"))
    assert parent["flops"] == observed["packed_gate_parent_call"]["flops"]
    assert parent["total_calls"] == observed["packed_gate_parent_call"]["calls"]
    assert parent["residual"] == observed["packed_gate_parent_call"]["residual_seconds"]
    assert packed["status"] == "FAIL_TERMINAL"
    assert closure["decision"] == "CLOSE_PACKED_D4_FULL_GATE_WITHOUT_CANDIDATE_CALL"
    assert observed["candidate_calls"] == 0
    assert abs(
        observed["packed_gate_parent_call"]["residual_seconds"]
        - observed["prior_d4_call"]["residual_seconds"]
        - observed["residual_delta_seconds"]
    ) < 1e-15

    sites = remote_truth_calls(D4_SOURCE.read_text())
    assert [(line, kind) for line, kind, _ in sites] == [
        (721, "float"), (1884, "bool"), (8592, "bool"), (8606, "bool")
    ]
    assert audit["d4_remote_scalar_sync_surface"]["static_sites"] == len(sites) == 4
    assert sum(
        item["dynamic_count"]
        for item in audit["d4_remote_scalar_sync_surface"]["sites"]
    ) == audit["d4_remote_scalar_sync_surface"]["dynamic_sites_per_depth16_public_predict"] == 17
    assert audit["scope"] == {
        "estimator_imports": 0,
        "estimator_calls": 0,
        "public_rows_opened": 0,
        "targets_opened": 0,
        "scores_opened": 0,
        "network": False,
        "submission": False,
    }
    print(json.dumps({
        "status": "PASS_PROTOCOL_RECLASSIFICATION_NO_NUMERICAL_RUN",
        "audit_sha256": sha256(AUDIT),
        "same_parent_inventory": {
            "flops": parent["flops"],
            "calls": parent["total_calls"],
        },
        "residuals": {
            "prior": observed["prior_d4_call"]["residual_seconds"],
            "later_parent": observed["packed_gate_parent_call"]["residual_seconds"],
            "delta": observed["residual_delta_seconds"],
        },
        "remote_truth_static_sites": len(sites),
        "remote_truth_dynamic_sites": 17,
        "scope": audit["scope"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
