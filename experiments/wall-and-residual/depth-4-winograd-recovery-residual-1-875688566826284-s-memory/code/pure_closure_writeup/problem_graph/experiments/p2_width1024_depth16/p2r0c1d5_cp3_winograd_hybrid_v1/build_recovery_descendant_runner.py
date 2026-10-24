#!/usr/bin/env python3
"""Mechanically derive the recovery-descendant runner from the frozen runner."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_common_driver_gate.py"
OUTPUT = HERE / "run_recovery_descendant_candidate_gate.py"
SOURCE_SHA256 = "e0acba52fa730bc6d448ce873e953f0ddf50c6a908f5c8e69c5b02ff14a5ec0f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError((old[:80], text.count(old)))
    return text.replace(old, new)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    assert sha256(SOURCE) == SOURCE_SHA256
    text = SOURCE.read_text()
    text = replace_once(
        text,
        '"""One-shot target-free P2R0C1D5 common-driver gate."""',
        '"""One-shot zero-reference-call P2R0C1D5R1 candidate gate."""',
    )
    replacements = {
        'MANIFEST = HERE / "COMMON_DRIVER_MANIFEST.json"': 'MANIFEST = HERE / "RECOVERY_DESCENDANT_MANIFEST.json"',
        'SEAL = HERE / "COMMON_DRIVER_SEAL.json"': 'SEAL = HERE / "RECOVERY_DESCENDANT_LAUNCH_SEAL.json"',
        'REFERENCE_RESULT = HERE / "REFERENCE_RESULT.json"': 'REFERENCE_RESULT = HERE / "RECOVERED_REFERENCE_EVIDENCE.json"',
        'RESULT = HERE / "COMMON_DRIVER_RESULT.json"': 'RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"',
        'HERE / "candidate_rank35_first.npy"': 'HERE / "recovery_candidate_rank35_first.npy"',
        'HERE / "candidate_rank279.npy"': 'HERE / "recovery_candidate_rank279.npy"',
        'HERE / "candidate_rank35_repeat.npy"': 'HERE / "recovery_candidate_rank35_repeat.npy"',
        '"p2r0c1d5_candidate_gate"': '"p2r0c1d5r1_candidate_gate"',
        'namespace=f"p2r0c1d5_candidate_{ordinal}"': 'namespace=f"p2r0c1d5r1_candidate_{ordinal}"',
    }
    for old, new in replacements.items():
        text = replace_once(text, old, new)
    text = replace_once(
        text,
        'if seal.get("status") != "FROZEN_BEFORE_FIRST_COMMON_DRIVER_CALL":\n'
        '        raise AssertionError("common-driver seal status changed")',
        'if seal.get("status") != "FROZEN_RECOVERY_DESCENDANT_AUTHORIZED":\n'
        '        raise AssertionError("recovery-descendant launch seal status changed")',
    )

    old_load = '''def load_references(seal: dict[str, object]) -> tuple[dict[str, object], list[np.ndarray]]:
    reference_result = json.loads(REFERENCE_RESULT.read_text())
    if reference_result.get("status") != "PASS_REFERENCE_CAPTURE":
        raise AssertionError("reference capture did not pass")
    bindings = reference_result["bindings"]
    if bindings["common_driver_seal_sha256"] != sha256(SEAL):
        raise AssertionError("reference seal binding changed")
    if bindings["program_sha256"] != seal["sha256"]["run_reference_capture.py"]:
        raise AssertionError("reference runner binding changed")
    if bindings["source_sha256"] != seal["sha256"]["../public_noeigh_card_v1/estimator.py"]:
        raise AssertionError("reference source binding changed")
    if len(reference_result["trials"]) != 3:
        raise AssertionError("reference roster incomplete")
    arrays = []
    for ordinal, path in enumerate(REFERENCE_OUTPUTS):
        trial = reference_result["trials"][ordinal]
        if trial["output_file_sha256"] != sha256(path):
            raise AssertionError(f"reference output file changed: {ordinal}")
        if int(trial["literal_meter"]["flops_used"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference bill changed: {ordinal}")
        array = np.load(path, allow_pickle=False)
        if array_sha256(array) != trial["output_array_sha256"]:
            raise AssertionError(f"reference array changed: {ordinal}")
        arrays.append(np.ascontiguousarray(array, dtype=np.float32))
    if array_sha256(arrays[0]) != array_sha256(arrays[2]):
        raise AssertionError("reference rank35 repeat mismatch")
    return reference_result, arrays
'''
    new_load = '''def load_references(seal: dict[str, object]) -> tuple[dict[str, object], list[np.ndarray]]:
    reference_result = json.loads(REFERENCE_RESULT.read_text())
    if reference_result.get("status") != "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE":
        raise AssertionError("recovered reference evidence status changed")
    if reference_result.get("original_protocol_status") != "FAIL_TERMINAL_NO_RETRY":
        raise AssertionError("original terminal status changed")
    if reference_result.get("original_process_exit_code") != 1:
        raise AssertionError("original process exit changed")
    if reference_result.get("original_candidate_calls") != 0:
        raise AssertionError("candidate was already called")
    if reference_result.get("recovery_reference_estimator_calls") != 0:
        raise AssertionError("recovery made a reference call")
    if reference_result.get("candidate_calls_at_recovery") != 0:
        raise AssertionError("candidate was already called at recovery")
    if tuple(reference_result.get("call_order", ())) != CALL_ORDER:
        raise AssertionError("recovered reference call order changed")
    if reference_result.get("access_declarations") != {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
    }:
        raise AssertionError("recovered reference scope declaration changed")
    if sha256(REFERENCE_RESULT) != seal["sha256"]["RECOVERED_REFERENCE_EVIDENCE.json"]:
        raise AssertionError("recovered reference evidence changed")
    bindings = reference_result["bindings"]
    if bindings["COMMON_DRIVER_SEAL.json"] != seal["sha256"]["COMMON_DRIVER_SEAL.json"]:
        raise AssertionError("original seal binding changed")
    if bindings["run_reference_capture.py"] != seal["sha256"]["run_reference_capture.py"]:
        raise AssertionError("reference runner binding changed")
    if bindings["../public_noeigh_card_v1/estimator.py"] != seal["sha256"]["../public_noeigh_card_v1/estimator.py"]:
        raise AssertionError("reference source binding changed")
    if len(reference_result["trials"]) != 3:
        raise AssertionError("reference roster incomplete")
    arrays = []
    for ordinal, path in enumerate(REFERENCE_OUTPUTS):
        trial = reference_result["trials"][ordinal]
        if trial["ordinal"] != ordinal or trial["role_rank"] != CALL_ORDER[ordinal]:
            raise AssertionError(f"reference roster changed: {ordinal}")
        if trial["output_file"] != path.name:
            raise AssertionError(f"reference output path changed: {ordinal}")
        if trial["output_file_sha256"] != sha256(path):
            raise AssertionError(f"reference output file changed: {ordinal}")
        if int(trial["literal_meter"]["flops_used"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference bill changed: {ordinal}")
        if int(trial["literal_operation_sum"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference operation sum changed: {ordinal}")
        if not trial["gates"]["conjunctive_pass"] or not all(trial["gates"].values()):
            raise AssertionError(f"reference gate changed: {ordinal}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (16, 1024):
            raise AssertionError(f"reference schema changed: {ordinal}")
        if not np.isfinite(array).all():
            raise AssertionError(f"reference finiteness changed: {ordinal}")
        if array_sha256(array) != trial["output_array_sha256"]:
            raise AssertionError(f"reference array changed: {ordinal}")
        arrays.append(np.ascontiguousarray(array, dtype=np.float32))
    if array_sha256(arrays[0]) != array_sha256(arrays[2]):
        raise AssertionError("reference rank35 repeat mismatch")
    return reference_result, arrays
'''
    text = replace_once(text, old_load, new_load)

    old_finish = '''def finish(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None = None,
) -> None:
    result = {
        "schema": "p2r0c1d5-target-free-common-driver-gate-v1",
        "status": status,
        "node": "P2R0C1D5",
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": false,
        "submission": false,
        "trials": trials,
        "aggregate": aggregate,
        "terminal_reason": reason,
        "bindings": {
            "common_driver_seal_sha256": sha256(SEAL),
            "manifest_sha256": sha256(MANIFEST),
            "program_sha256": sha256(Path(__file__)),
            "candidate_source_sha256": sha256(SOURCE),
            "reference_result_sha256": sha256(REFERENCE_RESULT),
        },
    }
    write_json_once(RESULT, result)
    print(json.dumps({"status": status, "reason": reason, "aggregate": aggregate}, sort_keys=True), flush=True)
'''
    new_finish = '''def build_result(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None,
    bindings: dict[str, str],
) -> dict[str, object]:
    return {
        "schema": "p2r0c1d5r1-target-free-recovery-descendant-gate-v1",
        "status": status,
        "node": "P2R0C1D5R1",
        "original_protocol_status": "FAIL_TERMINAL_NO_RETRY",
        "reference_estimator_calls_in_descendant": 0,
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
        "trials": trials,
        "aggregate": aggregate,
        "terminal_reason": reason,
        "bindings": bindings,
    }


def finish(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None = None,
) -> None:
    bindings = {
        "recovery_descendant_launch_seal_sha256": sha256(SEAL),
        "recovery_descendant_manifest_sha256": sha256(MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "candidate_source_sha256": sha256(SOURCE),
        "recovered_reference_evidence_sha256": sha256(REFERENCE_RESULT),
    }
    result = build_result(status, trials, aggregate, reason, bindings)
    write_json_once(RESULT, result)
    print(json.dumps({"status": status, "reason": reason, "aggregate": aggregate}, sort_keys=True), flush=True)
'''
    text = replace_once(text, old_finish, new_finish)
    tree = ast.parse(text)
    forbidden = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in {"false", "true", "null"}
    }
    assert not forbidden
    OUTPUT.write_text(text)
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
