#!/usr/bin/env python3
"""Independently verify the terminal reference evidence and disposition."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


BOUND = {
    "REFERENCE_EXECUTION_TRANSCRIPT.txt": "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28",
    "REFERENCE_TERMINAL_FAILURE_RECEIPT.json": "4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157",
    "COMMON_DRIVER_SEAL.json": "776859ea1d510d7e7178533c888100fad23ac82d7412a03e1898bf77b4f703f0",
    "COMMON_DRIVER_MANIFEST.json": "b1c26b795b249bf02b6cc0429ec09f49a617e8e62ba0b6d02d0c2dcac0b83670",
    "run_reference_capture.py": "f5732ac16c9246b25335382d85bcd86c4b653b3b21a3bc6bd14389d61b3b1f3c",
    "run_common_driver_gate.py": "e0acba52fa730bc6d448ce873e953f0ddf50c6a908f5c8e69c5b02ff14a5ec0f",
    "verify_reference_terminal_failure.py": "d8b54ddd75af2e82c1217a621bb9943d4b524b367e3c1d4bd1cd9fce1b2cc86c",
    "recover_reference_terminal_failure.py": "5a0f711efe70ac5cbf7d72309258501f2890e05cf478be897b26fe7a3d2a0ee7",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    "reference_rank35_first.npy": "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    "reference_rank279.npy": "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    "reference_rank35_repeat.npy": "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
}
for relative, expected in BOUND.items():
    observed = sha256(HERE / relative)
    assert observed == expected, (relative, observed, expected)

transcript_path = HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt"
receipt_path = HERE / "REFERENCE_TERMINAL_FAILURE_RECEIPT.json"
transcript = transcript_path.read_text()
receipt = json.loads(receipt_path.read_text())
assert receipt["status"] == "FAIL_TERMINAL_NO_RETRY"
assert receipt["original_protocol_closed"] is True
assert receipt["reference_science_calls_completed"] == 3
assert receipt["reference_science_gates"] == "ALL_THREE_REPORTED_CONJUNCTIVE_PASS"
assert receipt["reference_receipt_status"] == "NOT_WRITTEN_DUE_HARNESS_NAMEERROR"
assert receipt["process_exit_code"] == 1
assert receipt["candidate_launched"] is False
assert receipt["candidate_gate_status"] == "NOT_RUN"
assert receipt["call_order"] == [35, 279, 35]

trial_lines = [
    line for line in transcript.splitlines()
    if line.startswith('{"exact_outer_wall_seconds"')
]
assert len(trial_lines) == 3
trials = [json.loads(line) for line in trial_lines]
expected_flops = [1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552]
expected_outer = [9.890440291725099, 10.023627415765077, 9.825676208361983]
expected_meter = [9.890441291965544, 10.023652957752347, 9.825698040891439]
expected_residual = [0.3825201806612313, 0.3858333397656679, 0.37958328472450376]
expected_rss = [2_950_840_320, 3_305_242_624, 3_334_782_976]
assert [row["ordinal"] for row in trials] == [0, 1, 2]
assert [row["role_rank"] for row in trials] == [35, 279, 35]
assert [row["literal_meter"]["flops_used"] for row in trials] == expected_flops
assert [row["literal_operation_sum"] for row in trials] == expected_flops
assert [row["exact_outer_wall_seconds"] for row in trials] == expected_outer
assert [row["literal_meter"]["wall_time_s"] for row in trials] == expected_meter
assert [row["literal_meter"]["residual_wall_time_s"] for row in trials] == expected_residual
assert [row["maximum_process_rss_bytes"] for row in trials] == expected_rss
assert all(row["gates"]["conjunctive_pass"] for row in trials)
assert all(all(row["gates"].values()) for row in trials)
assert expected_residual[1] < 0.4 and expected_residual[1] > 0.385

expected_array_hashes = [
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
    "6f94214ffbcbac95656dd72687d771ff349108e0e6f7625d8eff8c3243e60c50",
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
]
for ordinal, row in enumerate(trials):
    path = HERE / row["output_file"]
    value = np.load(path, allow_pickle=False)
    assert value.dtype == np.float32
    assert value.shape == (16, 1024)
    assert bool(np.isfinite(value).all())
    assert sha256(path) == row["output_file_sha256"]
    assert array_sha256(value) == row["output_array_sha256"] == expected_array_hashes[ordinal]
    summary = receipt["trials"][ordinal]
    assert summary["printed_trial_json_sha256"] == sha256_bytes(trial_lines[ordinal].encode())
    operations = json.dumps(
        row["literal_meter"]["operations"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert summary["literal_operations_sha256"] == sha256_bytes(operations)
    assert summary["literal_flops"] == expected_flops[ordinal]
    assert summary["literal_operation_sum"] == expected_flops[ordinal]
    assert summary["gates"] == row["gates"]

assert (HERE / "reference_rank35_first.npy").read_bytes() == (
    HERE / "reference_rank35_repeat.npy"
).read_bytes()

# The traceback proves the success finalizer was reached after ordinal 2, then
# failed only when evaluating the invalid lowercase literal.
assert 'finish("PASS_REFERENCE_CAPTURE", seal, trials)' in transcript
assert 'line 222, in main' in transcript
assert 'line 138, in finish' in transcript
assert '"board_opened": false' in transcript
assert "NameError: name 'false' is not defined. Did you mean: 'False'?" in transcript

reference_source = (HERE / "run_reference_capture.py").read_text()
candidate_source = (HERE / "run_common_driver_gate.py").read_text()
assert '"board_opened": false,' in reference_source
assert '"board_opened": false,' in candidate_source
assert '"submission": false,' in candidate_source

# Python accepts the invalid spellings as Name nodes, explaining why parse-only
# static checks did not catch the runtime receipt defect.
for source in (reference_source, candidate_source):
    loaded_names = {
        node.id for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "false" in loaded_names

assert not (HERE / "REFERENCE_RESULT.json").exists()
assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
assert not list(HERE.glob("candidate_rank*.npy"))

audit = (HERE / "INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md").read_text()
for required in (
    "the three reference science calls all passed",
    "original sealed\nprotocol nevertheless failed terminally and is closed",
    "AUTHORIZE design, static audit, and sealing of a fresh recovery descendant",
    "do not yet authorize its candidate execution",
    "zero additional reference estimator",
):
    assert required in audit, required

print(json.dumps({
    "status": "PASS_INDEPENDENT_REFERENCE_TERMINAL_AUDIT",
    "reference_science": "ALL_THREE_CALLS_PASS",
    "original_protocol": "FAIL_TERMINAL_NO_RETRY_CLOSED",
    "candidate_calls": 0,
    "disposition": "AUTHORIZE_RECOVERY_DESCENDANT_PREPARATION_ONLY",
}, indent=2, sort_keys=True))
