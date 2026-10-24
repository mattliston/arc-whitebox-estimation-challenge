#!/usr/bin/env python3
"""Independent verifier for the terminal P2R0C1D5R1 execution."""

from __future__ import annotations

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


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


bound = {
    "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt": "afa66d0a953b4d3f0d703bbbb62321dfd3b10df0aec7b716ec6847e3ddaf68b3",
    "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json": "8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0",
    "RECOVERY_DESCENDANT_RESULT.json": "a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549",
    "recovery_candidate_rank35_first.npy": "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
    "verify_descendant_terminal_execution.py": "cb9fbb896d004378f658ccb012bc94dfd59073bc81a3a50eb36e2d913104dd82",
    "recover_descendant_terminal_execution.py": "5f53e72b26c330717e8da7af830a0bc3a8299b8ffff8797b61a6ea08e622c2c3",
    "RECOVERY_DESCENDANT_LAUNCH_SEAL.json": "4da8ddd17b954a9f8f2c0b78ff5f5d90f1ef958444cda2f2b0b7c27a6b1efd33",
    "run_recovery_descendant_candidate_gate.py": "42ca250d6686e5e48b2730063d22dfd3407f870a16dab59be0929f6a5d81928f",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    "RECOVERED_REFERENCE_EVIDENCE.json": "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
}
for relative, expected in bound.items():
    observed = sha256(HERE / relative)
    assert observed == expected, (relative, observed, expected)

transcript = (HERE / "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt").read_text()
receipt = json.loads((HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json").read_text())
result = json.loads((HERE / "RECOVERY_DESCENDANT_RESULT.json").read_text())
trial_lines = [
    line for line in transcript.splitlines()
    if line.startswith('{"drift_vs_exact_incumbent"')
]
status_lines = [
    line for line in transcript.splitlines()
    if line.startswith('{"aggregate"')
]
assert len(trial_lines) == 1
assert len(status_lines) == 1
trial = json.loads(trial_lines[0])
status_line = json.loads(status_lines[0])
assert status_line == {
    "aggregate": {},
    "reason": "candidate call 0 gate failure",
    "status": "FAIL_TERMINAL",
}

assert result["status"] == "FAIL_TERMINAL"
assert result["terminal_reason"] == "candidate call 0 gate failure"
assert result["trials"] == [trial]
assert result["aggregate"] == {}
assert result["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
assert result["reference_estimator_calls_in_descendant"] == 0
assert result["call_order"] == [35, 279, 35]
assert result["bindings"] == {
    "candidate_source_sha256": bound["estimator.py"],
    "program_sha256": bound["run_recovery_descendant_candidate_gate.py"],
    "recovered_reference_evidence_sha256": bound["RECOVERED_REFERENCE_EVIDENCE.json"],
    "recovery_descendant_launch_seal_sha256": bound["RECOVERY_DESCENDANT_LAUNCH_SEAL.json"],
    "recovery_descendant_manifest_sha256": "cafa3d88a46d4fc45dca6cfe52acaf148bb9123e5e88b31d58fb37d0135e24c1",
}
assert result["target_columns_opened"] == []
assert result["scores_opened"] == []
assert result["names_opened"] == []
assert result["board_opened"] is False
assert result["submission"] is False
assert result["training"] is False
assert result["network"] is False

assert receipt["status"] == "FAIL_TERMINAL_NO_RETRY"
assert receipt["candidate_calls_completed"] == 1
assert receipt["failed_ordinal"] == 0
assert receipt["uncalled_ordinals"] == [1, 2]
assert receipt["uncalled_role_ranks"] == [279, 35]
assert receipt["reference_estimator_calls_in_descendant"] == 0
assert receipt["process_exit_code"] == 1
assert receipt["trial"] == trial
assert receipt["disposition"].startswith("CLOSED_THIS_IMPLEMENTATION_FORM")

flops = 1_320_876_655_991
reference_flops = 1_692_484_904_311
saving = 371_608_248_320
assert trial["ordinal"] == 0 and trial["role_rank"] == 35
assert trial["literal_meter"]["flops_used"] == flops
assert trial["literal_operation_sum"] == flops
assert sum(int(item["flop_cost"]) for item in trial["literal_meter"]["operations"].values()) == flops
assert trial["measured_candidate_flops"] == flops
assert trial["measured_reference_flops"] == reference_flops
assert trial["measured_reference_minus_candidate_flops"] == saving == reference_flops - flops
assert trial["exact_outer_wall_seconds"] == 24.356781082693487
assert trial["literal_meter"]["wall_time_s"] == 24.35678375000134
assert trial["literal_meter"]["residual_wall_time_s"] == 1.875688566826284
assert trial["maximum_process_rss_bytes"] == 11_953_569_792
assert trial["gates"]["literal_flops_exact"] is True
assert trial["gates"]["literal_operation_sum_equals_total"] is True
assert trial["gates"]["measured_exact_delta_vs_measured_incumbent"] is True
assert trial["gates"]["final_layer_maximum_absolute_delta"] is True
assert trial["gates"]["residual_strictly_below_0_4"] is False
assert trial["gates"]["deployment_residual_at_most_0_385"] is False
assert trial["gates"]["rss_strictly_below_8_gib"] is False
assert trial["gates"]["conjunctive_call_pass"] is False

candidate = np.load(HERE / "recovery_candidate_rank35_first.npy", allow_pickle=False)
reference = np.load(HERE / "reference_rank35_first.npy", allow_pickle=False)
assert candidate.dtype == np.float32 and candidate.shape == (16, 1024)
assert reference.dtype == np.float32 and reference.shape == (16, 1024)
assert bool(np.isfinite(candidate).all())
assert array_sha256(candidate) == trial["output_array_sha256"] == "2acfbe6f8acbe6ba525284f681ed90158751c018e1d838415a9ef8e00d5781af"
delta = candidate.astype(np.float64) - reference.astype(np.float64)
assert float(np.sqrt(np.mean(delta[-1] * delta[-1]))) == 4.109649235540117e-8
assert float(np.max(np.abs(delta[-1]))) == 2.384185791015625e-7

assert not (HERE / "recovery_candidate_rank279.npy").exists()
assert not (HERE / "recovery_candidate_rank35_repeat.npy").exists()
assert 'RuntimeError: terminal candidate failure; no retry' in transcript

audit = (HERE / "INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md").read_text()
for required in (
    "CLOSED_THIS_IMPLEMENTATION_FORM. FAIL_TERMINAL_NO_RETRY.",
    "Only rank-35 ordinal 0 executed.",
    "The ordinal-0 repeat flag is\nvacuously true",
    "No further D5R1 execution is authorized.",
):
    assert required in audit, required

print(json.dumps({
    "status": "PASS_INDEPENDENT_RECOVERY_DESCENDANT_TERMINAL",
    "node": "P2R0C1D5R1",
    "candidate_calls": 1,
    "uncalled_ordinals": [1, 2],
    "exact_saving_call0": saving,
    "terminal_failed_gates": [
        "residual_strictly_below_0_4",
        "deployment_residual_at_most_0_385",
        "rss_strictly_below_8_gib",
    ],
    "disposition": "CLOSED_THIS_IMPLEMENTATION_FORM_NO_RETRY",
}, indent=2, sort_keys=True))
