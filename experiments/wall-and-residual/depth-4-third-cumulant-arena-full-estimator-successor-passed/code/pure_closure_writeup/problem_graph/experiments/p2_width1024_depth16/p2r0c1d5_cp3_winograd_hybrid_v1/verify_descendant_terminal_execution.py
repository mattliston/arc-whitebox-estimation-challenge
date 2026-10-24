#!/usr/bin/env python3
"""Verify the terminal one-shot P2R0C1D5R1 candidate execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TRANSCRIPT = HERE / "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt"
RECEIPT = HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json"
RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"
OUTPUT = HERE / "recovery_candidate_rank35_first.npy"
EXPECTED_SHA256 = {
    "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt": "afa66d0a953b4d3f0d703bbbb62321dfd3b10df0aec7b716ec6847e3ddaf68b3",
    "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json": "8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0",
    "RECOVERY_DESCENDANT_RESULT.json": "a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549",
    "recovery_candidate_rank35_first.npy": "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
    "RECOVERY_DESCENDANT_LAUNCH_SEAL.json": "4da8ddd17b954a9f8f2c0b78ff5f5d90f1ef958444cda2f2b0b7c27a6b1efd33",
    "run_recovery_descendant_candidate_gate.py": "42ca250d6686e5e48b2730063d22dfd3407f870a16dab59be0929f6a5d81928f",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
}
EXPECTED_ARRAY_SHA256 = "2acfbe6f8acbe6ba525284f681ed90158751c018e1d838415a9ef8e00d5781af"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def main() -> None:
    for relative, expected in EXPECTED_SHA256.items():
        assert sha256(HERE / relative) == expected, relative
    transcript = TRANSCRIPT.read_text()
    receipt = json.loads(RECEIPT.read_text())
    result = json.loads(RESULT.read_text())
    trial_lines = [
        line for line in transcript.splitlines()
        if line.startswith('{"drift_vs_exact_incumbent"')
    ]
    assert len(trial_lines) == 1
    trial = json.loads(trial_lines[0])
    assert result["status"] == "FAIL_TERMINAL"
    assert result["terminal_reason"] == "candidate call 0 gate failure"
    assert result["trials"] == [trial]
    assert result["reference_estimator_calls_in_descendant"] == 0
    assert result["bindings"]["recovery_descendant_launch_seal_sha256"] == EXPECTED_SHA256["RECOVERY_DESCENDANT_LAUNCH_SEAL.json"]
    assert result["bindings"]["program_sha256"] == EXPECTED_SHA256["run_recovery_descendant_candidate_gate.py"]
    assert result["bindings"]["candidate_source_sha256"] == EXPECTED_SHA256["estimator.py"]
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
    assert receipt["bindings"][TRANSCRIPT.name] == EXPECTED_SHA256[TRANSCRIPT.name]
    assert receipt["bindings"]["printed_trial_json_sha256"] == sha256_bytes(trial_lines[0].encode())
    assert receipt["disposition"].startswith("CLOSED_THIS_IMPLEMENTATION_FORM")

    assert trial["literal_meter"]["flops_used"] == 1_320_876_655_991
    assert trial["literal_operation_sum"] == 1_320_876_655_991
    assert sum(int(item["flop_cost"]) for item in trial["literal_meter"]["operations"].values()) == 1_320_876_655_991
    assert trial["measured_reference_flops"] == 1_692_484_904_311
    assert trial["measured_reference_minus_candidate_flops"] == 371_608_248_320
    assert trial["exact_outer_wall_seconds"] == 24.356781082693487
    assert trial["literal_meter"]["wall_time_s"] == 24.35678375000134
    assert trial["literal_meter"]["residual_wall_time_s"] == 1.875688566826284
    assert trial["maximum_process_rss_bytes"] == 11_953_569_792
    assert trial["drift_vs_exact_incumbent"]["final_layer_rms"] == 4.109649235540117e-8
    assert trial["drift_vs_exact_incumbent"]["final_layer_maximum_absolute"] == 2.384185791015625e-7
    assert trial["gates"]["literal_flops_exact"] is True
    assert trial["gates"]["literal_operation_sum_equals_total"] is True
    assert trial["gates"]["measured_exact_delta_vs_measured_incumbent"] is True
    assert trial["gates"]["final_layer_maximum_absolute_delta"] is True
    assert trial["gates"]["residual_strictly_below_0_4"] is False
    assert trial["gates"]["deployment_residual_at_most_0_385"] is False
    assert trial["gates"]["rss_strictly_below_8_gib"] is False
    assert trial["gates"]["conjunctive_call_pass"] is False
    assert 'RuntimeError: terminal candidate failure; no retry' in transcript

    value = np.load(OUTPUT, allow_pickle=False)
    assert value.dtype == np.float32 and value.shape == (16, 1024)
    assert np.isfinite(value).all()
    assert array_sha256(value) == EXPECTED_ARRAY_SHA256 == trial["output_array_sha256"]
    assert not (HERE / "recovery_candidate_rank279.npy").exists()
    assert not (HERE / "recovery_candidate_rank35_repeat.npy").exists()

    expected_events = receipt["bindings"]["session_event_raw_line_sha256"]
    observed_events: dict[str, str] = {}
    with Path(receipt["bindings"]["session_log_locator"]).open("rb") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            key = str(record.get("ordinal"))
            if key in expected_events:
                observed_events[key] = sha256_bytes(raw_line)
    assert observed_events == expected_events
    print("PASS_RECOVERY_DESCENDANT_TERMINAL_EXECUTION")


if __name__ == "__main__":
    main()
