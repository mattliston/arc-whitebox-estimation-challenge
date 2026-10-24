#!/usr/bin/env python3
"""Seal the exact one-shot P2R0C1D5R1 terminal execution evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SESSION_LOG = Path(
    "/Users/matt/.codex/sessions/2026/08/26/"
    "rollout-2026-08-26T11-42-04-01a03f61-1edf-7560-a686-e27f3f701a30.jsonl"
)
TRANSCRIPT = HERE / "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt"
RECEIPT = HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json"
RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"
OUTPUT = HERE / "recovery_candidate_rank35_first.npy"
EVENT_ORDINALS = (2250, 2251, 2255, 2256, 2260, 2262)
FINAL_OUTPUT_ORDINAL = 2262
EXPECTED_COMMAND = (
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "data_gate/.venv/bin/python "
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "p2r0c1d5_cp3_winograd_hybrid_v1/run_recovery_descendant_candidate_gate.py"
)
BINDINGS = {
    "RECOVERY_DESCENDANT_LAUNCH_SEAL.json": "4da8ddd17b954a9f8f2c0b78ff5f5d90f1ef958444cda2f2b0b7c27a6b1efd33",
    "verify_recovery_descendant_launch_seal.py": "5b9327d76e608b22aaf8b2af61dfee33264df5ec3f85ae2ed6ab6a3d58ac3f8d",
    "run_recovery_descendant_candidate_gate.py": "42ca250d6686e5e48b2730063d22dfd3407f870a16dab59be0929f6a5d81928f",
    "RECOVERY_DESCENDANT_MANIFEST.json": "cafa3d88a46d4fc45dca6cfe52acaf148bb9123e5e88b31d58fb37d0135e24c1",
    "RECOVERED_REFERENCE_EVIDENCE.json": "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    "RECOVERY_DESCENDANT_RESULT.json": "a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549",
    "recovery_candidate_rank35_first.npy": "4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290",
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


def output_text(record: dict[str, object]) -> list[str]:
    output = record["payload"]["output"]
    assert isinstance(output, list)
    return [str(item["text"]) for item in output]


def main() -> None:
    if TRANSCRIPT.exists() or RECEIPT.exists():
        raise FileExistsError("terminal evidence already sealed")
    records: dict[int, dict[str, object]] = {}
    raw_records: dict[int, bytes] = {}
    with SESSION_LOG.open("rb") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            ordinal = record.get("ordinal")
            if ordinal in EVENT_ORDINALS:
                records[int(ordinal)] = record
                raw_records[int(ordinal)] = raw_line
    assert tuple(sorted(records)) == EVENT_ORDINALS
    assert EXPECTED_COMMAND in str(records[2250]["payload"]["input"])
    texts = output_text(records[FINAL_OUTPUT_ORDINAL])
    assert len(texts) == 2
    wrapper = json.loads(texts[1])
    assert wrapper["exit_code"] == 1
    transcript_text = str(wrapper["output"])
    trial_lines = [
        line for line in transcript_text.splitlines()
        if line.startswith('{"drift_vs_exact_incumbent"')
    ]
    status_lines = [
        line for line in transcript_text.splitlines()
        if line.startswith('{"aggregate"')
    ]
    assert len(trial_lines) == 1 and len(status_lines) == 1
    trial = json.loads(trial_lines[0])
    printed_status = json.loads(status_lines[0])
    result = json.loads(RESULT.read_text())
    assert result["status"] == "FAIL_TERMINAL"
    assert result["terminal_reason"] == "candidate call 0 gate failure"
    assert result["trials"] == [trial]
    assert printed_status == {
        "aggregate": {},
        "reason": "candidate call 0 gate failure",
        "status": "FAIL_TERMINAL",
    }
    assert trial["ordinal"] == 0 and trial["role_rank"] == 35
    assert trial["literal_meter"]["flops_used"] == 1_320_876_655_991
    assert trial["literal_operation_sum"] == 1_320_876_655_991
    assert trial["measured_reference_flops"] == 1_692_484_904_311
    assert trial["measured_reference_minus_candidate_flops"] == 371_608_248_320
    assert trial["gates"]["literal_flops_exact"] is True
    assert trial["gates"]["measured_exact_delta_vs_measured_incumbent"] is True
    assert trial["gates"]["final_layer_maximum_absolute_delta"] is True
    assert trial["gates"]["residual_strictly_below_0_4"] is False
    assert trial["gates"]["deployment_residual_at_most_0_385"] is False
    assert trial["gates"]["rss_strictly_below_8_gib"] is False
    assert trial["gates"]["conjunctive_call_pass"] is False
    assert 'RuntimeError: terminal candidate failure; no retry' in transcript_text
    for relative, expected in BINDINGS.items():
        assert sha256((HERE / relative).resolve()) == expected, relative
    value = np.load(OUTPUT, allow_pickle=False)
    assert value.dtype == np.float32 and value.shape == (16, 1024)
    assert np.isfinite(value).all()
    assert array_sha256(value) == EXPECTED_ARRAY_SHA256 == trial["output_array_sha256"]
    assert not (HERE / "recovery_candidate_rank279.npy").exists()
    assert not (HERE / "recovery_candidate_rank35_repeat.npy").exists()

    with TRANSCRIPT.open("xb") as handle:
        handle.write(transcript_text.encode("utf-8"))
    receipt = {
        "schema": "p2r0c1d5r1-terminal-execution-receipt-v1",
        "status": "FAIL_TERMINAL_NO_RETRY",
        "node": "P2R0C1D5R1",
        "parent_protocol_status": "FAIL_TERMINAL_NO_RETRY",
        "process_exit_code": 1,
        "candidate_calls_completed": 1,
        "failed_ordinal": 0,
        "uncalled_ordinals": [1, 2],
        "uncalled_role_ranks": [279, 35],
        "reference_estimator_calls_in_descendant": 0,
        "terminal_reason": "candidate call 0 failed frozen residual and RSS gates",
        "trial": trial,
        "gate_summary": {
            "literal_flops_exact": True,
            "measured_exact_delta_371608248320": True,
            "final_layer_rms_delta": trial["drift_vs_exact_incumbent"]["final_layer_rms"],
            "final_layer_maximum_absolute_delta": trial["drift_vs_exact_incumbent"]["final_layer_maximum_absolute"],
            "outer_wall_seconds": trial["exact_outer_wall_seconds"],
            "meter_wall_seconds": trial["literal_meter"]["wall_time_s"],
            "residual_wall_seconds": trial["literal_meter"]["residual_wall_time_s"],
            "maximum_process_rss_bytes": trial["maximum_process_rss_bytes"],
            "failed_gates": [
                "residual_strictly_below_0_4",
                "deployment_residual_at_most_0_385",
                "rss_strictly_below_8_gib",
            ],
        },
        "bindings": {
            **BINDINGS,
            "RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt": sha256(TRANSCRIPT),
            "printed_trial_json_sha256": sha256_bytes(trial_lines[0].encode("utf-8")),
            "printed_status_json_sha256": sha256_bytes(status_lines[0].encode("utf-8")),
            "session_event_ordinals": list(EVENT_ORDINALS),
            "session_event_raw_line_sha256": {
                str(ordinal): sha256_bytes(raw_records[ordinal]) for ordinal in EVENT_ORDINALS
            },
            "session_log_locator": str(SESSION_LOG),
        },
        "access_declarations": {
            "target_columns_opened": [],
            "scores_opened": [],
            "names_opened": [],
            "board_opened": False,
            "submission": False,
            "training": False,
            "network": False,
        },
        "disposition": "CLOSED_THIS_IMPLEMENTATION_FORM; no retry, remaining calls, score, or submission",
    }
    with RECEIPT.open("xb") as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": receipt["status"], "transcript_sha256": sha256(TRANSCRIPT), "receipt_sha256": sha256(RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    main()
