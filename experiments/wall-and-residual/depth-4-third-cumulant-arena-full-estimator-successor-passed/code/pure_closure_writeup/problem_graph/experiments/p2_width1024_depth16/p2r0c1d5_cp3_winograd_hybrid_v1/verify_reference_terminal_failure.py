#!/usr/bin/env python3
"""Verify the sealed P2R0C1D5 reference terminal-failure evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TRANSCRIPT = HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt"
RECEIPT = HERE / "REFERENCE_TERMINAL_FAILURE_RECEIPT.json"
EXPECTED_TRANSCRIPT_SHA256 = "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28"
EXPECTED_RECEIPT_SHA256 = "4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157"
EXPECTED_CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
EXPECTED_FILES = (
    ("reference_rank35_first.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812"),
    ("reference_rank279.npy", "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba"),
    ("reference_rank35_repeat.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812"),
)
EXPECTED_ARRAY_SHA256 = (
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
    "6f94214ffbcbac95656dd72687d771ff349108e0e6f7625d8eff8c3243e60c50",
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
)


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
    assert sha256(TRANSCRIPT) == EXPECTED_TRANSCRIPT_SHA256
    assert sha256(RECEIPT) == EXPECTED_RECEIPT_SHA256
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["schema"] == "p2r0c1d5-reference-terminal-failure-v1"
    assert receipt["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert receipt["node"] == "P2R0C1D5"
    assert receipt["original_protocol_closed"] is True
    assert receipt["candidate_launched"] is False
    assert receipt["candidate_gate_status"] == "NOT_RUN"
    assert receipt["call_order"] == list(EXPECTED_CALL_ORDER)
    assert receipt["process_exit_code"] == 1
    assert receipt["bindings"][TRANSCRIPT.name] == EXPECTED_TRANSCRIPT_SHA256

    transcript = TRANSCRIPT.read_text()
    trial_lines = [
        line
        for line in transcript.splitlines()
        if line.startswith('{"exact_outer_wall_seconds"')
    ]
    assert len(trial_lines) == 3
    printed = [json.loads(line) for line in trial_lines]
    assert [row["ordinal"] for row in printed] == [0, 1, 2]
    assert [row["role_rank"] for row in printed] == list(EXPECTED_CALL_ORDER)
    assert [row["literal_meter"]["flops_used"] for row in printed] == list(EXPECTED_FLOPS)
    assert [row["literal_operation_sum"] for row in printed] == list(EXPECTED_FLOPS)
    assert all(row["gates"]["conjunctive_pass"] for row in printed)
    assert all(all(row["gates"].values()) for row in printed)

    for ordinal, ((name, file_hash), expected_array_hash) in enumerate(
        zip(EXPECTED_FILES, EXPECTED_ARRAY_SHA256, strict=True)
    ):
        path = HERE / name
        assert path.stat().st_size == 65_664
        assert sha256(path) == file_hash
        value = np.load(path, allow_pickle=False)
        assert value.dtype == np.float32
        assert value.shape == (16, 1024)
        assert np.isfinite(value).all()
        assert array_sha256(value) == expected_array_hash
        row = printed[ordinal]
        summary = receipt["trials"][ordinal]
        assert row["output_file"] == name
        assert row["output_file_sha256"] == file_hash
        assert row["output_array_sha256"] == expected_array_hash
        assert summary["output_file_sha256"] == file_hash
        assert summary["output_array_sha256"] == expected_array_hash
        assert summary["literal_flops"] == EXPECTED_FLOPS[ordinal]
        assert summary["literal_operation_sum"] == EXPECTED_FLOPS[ordinal]
        assert summary["outer_wall_seconds"] == row["exact_outer_wall_seconds"]
        assert summary["meter_wall_seconds"] == row["literal_meter"]["wall_time_s"]
        assert summary["residual_wall_seconds"] == row["literal_meter"]["residual_wall_time_s"]
        assert summary["maximum_process_rss_bytes"] == row["maximum_process_rss_bytes"]
        assert summary["printed_trial_json_sha256"] == sha256_bytes(trial_lines[ordinal].encode())
        operations = json.dumps(
            row["literal_meter"]["operations"], sort_keys=True, separators=(",", ":")
        ).encode()
        assert summary["literal_operations_sha256"] == sha256_bytes(operations)

    assert np.array_equal(
        np.load(HERE / EXPECTED_FILES[0][0], allow_pickle=False),
        np.load(HERE / EXPECTED_FILES[2][0], allow_pickle=False),
    )
    assert receipt["repeat_rank35"] == {
        "array_hash_identical": True,
        "file_hash_identical": True,
    }

    assert "Traceback (most recent call last):" in transcript
    assert 'line 138, in finish' in transcript
    assert '"board_opened": false' in transcript
    assert "NameError: name 'false' is not defined. Did you mean: 'False'?" in transcript
    reference_source = (HERE / "run_reference_capture.py").read_text()
    candidate_runner = (HERE / "run_common_driver_gate.py").read_text()
    assert '"board_opened": false,' in reference_source
    assert '"board_opened": false,' in candidate_runner
    assert '"submission": false,' in candidate_runner

    for relative, expected in receipt["bindings"].items():
        if relative in {
            TRANSCRIPT.name,
            "session_event_raw_line_sha256",
            "session_event_ordinals",
            "session_log_locator",
        }:
            continue
        assert sha256((HERE / relative).resolve()) == expected

    session_log = Path(receipt["bindings"]["session_log_locator"])
    expected_events = receipt["bindings"]["session_event_raw_line_sha256"]
    observed_events: dict[str, str] = {}
    with session_log.open("rb") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            key = str(record.get("ordinal"))
            if key in expected_events:
                observed_events[key] = sha256_bytes(raw_line)
    assert observed_events == expected_events

    assert not (HERE / "REFERENCE_RESULT.json").exists()
    assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
    assert not any(HERE.glob("candidate_rank*.npy"))
    assert receipt["access_declarations"] == {
        "board_opened": False,
        "names_opened": [],
        "network": False,
        "scores_opened": [],
        "submission": False,
        "target_columns_opened": [],
        "training": False,
    }
    print("PASS_REFERENCE_TERMINAL_FAILURE")


if __name__ == "__main__":
    main()
