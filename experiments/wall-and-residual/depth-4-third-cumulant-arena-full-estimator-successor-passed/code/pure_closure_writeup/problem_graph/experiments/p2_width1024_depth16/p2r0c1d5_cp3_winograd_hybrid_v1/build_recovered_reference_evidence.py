#!/usr/bin/env python3
"""Build recovered reference science evidence without any estimator call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TRANSCRIPT = HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt"
TERMINAL_RECEIPT = HERE / "REFERENCE_TERMINAL_FAILURE_RECEIPT.json"
OUTPUT = HERE / "RECOVERED_REFERENCE_EVIDENCE.json"
TRANSCRIPT_SHA256 = "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28"
TERMINAL_RECEIPT_SHA256 = "4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157"
EXPECTED_CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
OUTPUT_NAMES = (
    "reference_rank35_first.npy",
    "reference_rank279.npy",
    "reference_rank35_repeat.npy",
)
OUTPUT_FILE_SHA256 = (
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
)
OUTPUT_ARRAY_SHA256 = (
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
    "6f94214ffbcbac95656dd72687d771ff349108e0e6f7625d8eff8c3243e60c50",
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
)
BINDINGS = {
    "COMMON_DRIVER_SEAL.json": "776859ea1d510d7e7178533c888100fad23ac82d7412a03e1898bf77b4f703f0",
    "COMMON_DRIVER_MANIFEST.json": "b1c26b795b249bf02b6cc0429ec09f49a617e8e62ba0b6d02d0c2dcac0b83670",
    "run_reference_capture.py": "f5732ac16c9246b25335382d85bcd86c4b653b3b21a3bc6bd14389d61b3b1f3c",
    "../public_noeigh_card_v1/estimator.py": "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    "REFERENCE_EXECUTION_TRANSCRIPT.txt": TRANSCRIPT_SHA256,
    "REFERENCE_TERMINAL_FAILURE_RECEIPT.json": TERMINAL_RECEIPT_SHA256,
    "verify_reference_terminal_failure.py": "d8b54ddd75af2e82c1217a621bb9943d4b524b367e3c1d4bd1cd9fce1b2cc86c",
    "INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md": "c884bee303b2b9a74c1ddaed8a6528e38a25860136b5608fb021b96283d31c9e",
    "verify_independent_reference_terminal.py": "fd7e2719f5ff0a7b16bf3e848b7a4fab76ea41525bddc8f98357e3faabe6d436",
}


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
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    assert sha256(TRANSCRIPT) == TRANSCRIPT_SHA256
    assert sha256(TERMINAL_RECEIPT) == TERMINAL_RECEIPT_SHA256
    for relative, expected in BINDINGS.items():
        assert sha256((HERE / relative).resolve()) == expected, relative
    terminal = json.loads(TERMINAL_RECEIPT.read_text())
    assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
    assert terminal["process_exit_code"] == 1
    assert terminal["candidate_launched"] is False

    transcript = TRANSCRIPT.read_text()
    raw_trial_lines = [
        line for line in transcript.splitlines()
        if line.startswith('{"exact_outer_wall_seconds"')
    ]
    assert len(raw_trial_lines) == 3
    trials = [json.loads(line) for line in raw_trial_lines]
    for ordinal, (trial, role_rank, flops, name, file_hash, value_hash) in enumerate(
        zip(
            trials,
            EXPECTED_CALL_ORDER,
            EXPECTED_FLOPS,
            OUTPUT_NAMES,
            OUTPUT_FILE_SHA256,
            OUTPUT_ARRAY_SHA256,
            strict=True,
        )
    ):
        assert trial["ordinal"] == ordinal and trial["role_rank"] == role_rank
        assert trial["literal_meter"]["flops_used"] == flops
        assert trial["literal_operation_sum"] == flops
        assert sum(int(item["flop_cost"]) for item in trial["literal_meter"]["operations"].values()) == flops
        assert trial["gates"]["conjunctive_pass"] is True
        assert all(trial["gates"].values())
        path = HERE / name
        assert trial["output_file"] == name
        assert trial["output_file_sha256"] == file_hash == sha256(path)
        value = np.load(path, allow_pickle=False)
        assert value.dtype == np.float32 and value.shape == (16, 1024)
        assert np.isfinite(value).all()
        assert trial["output_array_sha256"] == value_hash == array_sha256(value)
    assert OUTPUT_FILE_SHA256[0] == OUTPUT_FILE_SHA256[2]
    assert OUTPUT_ARRAY_SHA256[0] == OUTPUT_ARRAY_SHA256[2]

    traceback = transcript[transcript.index("Traceback (most recent call last):") :]
    assert 'line 138, in finish' in traceback
    assert "NameError: name 'false' is not defined" in traceback
    receipt = {
        "schema": "p2r0c1d5-recovered-reference-scientific-evidence-v1",
        "status": "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE",
        "node": "P2R0C1D5_RECOVERY_DESCENDANT",
        "original_protocol_status": "FAIL_TERMINAL_NO_RETRY",
        "original_process_exit_code": 1,
        "original_reference_result_written": False,
        "original_candidate_calls": 0,
        "recovery_reference_estimator_calls": 0,
        "candidate_calls_at_recovery": 0,
        "call_order": list(EXPECTED_CALL_ORDER),
        "evidence_semantics": "Recovered science evidence only; this never relabels the failed original process as a successful receipt run.",
        "trials": trials,
        "traceback": traceback,
        "session_event_raw_line_sha256": terminal["bindings"]["session_event_raw_line_sha256"],
        "bindings": BINDINGS,
        "access_declarations": {
            "target_columns_opened": [],
            "scores_opened": [],
            "names_opened": [],
            "board_opened": False,
            "submission": False,
            "training": False,
            "network": False,
        },
    }
    with OUTPUT.open("xb") as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": receipt["status"], "sha256": sha256(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
