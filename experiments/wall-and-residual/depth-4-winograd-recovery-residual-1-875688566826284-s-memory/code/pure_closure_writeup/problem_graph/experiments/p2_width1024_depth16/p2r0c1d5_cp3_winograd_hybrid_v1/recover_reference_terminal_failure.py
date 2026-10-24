#!/usr/bin/env python3
"""Recover the exact failed reference transcript without calling an estimator.

This is a post-mortem extractor only.  It copies the immutable PTY output from
the two original tool-result events, parses the three already-printed trial
JSON lines, and writes a terminal failure receipt.  It never imports or calls
an estimator and refuses to overwrite either output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SESSION_LOG = Path(
    "/Users/matt/.codex/sessions/2026/08/26/"
    "rollout-2026-08-26T11-42-04-01a03f61-1edf-7560-a686-e27f3f701a30.jsonl"
)
TRANSCRIPT = HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt"
RECEIPT = HERE / "REFERENCE_TERMINAL_FAILURE_RECEIPT.json"
EVENT_ORDINALS = (1638, 1639, 1645, 1646, 1658, 1659)
FIRST_OUTPUT_ORDINAL = 1646
FINAL_OUTPUT_ORDINAL = 1659
EXPECTED_COMMAND = (
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "data_gate/.venv/bin/python "
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "p2r0c1d5_cp3_winograd_hybrid_v1/run_reference_capture.py"
)
EXPECTED_CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
EXPECTED_OUTPUT_FILES = (
    "reference_rank35_first.npy",
    "reference_rank279.npy",
    "reference_rank35_repeat.npy",
)
EXPECTED_FILE_SHA256 = (
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
)
EXPECTED_ARRAY_SHA256 = (
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
    "6f94214ffbcbac95656dd72687d771ff349108e0e6f7625d8eff8c3243e60c50",
    "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178",
)
EXPECTED_BINDINGS = {
    "COMMON_DRIVER_SEAL.json": "776859ea1d510d7e7178533c888100fad23ac82d7412a03e1898bf77b4f703f0",
    "COMMON_DRIVER_MANIFEST.json": "b1c26b795b249bf02b6cc0429ec09f49a617e8e62ba0b6d02d0c2dcac0b83670",
    "run_reference_capture.py": "f5732ac16c9246b25335382d85bcd86c4b653b3b21a3bc6bd14389d61b3b1f3c",
    "run_common_driver_gate.py": "e0acba52fa730bc6d448ce873e953f0ddf50c6a908f5c8e69c5b02ff14a5ec0f",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    "../public_noeigh_card_v1/estimator.py": "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)


def output_text(record: dict[str, object]) -> list[str]:
    payload = record["payload"]
    assert isinstance(payload, dict)
    output = payload["output"]
    assert isinstance(output, list)
    texts: list[str] = []
    for item in output:
        assert isinstance(item, dict) and item.get("type") == "input_text"
        texts.append(str(item["text"]))
    return texts


def main() -> None:
    if TRANSCRIPT.exists() or RECEIPT.exists():
        raise FileExistsError("post-mortem outputs already exist")

    raw_records: dict[int, bytes] = {}
    records: dict[int, dict[str, object]] = {}
    with SESSION_LOG.open("rb") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            ordinal = record.get("ordinal")
            if ordinal in EVENT_ORDINALS:
                raw_records[int(ordinal)] = raw_line
                records[int(ordinal)] = record
    assert tuple(sorted(records)) == EVENT_ORDINALS

    launch = records[1638]["payload"]
    assert isinstance(launch, dict)
    launch_input = str(launch["input"])
    assert EXPECTED_COMMAND in launch_input

    first_texts = output_text(records[FIRST_OUTPUT_ORDINAL])
    final_texts = output_text(records[FINAL_OUTPUT_ORDINAL])
    assert len(first_texts) == 2 and len(final_texts) == 2
    first_wrapper = json.loads(first_texts[1])
    final_wrapper = json.loads(final_texts[1])
    assert first_wrapper["session_id"] == 67729
    assert final_wrapper["exit_code"] == 1
    assert "session_id" not in final_wrapper
    transcript_text = str(first_wrapper["output"]) + str(final_wrapper["output"])
    transcript_bytes = transcript_text.encode("utf-8")

    trial_lines = [
        line
        for line in transcript_text.splitlines()
        if line.startswith('{"exact_outer_wall_seconds"')
    ]
    assert len(trial_lines) == 3
    trials = [json.loads(line) for line in trial_lines]
    summaries: list[dict[str, object]] = []
    for ordinal, (trial, rank, flops, name, file_hash, array_hash) in enumerate(
        zip(
            trials,
            EXPECTED_CALL_ORDER,
            EXPECTED_FLOPS,
            EXPECTED_OUTPUT_FILES,
            EXPECTED_FILE_SHA256,
            EXPECTED_ARRAY_SHA256,
            strict=True,
        )
    ):
        meter = trial["literal_meter"]
        gates = trial["gates"]
        assert trial["ordinal"] == ordinal
        assert trial["role_rank"] == rank
        assert meter["flops_used"] == flops
        assert trial["literal_operation_sum"] == flops
        assert trial["output_file"] == name
        assert trial["output_file_sha256"] == file_hash
        assert trial["output_array_sha256"] == array_hash
        assert gates["conjunctive_pass"] is True and all(gates.values())
        path = HERE / name
        assert path.stat().st_size == 65_664
        assert sha256(path) == file_hash
        summaries.append(
            {
                "ordinal": ordinal,
                "role_rank": rank,
                "literal_flops": flops,
                "literal_operation_sum": trial["literal_operation_sum"],
                "outer_wall_seconds": trial["exact_outer_wall_seconds"],
                "meter_wall_seconds": meter["wall_time_s"],
                "residual_wall_seconds": meter["residual_wall_time_s"],
                "maximum_process_rss_bytes": trial["maximum_process_rss_bytes"],
                "output_array_sha256": array_hash,
                "output_file": name,
                "output_file_sha256": file_hash,
                "output_file_bytes": path.stat().st_size,
                "printed_trial_json_sha256": sha256_bytes(trial_lines[ordinal].encode("utf-8")),
                "literal_operations_sha256": sha256_bytes(
                    json.dumps(meter["operations"], sort_keys=True, separators=(",", ":")).encode("utf-8")
                ),
                "gates": gates,
            }
        )

    traceback = transcript_text[transcript_text.index("Traceback (most recent call last):") :]
    assert 'line 138, in finish' in traceback
    assert '"board_opened": false' in traceback
    assert "NameError: name 'false' is not defined" in traceback
    assert not (HERE / "REFERENCE_RESULT.json").exists()
    assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
    assert not any(HERE.glob("candidate_rank*.npy"))

    observed_bindings = {
        relative: sha256((HERE / relative).resolve())
        for relative in EXPECTED_BINDINGS
    }
    assert observed_bindings == EXPECTED_BINDINGS
    write_once(TRANSCRIPT, transcript_bytes)

    receipt = {
        "schema": "p2r0c1d5-reference-terminal-failure-v1",
        "status": "FAIL_TERMINAL_NO_RETRY",
        "node": "P2R0C1D5",
        "original_protocol_closed": True,
        "candidate_launched": False,
        "candidate_gate_status": "NOT_RUN",
        "reference_science_calls_completed": 3,
        "reference_science_gates": "ALL_THREE_REPORTED_CONJUNCTIVE_PASS",
        "reference_receipt_status": "NOT_WRITTEN_DUE_HARNESS_NAMEERROR",
        "call_order": list(EXPECTED_CALL_ORDER),
        "command": EXPECTED_COMMAND,
        "pty_stream_note": "The launch used tty=true, so stdout and stderr are preserved as one exact combined PTY transcript.",
        "process_exit_code": final_wrapper["exit_code"],
        "terminal_exception": "NameError: name 'false' is not defined. Did you mean: 'False'?",
        "terminal_location": "run_reference_capture.py:138 in finish(), after the three-call loop",
        "latent_unexecuted_candidate_receipt_defect": "run_common_driver_gate.py:193-194 also uses lowercase false; candidate was never launched",
        "trials": summaries,
        "repeat_rank35": {
            "array_hash_identical": summaries[0]["output_array_sha256"] == summaries[2]["output_array_sha256"],
            "file_hash_identical": summaries[0]["output_file_sha256"] == summaries[2]["output_file_sha256"],
        },
        "bindings": {
            **EXPECTED_BINDINGS,
            "REFERENCE_EXECUTION_TRANSCRIPT.txt": sha256(TRANSCRIPT),
            "session_event_raw_line_sha256": {
                str(ordinal): sha256_bytes(raw_records[ordinal]) for ordinal in EVENT_ORDINALS
            },
            "session_event_ordinals": list(EVENT_ORDINALS),
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
        "disposition": "STOP; no patch, retry, candidate launch, score, or submission under the original seal",
    }
    write_once(RECEIPT, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "transcript_sha256": sha256(TRANSCRIPT),
                "receipt_sha256": sha256(RECEIPT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
