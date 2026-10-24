#!/usr/bin/env python3
"""Verify the zero-call recovered reference scientific evidence."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_recovered_reference_evidence.py"
EVIDENCE = HERE / "RECOVERED_REFERENCE_EVIDENCE.json"
TRANSCRIPT = HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt"
EXPECTED_EVIDENCE_SHA256 = "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6"
EXPECTED_TRANSCRIPT_SHA256 = "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28"
EXPECTED_CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
EXPECTED_OUTPUTS = (
    ("reference_rank35_first.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812", "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178"),
    ("reference_rank279.npy", "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba", "6f94214ffbcbac95656dd72687d771ff349108e0e6f7625d8eff8c3243e60c50"),
    ("reference_rank35_repeat.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812", "54f1829cd5fc4a5276565108096dc7c508a4f143cb80dcd8cd804f673416d178"),
)


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
    assert sha256(EVIDENCE) == EXPECTED_EVIDENCE_SHA256
    assert sha256(TRANSCRIPT) == EXPECTED_TRANSCRIPT_SHA256
    source = BUILDER.read_text()
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"flopscope", "whestbench", "importlib"} & imported)
    loaded_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    loaded_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "Estimator" not in loaded_names and "Estimator" not in loaded_attributes
    assert "predict" not in loaded_names and "predict" not in loaded_attributes

    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["status"] == "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE"
    assert evidence["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert evidence["original_process_exit_code"] == 1
    assert evidence["original_reference_result_written"] is False
    assert evidence["original_candidate_calls"] == 0
    assert evidence["recovery_reference_estimator_calls"] == 0
    assert evidence["candidate_calls_at_recovery"] == 0
    assert evidence["call_order"] == list(EXPECTED_CALL_ORDER)

    transcript_trials = [
        json.loads(line) for line in TRANSCRIPT.read_text().splitlines()
        if line.startswith('{"exact_outer_wall_seconds"')
    ]
    assert evidence["trials"] == transcript_trials
    assert len(transcript_trials) == 3
    for ordinal, (trial, rank, flops, output_info) in enumerate(
        zip(transcript_trials, EXPECTED_CALL_ORDER, EXPECTED_FLOPS, EXPECTED_OUTPUTS, strict=True)
    ):
        name, file_hash, value_hash = output_info
        assert trial["ordinal"] == ordinal and trial["role_rank"] == rank
        assert trial["literal_meter"]["flops_used"] == flops
        assert trial["literal_operation_sum"] == flops
        assert sum(int(item["flop_cost"]) for item in trial["literal_meter"]["operations"].values()) == flops
        assert all(trial["gates"].values())
        path = HERE / name
        assert sha256(path) == file_hash == trial["output_file_sha256"]
        value = np.load(path, allow_pickle=False)
        assert value.dtype == np.float32 and value.shape == (16, 1024)
        assert np.isfinite(value).all()
        assert array_sha256(value) == value_hash == trial["output_array_sha256"]

    assert evidence["trials"][0]["output_array_sha256"] == evidence["trials"][2]["output_array_sha256"]
    assert 'line 138, in finish' in evidence["traceback"]
    assert "NameError: name 'false' is not defined" in evidence["traceback"]
    for relative, expected in evidence["bindings"].items():
        assert sha256((HERE / relative).resolve()) == expected, relative
    assert not (HERE / "REFERENCE_RESULT.json").exists()
    assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
    assert not (HERE / "RECOVERY_DESCENDANT_RESULT.json").exists()
    assert not any(HERE.glob("candidate_rank*.npy"))
    assert not any(HERE.glob("recovery_candidate_rank*.npy"))
    print("PASS_RECOVERED_REFERENCE_EVIDENCE")


if __name__ == "__main__":
    main()
