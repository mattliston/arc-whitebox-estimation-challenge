#!/usr/bin/env python3
"""Independent static verifier for zero-call recovered reference evidence."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


bound = {
    "build_recovered_reference_evidence.py": "969463c12d569d9d1959f950949955e3871abe26b48b9eecb3b84448c570d2dd",
    "RECOVERED_REFERENCE_EVIDENCE.json": "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    "verify_recovered_reference_evidence.py": "807c1b3a91871c129553c2f4975736007ac000eda6aefb90ad068d089bdc6518",
    "REFERENCE_EXECUTION_TRANSCRIPT.txt": "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28",
    "REFERENCE_TERMINAL_FAILURE_RECEIPT.json": "4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157",
    "INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md": "c884bee303b2b9a74c1ddaed8a6528e38a25860136b5608fb021b96283d31c9e",
    "verify_independent_reference_terminal.py": "fd7e2719f5ff0a7b16bf3e848b7a4fab76ea41525bddc8f98357e3faabe6d436",
    "reference_rank35_first.npy": "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    "reference_rank279.npy": "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    "reference_rank35_repeat.npy": "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
}
for relative, expected in bound.items():
    assert sha256(HERE / relative) == expected, relative

source = (HERE / "build_recovered_reference_evidence.py").read_text()
tree = ast.parse(source)
imports = {
    alias.name.split(".")[0]
    for node in ast.walk(tree)
    if isinstance(node, (ast.Import, ast.ImportFrom))
    for alias in node.names
}
assert not ({"flopscope", "whestbench", "importlib"} & imports)
loaded_names = {
    node.id for node in ast.walk(tree)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
}
attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
assert not ({"Estimator", "predict"} & (loaded_names | attributes))
assert not ({"false", "true", "null"} & loaded_names)

evidence = json.loads((HERE / "RECOVERED_REFERENCE_EVIDENCE.json").read_text())
assert evidence["schema"] == "p2r0c1d5-recovered-reference-scientific-evidence-v1"
assert evidence["status"] == "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE"
assert evidence["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
assert evidence["original_process_exit_code"] == 1
assert evidence["original_reference_result_written"] is False
assert evidence["original_candidate_calls"] == 0
assert evidence["recovery_reference_estimator_calls"] == 0
assert evidence["candidate_calls_at_recovery"] == 0
assert evidence["call_order"] == [35, 279, 35]
assert "never relabels the failed original process" in evidence["evidence_semantics"]

transcript_trials = [
    json.loads(line)
    for line in (HERE / "REFERENCE_EXECUTION_TRANSCRIPT.txt").read_text().splitlines()
    if line.startswith('{"exact_outer_wall_seconds"')
]
assert evidence["trials"] == transcript_trials
assert [trial["literal_meter"]["flops_used"] for trial in evidence["trials"]] == [
    1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552,
]
assert all(all(trial["gates"].values()) for trial in evidence["trials"])
assert evidence["access_declarations"] == {
    "board_opened": False,
    "names_opened": [],
    "network": False,
    "scores_opened": [],
    "submission": False,
    "target_columns_opened": [],
    "training": False,
}

assert not (HERE / "REFERENCE_RESULT.json").exists()
assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
assert not (HERE / "RECOVERY_DESCENDANT_RESULT.json").exists()
assert not list(HERE.glob("candidate_rank*.npy"))
assert not list(HERE.glob("recovery_candidate_rank*.npy"))

audit = (HERE / "INDEPENDENT_REDTEAM_RECOVERED_REFERENCE_AUDIT.md").read_text()
assert "ACCEPTED AS ZERO-CALL SCIENTIFIC EVIDENCE" in audit
assert "NOT A SUCCESS RECEIPT FOR\nTHE ORIGINAL PROCESS" in audit
assert "Candidate launch remains contingent" in audit

print(json.dumps({
    "status": "PASS_INDEPENDENT_RECOVERED_REFERENCE_EVIDENCE",
    "reference_estimator_calls_added": 0,
    "candidate_calls": 0,
    "launch_authorized": False,
}, indent=2, sort_keys=True))
