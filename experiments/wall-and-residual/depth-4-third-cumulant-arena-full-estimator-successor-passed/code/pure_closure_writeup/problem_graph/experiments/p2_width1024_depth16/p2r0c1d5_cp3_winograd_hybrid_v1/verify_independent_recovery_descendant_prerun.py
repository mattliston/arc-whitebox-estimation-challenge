#!/usr/bin/env python3
"""Independent pre-run verifier for the P2R0C1D5R1 recovery descendant."""

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
    "RECOVERY_DESCENDANT_PRESEAL.json": "a2d4a2263543d6d86c7e2db0c3f17baef6723ee93f49171a83ae83c94893f1e8",
    "RECOVERY_DESCENDANT_MANIFEST.json": "cafa3d88a46d4fc45dca6cfe52acaf148bb9123e5e88b31d58fb37d0135e24c1",
    "run_recovery_descendant_candidate_gate.py": "42ca250d6686e5e48b2730063d22dfd3407f870a16dab59be0929f6a5d81928f",
    "build_recovery_descendant_runner.py": "5e51e00e52a4e9d45c93ffba5369cb3dcf86360ee2374f489b0ce134917b51df",
    "test_recovery_descendant_receipt_serialization.py": "61c44b5505cf511715e292b9e7b2d384127ee58e88c0e224796f91b94c2cbb91",
    "verify_recovery_descendant_static.py": "2c6958544d593cc348fa85272916a5f7496e1e51e94db7cbfdb387a4a8d83b32",
    "verify_recovery_descendant_preseal.py": "54cbacb3a00cbd26cf8d79a9eb560e3170f62e382573ca66cd26c5dd65caf4ad",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    "RECOVERED_REFERENCE_EVIDENCE.json": "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    "COMMON_DRIVER_SEAL.json": "776859ea1d510d7e7178533c888100fad23ac82d7412a03e1898bf77b4f703f0",
    "REFERENCE_EXECUTION_TRANSCRIPT.txt": "5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28",
    "REFERENCE_TERMINAL_FAILURE_RECEIPT.json": "4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157",
    "INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md": "c884bee303b2b9a74c1ddaed8a6528e38a25860136b5608fb021b96283d31c9e",
    "verify_independent_reference_terminal.py": "fd7e2719f5ff0a7b16bf3e848b7a4fab76ea41525bddc8f98357e3faabe6d436",
    "INDEPENDENT_REDTEAM_RECOVERED_REFERENCE_AUDIT.md": "c4a829804531190f964129c7e768a41856f37c29fd9ea618bc1f7b80d87c3066",
    "verify_independent_recovered_reference.py": "9533f5b7edd2d25e8c5d1977fd26a6b2b9fa93c1fc2290138ededa52ef612b4d",
}
for relative, expected in bound.items():
    observed = sha256(HERE / relative)
    assert observed == expected, (relative, observed, expected)

preseal = json.loads((HERE / "RECOVERY_DESCENDANT_PRESEAL.json").read_text())
manifest = json.loads((HERE / "RECOVERY_DESCENDANT_MANIFEST.json").read_text())
evidence = json.loads((HERE / "RECOVERED_REFERENCE_EVIDENCE.json").read_text())
assert preseal["status"] == "FROZEN_FOR_INDEPENDENT_RECOVERY_DESCENDANT_REVIEW"
assert preseal["launch_authorization"] == "HELD"
assert preseal["node"] == manifest["candidate_node"] == "P2R0C1D5R1"
assert preseal["parent_protocol_status"] == manifest["parent_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
assert preseal["reference_estimator_calls_in_descendant"] == 0
assert preseal["original_candidate_calls"] == 0
assert preseal["descendant_candidate_calls"] == 0
assert manifest["process_protocol"] == {
    "reference": "zero estimator calls; consume only the sealed recovered transcript evidence and three immutable reference arrays",
    "candidate": "exactly one fresh process with one persistent candidate Estimator",
    "candidate_calls": "exactly [35,279,35] with no retry or retune",
    "original_candidate_calls": 0,
    "descendant_candidate_calls_at_freeze": 0,
    "reference_estimator_calls_in_descendant": 0,
}
assert manifest["call_order"] == [35, 279, 35]
assert manifest["candidate_runner_sha256"] == bound["run_recovery_descendant_candidate_gate.py"]
assert manifest["candidate_source_sha256"] == bound["estimator.py"]
assert manifest["recovered_reference_evidence_sha256"] == bound["RECOVERED_REFERENCE_EVIDENCE.json"]

incumbent = [1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552]
candidate = [1_320_876_655_991, 1_359_519_832_584, 1_320_873_503_232]
assert manifest["expected_incumbent_flops"] == incumbent
assert manifest["expected_candidate_flops"] == candidate
assert manifest["amended_exact_net_saving"] == 371_608_248_320
assert [a - b for a, b in zip(incumbent, candidate)] == [371_608_248_320] * 3
assert manifest["normal_path_noeigh_removal_counts"] == [14, 12, 14]

assert evidence["status"] == "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE"
assert evidence["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
assert evidence["original_process_exit_code"] == 1
assert evidence["original_reference_result_written"] is False
assert evidence["original_candidate_calls"] == 0
assert evidence["recovery_reference_estimator_calls"] == 0
assert evidence["candidate_calls_at_recovery"] == 0
assert evidence["call_order"] == [35, 279, 35]
assert len(evidence["trials"]) == 3
assert all(all(trial["gates"].values()) for trial in evidence["trials"])

runner_path = HERE / "run_recovery_descendant_candidate_gate.py"
builder_path = HERE / "build_recovery_descendant_runner.py"
runner_source = runner_path.read_text()
runner_tree = ast.parse(runner_source)
builder_tree = ast.parse(builder_path.read_text())
for name, tree in ((runner_path.name, runner_tree), (builder_path.name, builder_tree)):
    loaded = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not ({"false", "true", "null"} & loaded), name

assert runner_source.count("estimator = module.Estimator()") == 1
assert runner_source.count("estimator.predict(") == 1
assert runner_source.index("estimator = module.Estimator()") < runner_source.index(
    "for ordinal, role_rank in enumerate(CALL_ORDER)"
)
for required in (
    'SOURCE_SHA256 = "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5"',
    "CALL_ORDER = (35, 279, 35)",
    "EXPECTED_FLOPS = (1_320_876_655_991, 1_359_519_832_584, 1_320_873_503_232)",
    "EXPECTED_REFERENCE_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)",
    "EXACT_SAVING = 371_608_248_320",
    '"node": "P2R0C1D5R1"',
    '"reference_estimator_calls_in_descendant": 0',
    'reference_result.get("call_order", ())',
    'trial["ordinal"] != ordinal or trial["role_rank"] != CALL_ORDER[ordinal]',
    'trial["output_file"] != path.name',
    'measured_delta = measured_reference_flops - measured_candidate_flops',
    '"measured_exact_delta_vs_measured_incumbent": measured_delta == EXACT_SAVING',
    '"deployment_residual_at_most_0_385"',
    '"pooled_all_three_calls_final_rms_gate"',
    '"pooled_final_rms_gate"',
    '"maximum_final_absolute_gate"',
    '"repeat_rank35_candidate_byte_identical"',
):
    assert required in runner_source, required

for forbidden in (
    "targets.npy", "scores.npy", "row_manifest.json", "SUBMIT_WATCH",
    "requests.", "urllib.", "socket.", ".fit(", ".train(", "subprocess.",
):
    assert forbidden not in runner_source, forbidden

assert manifest["scope_guards"] == {
    "weights_only": True,
    "target_columns_opened": [],
    "scores_opened": [],
    "names_opened": [],
    "board_opened": False,
    "submission": False,
    "network": False,
    "training": False,
}
assert manifest["gates"]["exact_delta_each_call"] == 371_608_248_320
assert manifest["gates"]["maximum_process_rss_bytes_strict"] == 8 * 1024**3
assert manifest["gates"]["residual_wall_seconds_strict"] == 0.4
assert manifest["gates"]["deployment_residual_wall_seconds_maximum"] == 0.385

for relative, expected_absent in preseal["absence_gates"].items():
    assert expected_absent is True
    assert not (HERE / relative).exists(), relative

audit = (HERE / "INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_PRERUN_AUDIT.md").read_text()
for required in (
    "STATIC ACCEPTANCE. AUTHORIZE FINAL DESCENDANT SEAL CREATION",
    "AUTHORIZE THE\nONE-SHOT TARGET-FREE CANDIDATE GATE",
    "No remaining static objection exists.",
    "This authorization excludes all reference reruns",
):
    assert required in audit, required

print(json.dumps({
    "status": "PASS_INDEPENDENT_RECOVERY_DESCENDANT_PRERUN",
    "node": "P2R0C1D5R1",
    "reference_calls_added": 0,
    "candidate_calls_so_far": 0,
    "authorization": "FINAL_SEAL_THEN_ONE_SHOT_TARGET_FREE_CANDIDATE",
}, indent=2, sort_keys=True))
