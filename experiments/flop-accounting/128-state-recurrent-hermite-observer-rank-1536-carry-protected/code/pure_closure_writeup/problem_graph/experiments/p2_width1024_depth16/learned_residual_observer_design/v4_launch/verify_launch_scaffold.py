#!/usr/bin/env python3
"""Strict target-free static/runtime verifier for the V4 launch scaffold."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "v3"
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
MANIFEST_PATH = HERE / "IMPLEMENTATION_MANIFEST_V4.json"
FILES = {
    "bundle_binding": HERE / "BUNDLE_BINDING_V4.json",
    "bundle_gate": HERE / "sealed_bundle_gate.py",
    "feature_contract_json": HERE / "FEATURE_STORE_CONTRACT_V4.json",
    "feature_contract": HERE / "feature_store_contract.py",
    "training_core": HERE / "cpu_training_core.py",
    "trainer": HERE / "trainer_evaluator_scaffold.py",
    "converter": HERE / "checkpoint_converter.py",
    "flopscope_worker": HERE / "flopscope_nonzero_worker.py",
    "torch_worker": HERE / "torch_nonzero_worker.py",
    "parity_verifier": HERE / "verify_nonzero_parity.py",
    "parity_receipt": HERE / "NONZERO_PARITY_RECEIPT_V4.json",
}
FROZEN = {
    V3 / "observer_model.py": "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5",
    V3 / "selection_and_promotion.py": "806a2933afebd369040413a17f3e8f2cee939b0e870da7ab728697ba64eed516",
    V3 / "estimator.py": "f8d48e5fc05d4de53173302e6fd8334c46908dd659a39baaebf2b534f5bdbea6",
    V3 / "estimator_generalized.py": "0b18dba56b6790f21640314f5904c7cd91d165558d90bc5c265898bb632a9310",
    V3 / "estimator_primary_cp1536_h128.py": "800b3833c41a6809e83250e347f4234e54db943f2482ee96290c6b6173d1c019",
    V3 / "estimator_control_cp1792_h64.py": "c1ff172be0acde9ad020718f068fe613b8d04f8ca765c8b33f284cabd4f63418",
    V3 / "LAUNCH_COMMON_DRIVER_METERS_V3.json": "180225c8ea7d1568bd6a7cb3faec1dcf3243f86e008d132a3442fb55963602b8",
    DATA_GATE / "sealed_role_targets_v2.py": "43cdc72d6bcd7b534d6c661d670b5e718af75bf482aff954ca6507d8d640cdbe",
    DATA_GATE / "full_data_gate.py": "6828136668cece104521f1b34460bf9d06de427649852653352bb09d9e801bef",
    DATA_GATE / "FULL_SPLIT_CACHE_AUDIT.json": "43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd",
    DATA_GATE / "INDEPENDENT_PRODUCTION_CUSTODIAN_V3_POST_SEAL_AUDIT_PASS.json": "09a8c90833fedd44c541f9b61c74c1d4b4414f52b5b009cfe1ac630791a505ab",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(path)], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function absent: {name}")


def _call_lines(function: ast.FunctionDef, called_name: str) -> list[int]:
    result = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ""
        if name == called_name:
            result.append(node.lineno)
    return sorted(result)


def main() -> None:
    meter_sha = sha256_file(V3 / "LAUNCH_COMMON_DRIVER_METERS_V3.json")
    if FROZEN[V3 / "LAUNCH_COMMON_DRIVER_METERS_V3.json"] != meter_sha:
        raise AssertionError("launch common-driver meter summary changed")
    for path, expected in FROZEN.items():
        if sha256_file(path) != expected:
            raise AssertionError(f"frozen dependency changed: {path.name}")
    for path in FILES.values():
        if not path.is_file() or path.is_symlink():
            raise AssertionError(f"V4 source is absent or symlinked: {path.name}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_TARGET_FREE_SOURCES_FROZEN__HOLD_LAUNCH_EXECUTION":
        raise AssertionError("V4 implementation manifest status changed")
    expected_v4_names = {
        "BUNDLE_BINDING_V4.json", "sealed_bundle_gate.py",
        "FEATURE_STORE_CONTRACT_V4.json", "feature_store_contract.py",
        "checkpoint_converter.py", "flopscope_nonzero_worker.py",
        "torch_nonzero_worker.py", "verify_nonzero_parity.py",
        "cpu_training_core.py", "trainer_evaluator_scaffold.py",
        "verify_launch_scaffold.py", "DESIGN_V4.md",
        "NONZERO_PARITY_RECEIPT_V4.json",
    }
    if set(manifest.get("v4_sources", {})) != expected_v4_names:
        raise AssertionError("V4 implementation manifest source membership changed")
    for name, expected in manifest["v4_sources"].items():
        if sha256_file(HERE / name) != expected:
            raise AssertionError(f"V4 implementation manifest hash changed: {name}")
    if any(manifest["execution_state"].get(field) != 0 for field in (
        "feature_store_artifacts", "role_parquet_files_opened_or_hashed_by_v4",
        "target_values_decoded_by_v4", "optimizers_constructed_by_v4",
        "fits_by_v4", "scores_by_v4", "mps_uses_by_v4",
        "graph_or_html_edits_by_v4",
    )):
        raise AssertionError("V4 implementation manifest access counter is nonzero")

    bundle = _run_json(FILES["bundle_gate"])
    features = _run_json(FILES["feature_contract"])
    trainer = _run_json(FILES["trainer"])
    if bundle.get("status") != "PASS_POST_SEAL_METADATA_BINDING__INACTIVE_PENDING_LAUNCH_AUDIT_AND_AUTHORIZATION":
        raise AssertionError("post-seal bundle binding preflight failed")
    if features.get("status") != "PASS_FEATURE_STORE_CONTRACT__HOLD_GENERATION_AND_TRAINING":
        raise AssertionError("feature-store contract preflight failed")
    if trainer.get("status") != "PASS_TARGET_FREE_CPU_TRAINER_SCAFFOLD__HOLD_EXECUTION":
        raise AssertionError("CPU trainer preflight failed")
    zero_fields = (
        "role_parquet_files_opened_or_hashed", "target_values_decoded",
        "optimizers_constructed", "fits", "scores", "mps_used",
    )
    for field in zero_fields:
        if trainer.get(field) not in (0, False):
            raise AssertionError(f"trainer preflight counter is nonzero: {field}")
    if trainer.get("training_claim_exists") or trainer.get("training_output_exists"):
        raise AssertionError("training claim/output exists before authorization")

    trainer_text = FILES["trainer"].read_text(encoding="utf-8")
    trainer_tree = ast.parse(trainer_text)
    execute = _function(trainer_tree, "execute_authorized_once")
    claim_lines = _call_lines(execute, "_consume_claim")
    load_lines = _call_lines(execute, "_load_after_claim")
    run_lines = _call_lines(execute, "run_training_session")
    if len(claim_lines) != 1 or len(load_lines) != 4 or len(run_lines) != 1:
        raise AssertionError("one-use execution chronology call counts changed")
    if not claim_lines[0] < min(load_lines) < run_lines[0]:
        raise AssertionError("claim no longer precedes every fit-capable import/use")
    top_imports = [
        alias.name
        for node in trainer_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    if set(top_imports) & {"torch", "numpy", "pyarrow", "pyarrow.parquet"}:
        raise AssertionError("trainer imports a fit/data runtime before the claim")
    load_after_claim = _function(trainer_tree, "_load_after_claim")
    load_text = ast.get_source_segment(trainer_text, load_after_claim) or ""
    if "CLAIM_PATH.is_file()" not in load_text:
        raise AssertionError("post-claim import helper lost its claim guard")

    core_text = FILES["training_core"].read_text(encoding="utf-8")
    if "pyarrow" in core_text or "sealed_role_targets_v2" in core_text:
        raise AssertionError("CPU core bypasses the injected sealed-reader boundary")
    required_core_tokens = (
        "CONFIGS = (\n    (1536, 128, 256, \"primary\"),\n    (1792, 64, 96, \"host_fidelity_control\"),",
        "ARMS = (\"local_no_recurrence\", \"recurrent\")",
        "FIT_ROWS = 688",
        "SELECTION_ROWS = 88",
        "trace[\"physical_correction\"][0]",
        "exact_two_network_batch_loss",
        "one_optimizer_step",
        "planned_launch_selection",
        "observer_fit",
        "observer_selection",
        "audit_rows_opened\": 0",
        "lockbox_rows_opened\": 0",
        "mps_used\": False",
    )
    for token in required_core_tokens:
        if token not in core_text:
            raise AssertionError(f"CPU core contract token absent: {token}")
    if "paired_promotion(" in core_text:
        raise AssertionError("CPU fit/selection core reaches protected promotion")

    meter = json.loads((V3 / "LAUNCH_COMMON_DRIVER_METERS_V3.json").read_text(encoding="utf-8"))
    expected_meters = {
        "cp1536_h128_c256_local": 205868336750,
        "cp1536_h128_c256_recurrent": 209904700250,
        "cp1792_h64_c96_local": 216168217070,
        "cp1792_h64_c96_recurrent": 218186399450,
    }
    for name, flops in expected_meters.items():
        if meter["runs"][name]["flops"] != flops or meter["runs"][name]["floor_margin"] <= 0:
            raise AssertionError(f"frozen common-driver meter changed: {name}")
        if meter["runs"][name]["descendant_peak_rss_bytes"] >= 8 * 2**30:
            raise AssertionError(f"frozen meter exceeds RSS limit: {name}")
        if meter["runs"][name]["residual_seconds"] >= 0.4:
            raise AssertionError(f"frozen meter exceeds residual time: {name}")
    parity = json.loads(FILES["parity_receipt"].read_text(encoding="utf-8"))
    if parity.get("status") != "PASS_TARGET_FREE_NONZERO_CONVERTER_INTERNAL_STATE_CP_LATCH_PARITY":
        raise AssertionError("nonzero parity receipt is not PASS")
    parity_sources = {
        "checkpoint_converter.py": FILES["converter"],
        "flopscope_nonzero_worker.py": FILES["flopscope_worker"],
        "torch_nonzero_worker.py": FILES["torch_worker"],
        "verify_nonzero_parity.py": FILES["parity_verifier"],
    }
    for name, path in parity_sources.items():
        if parity["source_sha256"].get(name) != sha256_file(path):
            raise AssertionError(f"nonzero parity source binding changed: {name}")
    tolerance = {
        "standardized": (2.0e-6, 2.0e-6),
        "messages": (2.0e-4, 2.0e-4),
        "hidden": (2.0e-4, 2.0e-4),
        "unit": (5.0e-5, 2.0e-4),
        "physical": (5.0e-6, 2.0e-4),
        "predictions": (5.0e-6, 2.0e-4),
    }
    for name, (absolute, relative) in tolerance.items():
        observed = parity["worst_clean_disagreement"][name]
        if observed["max_abs"] > absolute or observed["relative_l2"] > relative:
            raise AssertionError(f"nonzero parity tolerance failed: {name}")
    if parity["worst_clean_disagreement"]["predictions"]["mse"] > 1.0e-10:
        raise AssertionError("nonzero final-row parity MSE failed")
    if parity["latch"]["torch_and_flopscope_exact_trace"] != [True, True] + [False] * 14:
        raise AssertionError("nonzero latch parity failed")
    for field in ("role_parquet_files_opened_or_hashed", "target_values_decoded", "optimizers_constructed", "fits", "scores"):
        if parity.get(field) != 0:
            raise AssertionError(f"nonzero parity target/fit counter changed: {field}")

    result = {
        "schema": "phase2-observer-v4-target-free-launch-scaffold-static-verifier-v1",
        "status": "PASS_TARGET_FREE_V4_LAUNCH_SCAFFOLD__HOLD_EXECUTION",
        "source_sha256": {name: sha256_file(path) for name, path in FILES.items()},
        "implementation_manifest_sha256": sha256_file(MANIFEST_PATH),
        "frozen_dependency_sha256": {str(path.relative_to(PHASE2)): digest for path, digest in FROZEN.items()},
        "claim_before_four_fit_capable_imports_and_training": True,
        "exact_primary": {"cap": 1536, "hidden_channels": 128, "cell_width": 256},
        "exact_control": {"cap": 1792, "hidden_channels": 64, "cell_width": 96},
        "roles": {"observer_fit": 688, "observer_selection": 88},
        "layer1_structural_zero": True,
        "layer2_non_regression_bound_to_frozen_selector": True,
        "common_driver_meter_receipt_bound": True,
        "feature_store_status": "HOLD_NOT_GENERATED_OR_AUDITED",
        "independent_launch_audit_status": "HOLD_NOT_YET_ISSUED",
        "reader_authorization_status": "HOLD_NOT_YET_ISSUED",
        "one_use_training_authorization_status": "HOLD_NOT_YET_ISSUED",
        "role_parquet_files_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "mps_used": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
