#!/usr/bin/env python3
"""Read-only verifier for the frozen V4 one-attempt board-validation seal."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
P2 = HERE.parent
SOURCE_DIR = P2 / "fullbudget_k21_k4_closure_v4"
ARCHIVE = REPO / "submissions/fb-ec4-32-v4-3ddddd87.tar.gz"
METER = P2 / "meter_protocol/runs/fb_ec4_v4_sparse_resetarena_cp3compact_n1_s8_seed20260824/receipt.json"
METER_VERIFIER = P2 / "meter_protocol/verify_meter_run.py"
PINNED_PYTHON = P2 / "data_gate/.venv/bin/python"
PINNED_WHEST = P2 / "data_gate/.venv/bin/whest"
MOMENT = P2 / "fullbudget_k21_k4_closure_v4_moment_diagnostic/runs/v4_target_free_moment_seed20260824/receipt.json"

EXPECTED = {
    SOURCE_DIR / "estimator.py": "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c",
    SOURCE_DIR / "SOURCE_MANIFEST.json": "0cad2cc4491fbb22de5dbf021612fac9d5dccd2c2dcb1f4e963314d0c90580bf",
    SOURCE_DIR / "FREEZE_RECEIPT_V4.json": "b730a3e21ba1d297de4b4be6c938cf842f955d951fd008518971cc0fafd8295e",
    SOURCE_DIR / "INDEPENDENT_SOURCE_AUDIT_V4.json": "c9553e1596e8e61b29c862256487586d38f66c92a8389cd1541c5b1e9ffa5a9d",
    METER: "14d6eeb53d2f202241457ad32669f82ed6dc26107e572a061184aa2f8fec5017",
    MOMENT: "93124980ee134d2b0bad4e6f00b46ced6a7329fcc356945c12736443f4723ee8",
    ARCHIVE: "2df56c53253494fc75da3de8a0a7bd29d0bf2b97fa8f052bd2378ba7534d5a82",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def exact_imports(tree: ast.AST) -> list[str]:
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(result)


def main() -> None:
    checks: dict[str, bool] = {}
    for path, expected in EXPECTED.items():
        checks[f"sha256:{path.name}:{expected[:8]}"] = path.is_file() and sha256_file(path) == expected

    source_bytes = (SOURCE_DIR / "estimator.py").read_bytes()
    tree = ast.parse(source_bytes)
    checks["imports_exact"] = exact_imports(tree) == sorted([
        "flopscope", "flopscope.numpy", "whestbench.BaseEstimator", "whestbench.domain.MLP"
    ])
    source_text = source_bytes.decode()
    checks["guard_constants"] = all(token in source_text for token in (
        "_PRODUCTION_WIDTH = 1024",
        "_PRODUCTION_DEPTH = 16",
        "_PHASE2_FLOP_BUDGET = 2_199_023_255_552",
    ))
    checks["guard_zero_shape_dtype"] = (
        "return fnp.zeros((smoke_depth, smoke_width), dtype=fnp.float32)" in source_text
        and "or int(budget) < _PHASE2_FLOP_BUDGET" in source_text
    )

    with tarfile.open(ARCHIVE, "r:gz") as handle:
        members = handle.getmembers()
        checks["archive_names_exact"] = [member.name for member in members] == ["estimator.py", "manifest.json"]
        checks["archive_regular_only"] = all(member.isfile() for member in members)
        payloads = {member.name: handle.extractfile(member).read() for member in members}
    checks["archive_estimator_exact"] = payloads["estimator.py"] == source_bytes
    checks["archive_estimator_sha"] = sha256_bytes(payloads["estimator.py"]) == EXPECTED[SOURCE_DIR / "estimator.py"]
    checks["archive_manifest_sha"] = sha256_bytes(payloads["manifest.json"]) == "58850abbd1f8fe93132187727a5173e27cb3fb8d6053ff06d1d55fdde8e70b5f"
    package_manifest = json.loads(payloads["manifest.json"])
    checks["archive_manifest_contract"] = (
        package_manifest["entrypoint"] == {"module": "estimator", "class": "Estimator"}
        and package_manifest["files"] == [{"name": "estimator.py", "sha256": EXPECTED[SOURCE_DIR / "estimator.py"]}]
        and package_manifest["whestbench_version"] == "0.16.0"
        and package_manifest["flopscope_version"] == "0.12.0"
        and package_manifest["numpy_version"] == "2.4.6"
    )

    validate_package = subprocess.run(
        [str(PINNED_WHEST), "validate-package", str(ARCHIVE), "--format", "json"],
        check=True, capture_output=True, text=True,
    )
    checks["validate_package"] = json.loads(validate_package.stdout)["ok"] is True
    with tempfile.TemporaryDirectory(prefix="fb-ec4-v4-board-smoke-") as temp_dir:
        estimator_path = Path(temp_dir) / "estimator.py"
        estimator_path.write_bytes(payloads["estimator.py"])
        smoke = subprocess.run(
            [str(PINNED_WHEST), "validate", "--estimator", str(estimator_path), "--format", "json"],
            check=True, capture_output=True, text=True,
        )
        smoke_json = json.loads(smoke.stdout)
    checks["exact_extracted_smoke"] = smoke_json["ok"] is True and smoke_json["output_shape"] == [2, 4]

    checks["installed_versions"] = (
        version("whestbench") == "0.16.0"
        and version("flopscope") == "0.12.0"
        and version("numpy") == "2.4.6"
    )
    meter_verify = subprocess.run(
        [str(PINNED_PYTHON), str(METER_VERIFIER), str(METER.parent)],
        check=True, capture_output=True, text=True,
    )
    checks["meter_verifier"] = json.loads(meter_verify.stdout)["status"] == "PASS_METER_RECEIPT_VERIFICATION"
    meter = load(METER)
    checks["meter_values"] = (
        meter["status"] == "PASS_PHASE2_SUBPROCESS_INTERFACE_AND_METER__ACCURACY_UNSCORED"
        and meter["measured"]["flops_used_per_mlp"] == [2179742954469]
        and meter["measured"]["residual_wall_time_s_per_mlp"] == [0.3926322143524885]
        and meter["measured"]["n_failed_mlps"] == 0
        and meter["measured"]["output_contract"]["all_values_finite"] is True
    )

    moment = load(MOMENT)
    rows = moment["rows"]
    negative_layers = sorted({
        row["layer"] for row in rows if row["count_A_lt_zero"] or row["count_B_lt_zero"]
    })
    checks["moment_hold_bound"] = (
        moment["status"] == "SCIENTIFIC_HOLD__SEE_GATES"
        and moment["source"]["v4_estimator_sha256"] == EXPECTED[SOURCE_DIR / "estimator.py"]
        and moment["gates"]["all_7x1024_values_finite_each_layer"] is True
        and moment["gates"]["all_exact_necessary_margins_nonnegative"] is False
        and all(row["count_r1_lt_zero"] == 0 and row["count_v_lt_zero"] == 0 for row in rows)
        and negative_layers == [10, 11, 12, 13, 14, 15, 16]
    )

    manifest = load(HERE / "PROMOTION_MANIFEST.json")
    binding = load(HERE / "MOMENT_RESULT_BINDING.json")
    audit = load(HERE / "STATIC_AUDIT.json")
    checks["one_attempt_law"] = (
        manifest["candidate"]["count"] == 1
        and manifest["authorization"]["max_attempts_this_preregistration"] == 1
        and manifest["outcome_law"]["tuning"] is False
        and manifest["outcome_law"]["fallback"] is False
        and manifest["outcome_law"]["retry"] is False
        and manifest["primary_criterion"] == {
            "candidate_count": 1,
            "metric": "observed_unrounded_board_raw_final_layer_mse",
            "operator": "<",
            "threshold": 1e-8,
            "equality_passes": False,
            "adjusted_or_secondary_metric_can_rescue": False,
        }
    )
    checks["pending_then_bound_hold"] = (
        binding["preregistered_pending_state"] == {
            "status": "PENDING", "result_path": None, "result_sha256": None, "verdict": None
        }
        and binding["current_state"]["result_sha256"] == EXPECTED[MOMENT]
        and binding["successor"]["status"] == "PENDING"
        and binding["successor"]["result_path"] is None
        and binding["successor"]["result_sha256"] is None
        and binding["board_boundary"]["scientific_promotion_authorized"] is False
    )
    checks["not_submitted"] = (
        manifest["submission"]["attempt_state"] == "PREPARED_NOT_ATTEMPTED"
        and manifest["submission"]["submission_id"] is None
        and audit["negative_controls"]["submissions"] == 0
        and audit["negative_controls"]["network_accesses"] == 0
        and audit["negative_controls"]["credentials_accesses"] == 0
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"board-validation verification failed: {failed}")
    print(json.dumps({
        "status": "PASS_EXPLORATORY_BOARD_VALIDATION_SEAL__SCIENTIFIC_PROMOTION_HOLD",
        "checks": checks,
        "submission_attempt_state": "PREPARED_NOT_ATTEMPTED",
        "remaining_scientific_blocker": "scale-normalized worst-unit moment classification",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

