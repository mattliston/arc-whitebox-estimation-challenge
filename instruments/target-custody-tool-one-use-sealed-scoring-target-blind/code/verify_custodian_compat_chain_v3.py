#!/usr/bin/env python3
"""Target-free static/runtime verifier for the additive v3 custody chain.

Only deterministic in-memory or /private/tmp synthetic values are decoded.
The verifier never imports or advances the production raw iterator, creates a
production claim, or accesses a production Parquet payload.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
SEALER = HERE / "custodian_seal_training_roles_once_v3.py"
SYNTHETIC = HERE / "run_synthetic_custodian_chain_v3.py"
PYARROW25_PYTHON = HERE / ".venv/bin/python"
PYARROW11_PYTHON = Path("/Users/matt/anaconda3/bin/python")
RAW_MODULE = "authorized_full_targets_v2"

NEW_FROZEN_SHA256 = {
    "authorized_full_targets_v2.py": "95ffd5ba0e55823a3176a35e42af052feebc4cb462dcfa9738b135d2110a7c83",
    "sealed_role_targets_v2.py": "43cdc72d6bcd7b534d6c661d670b5e718af75bf482aff954ca6507d8d640cdbe",
    "target_blind_mini_v2.py": "6778a3216377a4a7e588df71a08fb0e182dc7a14d42ab393c2ac80200cccf0e1",
    "role_target_artifact_codec_v1.py": "f09855d8843492ead6b3117535d19487a45988d6cd9fbacd2e642798e4c79b17",
    "custodian_seal_training_roles_once_v3.py": "96e0da1257c1da78cbd6cd9709b29b59d6afa4f5dd7ce0ec5d4d1ac336c711cb",
    "run_synthetic_custodian_chain_v3.py": "fa14f2090f76d5ae1e8cfb8b7f8fc6bbf087ffe571fec1da2b76637a941ec177",
}
PRESERVED_SHA256 = {
    "authority.json": "b33aea75ee83be5cc7c2778aefd4cc35d731dcd702d8b258269fc2c895ef4ec5",
    "full_network_splits.json": "ba325d5c9ee4382ebf856b2e8d15330852fba5ac5627665fcd81dd9a5ef461d1",
    "custodian_seal_training_roles_once.py": "cad1ac6c72b374b04ec53bde8d7aac6112ac2ccbaa818c2f2a0195160a2edab4",
    "verify_custodian_sealer_static.py": "f2ee6fb004d83d46b25ac15b1e247f766e6248cd6b99784fae49d0885a9d06fc",
    "CUSTODIAN_SEALER_STATIC_AUDIT.json": "1ee628c5849cc7f943ec52ad11ea8ea91ab8e2e526741f66717f256de466e7c3",
    "CUSTODIAN_SEALER_FROZEN_PROTOCOL.md": "9a0227c97741b936706e81236cf0c9f891a90ae588328581177d6468d26e5407",
    "custodian_seal_training_roles_once_v2.py": "f9125ecbec76ddb7ea78d35144fc04c2adeda5cc254e59740663b8f1f910c66a",
    "verify_custodian_sealer_static_v2.py": "0a47365baa01e20595a10d975fac5382c9ce7fab825304ea9eba1d7672dce6ae",
    "CUSTODIAN_SEALER_V2_STATIC_AUDIT.json": "ee65309fcc677bf9cf19b0ced6ff771c9d03116b12b146b2fafc0733230606ee",
    "CUSTODIAN_SEALER_V2_STRICT_PROTOCOL.md": "4aa87216812d09431a9cd248f174625b7201b42ad56724c630442f78aefa587c",
    "INDEPENDENT_CUSTODIAN_SEALER_STRICT_REAUDIT_HOLD_V1.json": "5265cea2278d73e868f5f8c0a4a2e7ab7c01c21ec6b73902f88c985141618f39",
    "INDEPENDENT_CUSTODIAN_SEALER_V2_AUDIT_HOLD.json": "c2f37b523e66ab39daa0c7ea5f489f2cb9959fed0bf22c54fd6c4ecc4f1070c9",
}
PRODUCTION_PATHS = {
    "v1_claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM",
    "v1_staging": HERE / "sealed_role_target_bundle.staging",
    "v2_claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V2.CLAIM",
    "v2_staging": HERE / "sealed_role_target_bundle.v2.staging",
    "v3_claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V3.CLAIM",
    "v3_staging": HERE / "sealed_role_target_bundle.v3.staging",
    "release": HERE / "sealed_role_target_bundle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_hashes() -> None:
    for name, expected in {**PRESERVED_SHA256, **NEW_FROZEN_SHA256}.items():
        actual = _sha256(HERE / name)
        if actual != expected:
            raise AssertionError(f"frozen source/receipt changed: {name}: {actual}")


def _assert_production_paths_absent() -> None:
    present = [
        label
        for label, path in PRODUCTION_PATHS.items()
        if path.exists() or path.is_symlink()
    ]
    if present:
        raise AssertionError(f"production custody paths exist: {present}")


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _assigned_call(statement: ast.stmt, target: str, call: str) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    assigned = statement.targets[0]
    return (
        isinstance(assigned, ast.Name)
        and assigned.id == target
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == call
    )


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one function: {name}")
    return matches[0]


def _verify_reader_compatibility_static() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "authorized_full_targets_v2.py",
        "sealed_role_targets_v2.py",
        "target_blind_mini_v2.py",
    ):
        source = (HERE / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "column_names"
        ]
        schema_names = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "names"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "schema"
            )
        ]
        if forbidden or not schema_names:
            raise AssertionError(f"reader is not schema.names-only: {name}")
        result[name] = {
            "sha256": NEW_FROZEN_SHA256[name],
            "batch_column_names_references": 0,
            "batch_schema_names_references": len(schema_names),
        }
    return result


def _verify_sealer_order_static() -> dict[str, Any]:
    source = SEALER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    if RAW_MODULE in top_level_imports or "importlib" in top_level_imports:
        raise AssertionError("raw/importlib loader is imported at module load")
    raw_static_imports = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "") == RAW_MODULE
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name == RAW_MODULE for alias in node.names)
        )
    ]
    if raw_static_imports:
        raise AssertionError("v3 contains a static raw-reader import")

    body = _function(tree, "execute").body
    labels: dict[str, list[int]] = {
        "runtime_check": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "runtime", "_validate_runtime_capabilities_preclaim")
        ],
        "codec_load": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "artifact_codec", "_load_artifact_codec_preclaim")
        ],
        "claim": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "claim_sha256", "_consume_fixed_claim")
        ],
        "importlib": [
            i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Import)
            and [alias.name for alias in stmt.names] == ["importlib"]
        ],
        "raw_module": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "raw_target_module", "import_module")
        ],
        "factory": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "raw_iterator_factory", "getattr")
        ],
        "iterator": [
            i for i, stmt in enumerate(body)
            if _assigned_call(stmt, "raw_iterator", "raw_iterator_factory")
        ],
        "decode": [
            i for i, stmt in enumerate(body)
            if isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "raw_row"
            and isinstance(stmt.iter, ast.Name)
            and stmt.iter.id == "raw_iterator"
        ],
    }
    if any(len(indices) != 1 for indices in labels.values()):
        raise AssertionError(f"v3 control-point cardinality changed: {labels}")
    order = {label: indices[0] for label, indices in labels.items()}
    if not (
        order["runtime_check"] < order["codec_load"] < order["claim"]
        and order["importlib"] == order["claim"] + 1
        and order["raw_module"] == order["importlib"] + 1
        and order["factory"] == order["raw_module"] + 1
        and order["iterator"] == order["factory"] + 1
        and order["iterator"] < order["decode"]
    ):
        raise AssertionError(f"runtime/claim/import/iterator/decode order changed: {order}")

    runtime_function = _function(tree, "_validate_runtime_capabilities_preclaim")
    runtime_source = ast.get_source_segment(source, runtime_function) or ""
    for required in (
        "EXPECTED_EXECUTABLE",
        "EXPECTED_PYTHON",
        "EXPECTED_PYARROW",
        "batch.schema.names",
        "ParquetFile.iter_batches",
        "FixedSizeListArray.from_arrays",
    ):
        if required not in runtime_source:
            raise AssertionError(f"runtime capability check lacks {required}")
    return {
        "sealer_sha256": NEW_FROZEN_SHA256[SEALER.name],
        "control_order": order,
        "runtime_checked_before_claim": True,
        "claim_immediately_precedes_raw_loader_chain": True,
        "raw_reader_import_at_module_load": False,
    }


def _guarded_sealer_run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    blocker = f'''\
import importlib.abc
import runpy
import sys
class RawBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {RAW_MODULE!r}:
            raise RuntimeError("RAW_IMPORT_BLOCKED")
        return None
sys.meta_path.insert(0, RawBlocker())
sys.argv = [{str(SEALER)!r}] + {arguments!r}
runpy.run_path({str(SEALER)!r}, run_name="__main__")
'''
    return subprocess.run(
        [str(PYARROW25_PYTHON), "-c", blocker],
        cwd=HERE,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def _verify_guarded_preclaim_runtime() -> dict[str, Any]:
    proof = _guarded_sealer_run(["--prove-runtime-preclaim-capabilities"])
    if proof.returncode != 0 or "RAW_IMPORT_BLOCKED" in proof.stderr:
        raise AssertionError(f"guarded preclaim proof failed: {proof.stderr}")
    payload = json.loads(proof.stdout)
    if (
        payload.get("status") != "PASS_V3_RUNTIME_AND_PRECLAIM_RAW_IMPORT_PROOF"
        or payload.get("raw_target_module_imported") is not False
        or payload.get("production_paths_absent") is not True
        or payload.get("claim_created") is not False
        or payload.get("target_values_decoded") != 0
    ):
        raise AssertionError("guarded preclaim proof payload changed")
    refused = _guarded_sealer_run([])
    if (
        refused.returncode == 0
        or "RAW_IMPORT_BLOCKED" in refused.stderr
        or "no executable preflight mode" not in refused.stderr
    ):
        raise AssertionError("ordinary refused invocation imported raw reader or succeeded")
    return {
        "exact_interpreter": payload["runtime"]["executable"],
        "python": payload["runtime"]["python"],
        "pyarrow": payload["runtime"]["pyarrow"],
        "required_apis": payload["runtime"]["required_apis"],
        "raw_import_blocker_triggered": False,
        "claim_created": False,
        "target_values_decoded": 0,
    }


def _run_synthetic(interpreter: Path, expected_arrow: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(interpreter), str(SYNTHETIC)],
        cwd=HERE,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"synthetic compatibility run failed under {interpreter}: {result.stderr}"
        )
    payload = json.loads(result.stdout)
    if (
        payload.get("status") != "PASS_ISOLATED_SYNTHETIC_FULL_SHAPE_CUSTODY_ROUNDTRIP"
        or payload.get("pyarrow") != expected_arrow
        or payload["synthetic_source"].get("all_layer_means_shape_per_row") != [16, 1024]
        or payload["raw_decode_accounting"].get("production_rows") != 0
        or payload.get("withheld_role_artifacts_written") != 0
        or payload.get("fit_or_score_performed") is not False
        or payload.get("production_paths_touched") != 0
        or payload.get("production_target_values_decoded") != 0
    ):
        raise AssertionError("synthetic compatibility receipt changed")
    return payload


def verify() -> dict[str, Any]:
    _assert_production_paths_absent()
    _assert_hashes()
    readers = _verify_reader_compatibility_static()
    sealer = _verify_sealer_order_static()
    guarded = _verify_guarded_preclaim_runtime()
    arrow25 = _run_synthetic(PYARROW25_PYTHON, "25.0.1")
    arrow11 = _run_synthetic(PYARROW11_PYTHON, "11.0.0")
    _assert_production_paths_absent()
    _assert_hashes()

    for role in ("observer_fit", "observer_selection"):
        hash25 = arrow25["sealed_artifacts"][role]["target_values_c_order_sha256"]
        hash11 = arrow11["sealed_artifacts"][role]["target_values_c_order_sha256"]
        if hash25 != hash11:
            raise AssertionError(f"cross-version target bytes changed: {role}")
    if (
        arrow25["synthetic_source"]["target_values_c_order_sha256"]
        != arrow11["synthetic_source"]["target_values_c_order_sha256"]
    ):
        raise AssertionError("cross-version source target bytes changed")

    return {
        "schema": "whest-phase2-custodian-compatibility-chain-v3-static-audit-v1",
        "status": "PASS_TARGET_FREE_V3_DESIGN_AND_SYNTHETIC_COMPATIBILITY",
        "authority_sha256": PRESERVED_SHA256["authority.json"],
        "roster_sha256": PRESERVED_SHA256["full_network_splits.json"],
        "preserved_v1_v2_files": PRESERVED_SHA256,
        "new_frozen_files": NEW_FROZEN_SHA256,
        "reader_static_compatibility": readers,
        "sealer_static_control_order": sealer,
        "guarded_preclaim_runtime": guarded,
        "synthetic_roundtrips": {
            "pyarrow_25_0_1": arrow25,
            "pyarrow_11_0_0": arrow11,
            "cross_version_target_bytes_equal": True,
        },
        "production_claim_created": False,
        "production_release_created": False,
        "production_target_values_decoded": 0,
        "fit_or_score_performed": False,
        "independent_reaudit_required_before_release": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
