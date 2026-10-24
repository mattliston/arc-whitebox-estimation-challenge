#!/usr/bin/env python3
"""Strict target-free audit of the claim-before-import v2 custodian sealer.

The verifier never imports the raw target module. It combines AST proof with
two guarded subprocess proofs: an ordinary refused invocation and the explicit
preclaim proof mode both run behind a meta-path blocker that raises on any
attempt to import ``authorized_full_targets``. No production claim or release
is created and no dataset payload is accessed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
SEALER_PATH = HERE / "custodian_seal_training_roles_once_v2.py"
PROOF_PYTHON = HERE / ".venv/bin/python"
RAW_MODULE = "authorized_full_targets"
EXPECTED_SEALER_SHA256 = (
    "f9125ecbec76ddb7ea78d35144fc04c2adeda5cc254e59740663b8f1f910c66a"
)
EXPECTED_ROSTER_SHA256 = (
    "ba325d5c9ee4382ebf856b2e8d15330852fba5ac5627665fcd81dd9a5ef461d1"
)
ASSIGNMENT_DOMAIN = b"p2-teacher-v1|v2-phase2|"
EXPECTED_ROLE_SEGMENTS = (
    ("retired_teacher_diagnostics", 0, 12),
    ("observer_fit", 12, 700),
    ("retired_teacher_diagnostics", 700, 712),
    ("observer_selection", 712, 800),
    ("observer_audit", 800, 900),
    ("observer_lockbox", 900, 1000),
)
EXPECTED_ROLE_COUNTS = {
    "observer_fit": 688,
    "observer_selection": 88,
    "observer_audit": 100,
    "observer_lockbox": 100,
    "retired_teacher_diagnostics": 24,
}
PRESERVED_SHA256 = {
    "authority.json": "b33aea75ee83be5cc7c2778aefd4cc35d731dcd702d8b258269fc2c895ef4ec5",
    "FULL_SPLIT_AUTHORITY_EXTENSION.json": "d61677658462f22949d8242f7bb92b24341502678f499d793184002fd3b6027a",
    "FULL_SPLIT_AUTHORITY_AUDIT.json": "a5f9e0b42b14f36a2736555d1c9a4a018d8cc5e6935937b40615d79d37edda10",
    "FULL_SPLIT_CACHE_AUDIT.json": "43857f77f86f2ce95f058bda7d314d7c836b60aedab5a40d37b7208f291494fd",
    "FULL_SPLIT_PREFIX_PRESERVATION_AUDIT.json": "6569d478615c1a68818229bf328b93340f342296be14cd5529b220e007961747",
    "full_network_splits.json": EXPECTED_ROSTER_SHA256,
    "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json": "b73e6ccf24434d27cc1bc34cc5ebcd87b9b38b504c3412bf9afad571ecb4b49c",
    "FULL_SPLIT_GATE_STRUCTURE_AUDIT.json": "8e75e15b52d411be194becf5f2f956e09b6b62d7970b9d0ea2d47a21e4cd5b77",
    "FULL_SPLIT_GATE.md": "dfee4994d120251512c564c702fe3df02b20ec01b52626962fd6da53b5373ade",
    "ROLE_TARGET_CUSTODY_PROTOCOL.md": "c26a12340433e3d4ffd5ddd2871a918fc70c5dc0bde789e7c7c0eeb75efe5f5b",
    "full_shards.tsv": "813308bb891c46e1849cace8dfef60c5faf0c3c4a0c8bfd5946afbb5c24a2155",
    "full_data_gate.py": "6828136668cece104521f1b34460bf9d06de427649852653352bb09d9e801bef",
    "download_full_split.py": "0c0f264adab6da7dadadd0fa2178668ccd36f200b0bd09382cfc01da1f36093a",
    "build_full_network_splits.py": "227c68eb468d04b9f01e70d61060cce911d11b1a53492619b9dbf0685173e637",
    "verify_full_network_splits.py": "0aba47c1324cc15b09f993584c355ee5c4890b47622b947c7951ff0b90d78fa3",
    "audit_role_target_addressability.py": "aef3226a74824fa14f952b93e9d2ea3645d76b7a38d06810ab817861594d4d81",
    "authorized_full_targets.py": "dfea9dc4f6e1096e9bd078f5594bc94a4dc5f85e550077f6e954a4f56512dd96",
    "sealed_role_targets.py": "194d41ea7afbeb4a2171c0ff55fe1a4321c0c817a6c23466cc21fcb23b850e91",
    "target_access_authorization.example.json": "92cd759992d6b82469d7b6cac4b854ae3691f0e322218c084bb950f08c4d56a5",
    "sealed_role_target_access_authorization.example.json": "5c8520f98b58d9328d31332fbda32a444ddd156e906ef375881c94c2a2e5cb21",
    "custodian_seal_training_roles_once.py": "cad1ac6c72b374b04ec53bde8d7aac6112ac2ccbaa818c2f2a0195160a2edab4",
    "verify_custodian_sealer_static.py": "f2ee6fb004d83d46b25ac15b1e247f766e6248cd6b99784fae49d0885a9d06fc",
    "CUSTODIAN_SEALER_STATIC_AUDIT.json": "1ee628c5849cc7f943ec52ad11ea8ea91ab8e2e526741f66717f256de466e7c3",
    "CUSTODIAN_SEALER_FROZEN_PROTOCOL.md": "9a0227c97741b936706e81236cf0c9f891a90ae588328581177d6468d26e5407",
}
RUNTIME_PATHS = {
    "v1_claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM",
    "v1_staging": HERE / "sealed_role_target_bundle.staging",
    "v2_claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE_V2.CLAIM",
    "v2_staging": HERE / "sealed_role_target_bundle.v2.staging",
    "release": HERE / "sealed_role_target_bundle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one function named {name}")
    return matches[0]


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            continue
        value = statement.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"frozenset", "set", "tuple"}
            and len(value.args) == 1
            and not value.keywords
        ):
            value = value.args[0]
        return ast.literal_eval(value)
    raise AssertionError(f"missing literal assignment: {name}")


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


def _role_for_rank(rank: int) -> str:
    matches = [
        role for role, begin, end in EXPECTED_ROLE_SEGMENTS if begin <= rank < end
    ]
    if len(matches) != 1:
        raise AssertionError("role segments do not uniquely cover roster")
    return matches[0]


def _verify_roster_target_free() -> dict[str, Any]:
    path = HERE / "full_network_splits.json"
    if _sha256(path) != EXPECTED_ROSTER_SHA256:
        raise AssertionError("frozen roster SHA-256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if bytes.fromhex(payload["assignment"]["domain_hex"]) != ASSIGNMENT_DOMAIN:
        raise AssertionError("roster assignment domain changed")
    if payload["assignment"]["assignment_inputs"] != ["mlp_seed"]:
        raise AssertionError("roster assignment is not seed-only")
    if payload["assignment"]["attestations_not_assignment_inputs"] != [
        "mlp_id",
        "weights_sha256",
    ]:
        raise AssertionError("roster assignment/attestation boundary changed")
    ranked: list[dict[str, Any]] = []
    for split, count in (
        ("train", 700),
        ("calibration", 100),
        ("validation", 100),
        ("test", 100),
    ):
        if len(payload["splits"][split]) != count:
            raise AssertionError("primary split size changed")
        ranked.extend(payload["splits"][split])
    if [int(row["global_hash_rank"]) for row in ranked] != list(range(1000)):
        raise AssertionError("roster global ranks changed")
    roles: Counter[str] = Counter()
    ids: set[int] = set()
    seeds: set[int] = set()
    digests: list[str] = []
    for rank, row in enumerate(ranked):
        mlp_id = int(row["mlp_id"])
        seed = int(row["mlp_seed"])
        digest = hashlib.sha256(
            ASSIGNMENT_DOMAIN + str(seed).encode("ascii")
        ).hexdigest()
        if row["assignment_sha256"] != digest:
            raise AssertionError("roster seed-only digest changed")
        role = _role_for_rank(rank)
        if row["observer_role"] != role:
            raise AssertionError("roster role slice changed")
        ids.add(mlp_id)
        seeds.add(seed)
        digests.append(digest)
        roles[role] += 1
    if len(ids) != 1000 or len(seeds) != 1000 or len(set(digests)) != 1000:
        raise AssertionError("roster identity uniqueness changed")
    if digests != sorted(digests) or dict(roles) != EXPECTED_ROLE_COUNTS:
        raise AssertionError("roster ordering or role counts changed")
    return {
        "sha256": EXPECTED_ROSTER_SHA256,
        "assignment_domain_ascii": ASSIGNMENT_DOMAIN.decode("ascii"),
        "assignment_inputs": ["mlp_seed"],
        "networks": 1000,
        "role_counts": EXPECTED_ROLE_COUNTS,
        "target_values_decoded": 0,
    }


def _verify_physical_constraint_target_free() -> dict[str, Any]:
    path = HERE / "FULL_SPLIT_ROLE_ADDRESSABILITY_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != (
        "EXTERNAL_CONSTRAINT__RAW_PARQUET_CANNOT_PRESERVE_ROLE_TARGET_SEALS"
    ):
        raise AssertionError("raw physical-seal constraint changed")
    collisions = payload["training_role_row_groups_colliding_with_audit_or_lockbox"]
    if (
        payload["raw_role_projection_physically_safe"] is not False
        or payload["row_groups"] != 63
        or len(collisions) != 62
        or payload["network_independent_row_groups"] != 0
        or payload["role_pure_row_groups"] != 0
        or payload["all_layer_means_column_index_available_row_groups"] != 0
        or payload["all_layer_means_offset_index_available_row_groups"] != 0
        or payload["target_values_decoded"] != 0
    ):
        raise AssertionError("raw addressability evidence changed")
    return {
        "sha256": _sha256(path),
        "row_groups": 63,
        "training_protected_collisions": 62,
        "raw_role_projection_physically_safe": False,
        "target_values_decoded": 0,
    }


def _verify_source_order() -> dict[str, Any]:
    raw = SEALER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SEALER_SHA256:
        raise AssertionError("frozen v2 sealer source changed")
    source = raw.decode("utf-8")
    compile(source, str(SEALER_PATH), "exec")
    tree = ast.parse(source)

    if _literal_assignment(tree, "RAW_TARGET_MODULE_NAME") != RAW_MODULE:
        raise AssertionError("raw target module name changed")
    if tuple(_literal_assignment(tree, "RAW_TARGET_COLUMNS")) != (
        "mlp_id",
        "all_layer_means",
    ):
        raise AssertionError("raw target projection changed")
    if frozenset(_literal_assignment(tree, "PERSISTED_ROLES")) != {
        "observer_fit",
        "observer_selection",
    }:
        raise AssertionError("persisted-role allow-list changed")
    if frozenset(_literal_assignment(tree, "WITHHELD_ROLES")) != {
        "observer_audit",
        "observer_lockbox",
        "retired_teacher_diagnostics",
    }:
        raise AssertionError("withheld-role deny-list changed")

    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    if RAW_MODULE in top_level_imports or "importlib" in top_level_imports:
        raise AssertionError("raw/importlib module is imported before claim")
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
        raise AssertionError("v2 contains a static raw target import")

    execute = _function(tree, "execute")
    body = execute.body
    claim_indices = [
        index
        for index, statement in enumerate(body)
        if _assigned_call(statement, "claim_sha256", "_consume_fixed_claim")
    ]
    import_indices = [
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Import)
        and [alias.name for alias in statement.names] == ["importlib"]
    ]
    module_indices = [
        index
        for index, statement in enumerate(body)
        if _assigned_call(statement, "raw_target_module", "import_module")
    ]
    factory_indices = [
        index
        for index, statement in enumerate(body)
        if _assigned_call(statement, "raw_iterator_factory", "getattr")
    ]
    iterator_indices = [
        index
        for index, statement in enumerate(body)
        if _assigned_call(statement, "raw_iterator", "raw_iterator_factory")
    ]
    decode_indices = [
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "raw_row"
        and isinstance(statement.iter, ast.Name)
        and statement.iter.id == "raw_iterator"
    ]
    cardinalities = tuple(
        len(indices)
        for indices in (
            claim_indices,
            import_indices,
            module_indices,
            factory_indices,
            iterator_indices,
            decode_indices,
        )
    )
    if cardinalities != (1, 1, 1, 1, 1, 1):
        raise AssertionError(f"claim/import/iterator/decode cardinality changed: {cardinalities}")
    claim_i = claim_indices[0]
    import_i = import_indices[0]
    module_i = module_indices[0]
    factory_i = factory_indices[0]
    iterator_i = iterator_indices[0]
    decode_i = decode_indices[0]
    if import_i != claim_i + 1 or module_i != import_i + 1:
        raise AssertionError("raw import is not the immediate action after claim")
    if not (claim_i < import_i < module_i < factory_i < iterator_i < decode_i):
        raise AssertionError("execute order is not claim -> import -> iterator -> decode")

    before_claim = ast.Module(body=body[:claim_i], type_ignores=[])
    forbidden_preclaim = [
        node
        for node in ast.walk(before_claim)
        if isinstance(node, ast.Import)
        or isinstance(node, ast.ImportFrom)
        or (
            isinstance(node, ast.Call)
            and _call_name(node) in {"import_module", "__import__"}
        )
    ]
    if forbidden_preclaim:
        raise AssertionError("module-loader action exists before fixed claim")

    module_assignment = body[module_i]
    assert isinstance(module_assignment, ast.Assign)
    module_call = module_assignment.value
    assert isinstance(module_call, ast.Call)
    if (
        len(module_call.args) != 1
        or not isinstance(module_call.args[0], ast.Name)
        or module_call.args[0].id != "RAW_TARGET_MODULE_NAME"
    ):
        raise AssertionError("dynamic raw import does not use frozen module constant")
    factory_assignment = body[factory_i]
    assert isinstance(factory_assignment, ast.Assign)
    factory_call = factory_assignment.value
    assert isinstance(factory_call, ast.Call)
    if (
        len(factory_call.args) != 2
        or not isinstance(factory_call.args[0], ast.Name)
        or factory_call.args[0].id != "raw_target_module"
        or not isinstance(factory_call.args[1], ast.Constant)
        or factory_call.args[1].value != "iter_trusted_custodian_all_layer_means"
    ):
        raise AssertionError("raw iterator symbol resolution changed")

    claim_function = _function(tree, "_consume_fixed_claim")
    exclusive_writer = _function(tree, "_write_exclusive_fsynced")
    exclusive_attributes = {
        node.attr for node in ast.walk(exclusive_writer) if isinstance(node, ast.Attribute)
    }
    claim_call_names = {
        _call_name(node)
        for node in ast.walk(claim_function)
        if isinstance(node, ast.Call)
    }
    if (
        "O_EXCL" not in exclusive_attributes
        or "fsync" not in exclusive_attributes
        or "_write_exclusive_fsynced" not in claim_call_names
        or "_fsync_directory" not in claim_call_names
    ):
        raise AssertionError("claim is not durable O_EXCL before import")

    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {_call_name(node) for node in all_calls}
    if called_names & {"unlink", "remove", "rmdir", "removedirs", "rmtree"}:
        raise AssertionError("claim cleanup/retry path entered v2")
    if called_names & {"fit", "fit_transform", "predict", "score", "backward"}:
        raise AssertionError("fit/predict/score operation entered v2")
    writer_calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and _call_name(node) == "_write_role_artifact"
    ]
    writer_roles = [
        node.args[0].value
        for node in writer_calls
        if node.args and isinstance(node.args[0], ast.Constant)
    ]
    if writer_roles != ["observer_fit", "observer_selection"]:
        raise AssertionError("v2 writes a role other than fit and selection")
    if "fsync" not in called_names or "replace" not in called_names or "rename" not in called_names:
        raise AssertionError("durable atomic publication primitives changed")

    decode_for = body[decode_i]
    validate_lines = [
        node.lineno
        for node in ast.walk(decode_for)
        if isinstance(node, ast.Call) and _call_name(node) == "_validate_target_array"
    ]
    if len(validate_lines) != 1 or validate_lines[0] <= decode_for.lineno:
        raise AssertionError("target decode validation is not inside post-iterator loop")
    return {
        "sha256": EXPECTED_SEALER_SHA256,
        "compiled_without_import_or_execution": True,
        "top_level_raw_imports": 0,
        "top_level_importlib_imports": 0,
        "preclaim_loader_actions": 0,
        "control_order": [
            "fsynced_o_excl_claim",
            "local_importlib_import",
            "raw_module_import",
            "raw_iterator_symbol_resolution",
            "raw_iterator_construction",
            "raw_target_decode_loop",
        ],
        "line_evidence": {
            "claim": body[claim_i].lineno,
            "local_importlib": body[import_i].lineno,
            "raw_module_import": body[module_i].lineno,
            "raw_iterator_symbol_resolution": body[factory_i].lineno,
            "raw_iterator_construction": body[iterator_i].lineno,
            "decode_loop": body[decode_i].lineno,
            "first_target_validation": validate_lines[0],
        },
        "persisted_roles": ["observer_fit", "observer_selection"],
        "withheld_roles": [
            "observer_audit",
            "observer_lockbox",
            "retired_teacher_diagnostics",
        ],
        "fit_predict_score_calls": [],
        "target_values_decoded": 0,
    }


GUARDED_WRAPPER = r'''
import importlib.abc
import json
import runpy
import sys

target = sys.argv[1]
arguments = json.loads(sys.argv[2])
expect_refusal = sys.argv[3] == "refusal"

class RawImportBlocker(importlib.abc.MetaPathFinder):
    attempted = False
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "authorized_full_targets" or fullname.startswith("authorized_full_targets."):
            type(self).attempted = True
            raise RuntimeError("BLOCKED_PRECLAIM_RAW_IMPORT:" + fullname)
        return None

sys.meta_path.insert(0, RawImportBlocker())
sys.argv = [target, *arguments]
try:
    runpy.run_path(target, run_name="__main__")
except SystemExit as exc:
    if not expect_refusal:
        raise
    if RawImportBlocker.attempted:
        raise RuntimeError("raw import attempted during refused invocation")
    print(json.dumps({
        "wrapper_status": "EXPECTED_REFUSAL_WITH_RAW_IMPORT_BLOCKER",
        "system_exit": str(exc),
        "raw_import_attempted": False,
    }, sort_keys=True))
else:
    if expect_refusal:
        raise RuntimeError("invocation unexpectedly did not refuse")
    if RawImportBlocker.attempted:
        raise RuntimeError("raw import attempted during preclaim proof")
'''


def _assert_runtime_paths_absent() -> dict[str, dict[str, Any]]:
    result = {}
    for role, path in RUNTIME_PATHS.items():
        exists = path.exists() or path.is_symlink()
        if exists:
            raise AssertionError(f"production path exists during v2 audit: {role}")
        result[role] = {"path": str(path), "exists": False}
    return result


def _guarded_subprocess(arguments: list[str], mode: str) -> dict[str, Any]:
    if not PROOF_PYTHON.is_file():
        raise FileNotFoundError(f"isolated data-gate Python is absent: {PROOF_PYTHON}")
    before = _assert_runtime_paths_absent()
    environment = os.environ.copy()
    environment.pop("WHEST_PHASE2_CUSTODIAN_BOUNDARY", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            str(PROOF_PYTHON),
            "-I",
            "-c",
            GUARDED_WRAPPER,
            str(SEALER_PATH),
            json.dumps(arguments),
            mode,
        ],
        cwd=HERE,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    after = _assert_runtime_paths_absent()
    if before != after or completed.returncode != 0:
        raise AssertionError(
            "guarded runtime proof failed: "
            f"returncode={completed.returncode} stderr={completed.stderr!r}"
        )
    payload = json.loads(completed.stdout)
    if mode == "refusal":
        if (
            payload.get("wrapper_status")
            != "EXPECTED_REFUSAL_WITH_RAW_IMPORT_BLOCKER"
            or payload.get("raw_import_attempted") is not False
            or "no executable preflight mode" not in payload.get("system_exit", "")
        ):
            raise AssertionError("guarded refused-invocation proof changed")
    else:
        if payload != {
            "status": "PASS_PRECLAIM_RAW_MODULE_NOT_IMPORTED",
            "raw_target_module": RAW_MODULE,
            "raw_target_module_imported": False,
            "production_paths_absent": True,
            "claim_created": False,
            "target_values_decoded": 0,
        }:
            raise AssertionError("guarded explicit preclaim proof changed")
    return {
        "mode": mode,
        "arguments": arguments,
        "raw_import_blocker_active": True,
        "raw_import_attempted": False,
        "returncode": completed.returncode,
        "production_paths_unchanged_and_absent": True,
        "payload": payload,
        "target_values_decoded": 0,
    }


def verify() -> dict[str, Any]:
    preserved = {}
    for name, expected in PRESERVED_SHA256.items():
        actual = _sha256(HERE / name)
        if actual != expected:
            raise AssertionError(f"pre-v2 frozen artifact changed: {name}")
        preserved[name] = actual
    runtime_before = _assert_runtime_paths_absent()
    source = _verify_source_order()
    refused = _guarded_subprocess([], "refusal")
    explicit = _guarded_subprocess(
        ["--prove-preclaim-no-raw-import"], "preclaim-proof"
    )
    runtime_after = _assert_runtime_paths_absent()
    if runtime_before != runtime_after:
        raise AssertionError("runtime paths changed during target-free v2 audit")
    return {
        "schema": "whest-phase2-custodian-sealer-static-audit-v2",
        "status": "PASS_V2_CLAIM_BEFORE_IMPORT_SEALER__NOT_EXECUTED",
        "sealer": source,
        "runtime_proofs": {
            "refused_invocation": refused,
            "explicit_preclaim_probe": explicit,
        },
        "roster": _verify_roster_target_free(),
        "raw_physical_constraint": _verify_physical_constraint_target_free(),
        "preserved_file_sha256": preserved,
        "runtime_paths": runtime_after,
        "production_claim_created": False,
        "production_release_created": False,
        "sealer_executed": False,
        "target_artifacts_created": 0,
        "fit_or_score_performed": False,
        "target_values_decoded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = verify()
    report["verifier_source_sha256"] = _sha256(Path(__file__).resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
        if temporary.exists() or args.receipt.exists():
            raise FileExistsError("v2 static receipt path already exists")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.receipt)
        descriptor = os.open(args.receipt.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    print(encoded, end="")


if __name__ == "__main__":
    main()
