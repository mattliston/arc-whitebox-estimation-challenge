#!/usr/bin/env python3
"""Independent target-free static audit of the one-use custodian sealer.

This verifier deliberately does not import or execute the sealer or its raw
target dependency.  It reads source text, AST, target-free roster/receipts,
and file hashes only.  Its optional receipt is not a custodian claim or target
release.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SEALER_PATH = HERE / "custodian_seal_training_roles_once.py"
EXPECTED_SEALER_SHA256 = (
    "cad1ac6c72b374b04ec53bde8d7aac6112ac2ccbaa818c2f2a0195160a2edab4"
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
}
RUNTIME_PATHS = {
    "claim": HERE / "PHASE2_TRAINING_TARGET_SEALER_ONCE.CLAIM",
    "staging": HERE / "sealed_role_target_bundle.staging",
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
        if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            if (
                isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id in {"frozenset", "set", "tuple"}
                and len(statement.value.args) == 1
                and not statement.value.keywords
            ):
                return ast.literal_eval(statement.value.args[0])
            return ast.literal_eval(statement.value)
    raise AssertionError(f"missing literal assignment: {name}")


def _role_for_rank(rank: int) -> str:
    matches = [
        role for role, begin, end in EXPECTED_ROLE_SEGMENTS if begin <= rank < end
    ]
    if len(matches) != 1:
        raise AssertionError("independent role segments do not uniquely cover roster")
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
        raise AssertionError("roster attestation boundary changed")
    ranked: list[dict[str, Any]] = []
    for split, count in (("train", 700), ("calibration", 100), ("validation", 100), ("test", 100)):
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
        digest = hashlib.sha256(ASSIGNMENT_DOMAIN + str(seed).encode("ascii")).hexdigest()
        if row["assignment_sha256"] != digest:
            raise AssertionError("roster seed-only assignment digest changed")
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
    if payload["raw_role_projection_physically_safe"] is not False:
        raise AssertionError("raw role projection unexpectedly marked safe")
    collisions = payload["training_role_row_groups_colliding_with_audit_or_lockbox"]
    if len(collisions) != 62 or payload["row_groups"] != 63:
        raise AssertionError("raw row-group collision evidence changed")
    if (
        payload["network_independent_row_groups"] != 0
        or payload["role_pure_row_groups"] != 0
        or payload["all_layer_means_column_index_available_row_groups"] != 0
        or payload["all_layer_means_offset_index_available_row_groups"] != 0
    ):
        raise AssertionError("raw target addressability evidence changed")
    if payload["target_values_decoded"] != 0:
        raise AssertionError("addressability audit claims a target decode")
    return {
        "sha256": _sha256(path),
        "row_groups": 63,
        "training_protected_collisions": 62,
        "raw_role_projection_physically_safe": False,
        "target_values_decoded": 0,
    }


def _verify_source_static() -> dict[str, Any]:
    raw = SEALER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SEALER_SHA256:
        raise AssertionError("frozen custodian sealer source changed")
    source = raw.decode("utf-8")
    compile(source, str(SEALER_PATH), "exec")
    tree = ast.parse(source)

    persisted_roles = frozenset(_literal_assignment(tree, "PERSISTED_ROLES"))
    withheld_roles = frozenset(_literal_assignment(tree, "WITHHELD_ROLES"))
    if persisted_roles != {"observer_fit", "observer_selection"}:
        raise AssertionError("sealer persisted-role allow-list changed")
    if withheld_roles != {
        "observer_audit",
        "observer_lockbox",
        "retired_teacher_diagnostics",
    }:
        raise AssertionError("sealer withheld-role deny-list changed")
    if _literal_assignment(tree, "EXPECTED_ROLE_COUNTS") != EXPECTED_ROLE_COUNTS:
        raise AssertionError("sealer exact role counts changed")

    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_imports = imports & {
        "torch",
        "tensorflow",
        "jax",
        "sklearn",
        "flopscope",
        "whestbench",
    }
    if forbidden_imports:
        raise AssertionError(f"fit/score dependency entered sealer: {forbidden_imports}")

    execute = _function(tree, "execute")
    calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)]
    claim_calls = [node for node in calls if _call_name(node) == "_consume_fixed_claim"]
    iterator_calls = [
        node
        for node in calls
        if _call_name(node) == "iter_trusted_custodian_all_layer_means"
    ]
    if len(claim_calls) != 1 or len(iterator_calls) != 1:
        raise AssertionError("claim/raw-iterator call cardinality changed")
    if claim_calls[0].lineno >= iterator_calls[0].lineno:
        raise AssertionError("raw target iterator appears before fixed claim")

    claim_function = _function(tree, "_consume_fixed_claim")
    claim_attributes = {
        node.attr for node in ast.walk(claim_function) if isinstance(node, ast.Attribute)
    }
    exclusive_writer = _function(tree, "_write_exclusive_fsynced")
    exclusive_attributes = {
        node.attr for node in ast.walk(exclusive_writer) if isinstance(node, ast.Attribute)
    }
    if "O_EXCL" not in exclusive_attributes or "_write_exclusive_fsynced" not in {
        _call_name(node)
        for node in ast.walk(claim_function)
        if isinstance(node, ast.Call)
    }:
        raise AssertionError("fixed claim is not created through O_EXCL")

    destructive_names = {
        "unlink",
        "remove",
        "rmdir",
        "removedirs",
        "rmtree",
    }
    called_names = {_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    if called_names & destructive_names:
        raise AssertionError("claim cleanup/retry path entered sealer")
    if called_names & {"fit", "fit_transform", "predict", "score", "backward"}:
        raise AssertionError("fit/predict/score operation entered sealer")
    if "fsync" not in called_names or "rename" not in called_names or "replace" not in called_names:
        raise AssertionError("durable atomic output primitives are incomplete")

    writer_calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and _call_name(node) == "_write_role_artifact"
    ]
    writer_roles = []
    for node in writer_calls:
        if not node.args or not isinstance(node.args[0], ast.Constant):
            raise AssertionError("role artifact writer lacks a literal role")
        writer_roles.append(node.args[0].value)
    if writer_roles != ["observer_fit", "observer_selection"]:
        raise AssertionError("sealer writes a role other than fit and selection")
    parquet_write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "write_table"
    ]
    if len(parquet_write_calls) != 1:
        raise AssertionError("unexpected target-bearing Parquet writer count")
    if "crash_retry_allowed\": False" not in source:
        raise AssertionError("manifest no-retry attestation is absent")
    if "withheld_rows_written\": 0" not in source:
        raise AssertionError("manifest withheld-write zero attestation is absent")

    return {
        "sha256": EXPECTED_SEALER_SHA256,
        "compiled_without_import_or_execution": True,
        "fixed_claim_before_raw_iterator": True,
        "fixed_claim_uses_o_excl": "O_EXCL" in exclusive_attributes,
        "claim_cleanup_or_retry_calls": [],
        "persisted_roles": sorted(persisted_roles),
        "withheld_roles": sorted(withheld_roles),
        "target_artifact_writer_call_count": len(writer_calls),
        "fit_predict_score_calls": [],
        "atomic_output_primitives": ["fsync", "replace", "rename"],
        "target_values_decoded": 0,
    }


def verify() -> dict[str, Any]:
    preserved = {}
    for name, expected in PRESERVED_SHA256.items():
        actual = _sha256(HERE / name)
        if actual != expected:
            raise AssertionError(f"pre-sealer frozen artifact changed: {name}")
        preserved[name] = actual
    runtime_state = {}
    for role, path in RUNTIME_PATHS.items():
        exists = path.exists() or path.is_symlink()
        if exists:
            raise AssertionError(f"sealer must remain unexecuted; {role} path exists")
        runtime_state[role] = {"path": str(path), "exists": False}
    return {
        "schema": "whest-phase2-custodian-sealer-static-audit-v1",
        "status": "PASS_FROZEN_ONE_USE_CUSTODIAN_SEALER__NOT_EXECUTED",
        "sealer": _verify_source_static(),
        "roster": _verify_roster_target_free(),
        "raw_physical_constraint": _verify_physical_constraint_target_free(),
        "preserved_file_sha256": preserved,
        "runtime_paths": runtime_state,
        "claim_created": False,
        "target_release_created": False,
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
            raise FileExistsError("static audit receipt path already exists")
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
