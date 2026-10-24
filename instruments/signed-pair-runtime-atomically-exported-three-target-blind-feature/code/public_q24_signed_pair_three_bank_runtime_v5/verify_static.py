#!/usr/bin/env python3
"""Read-only static verifier for lean signed-pair three-bank runtime V5.

This verifier parses code and small public control documents only.  It never
imports FlopScope, NumPy, whestbench, parent/source/adapter numerical modules,
row scopes, manifests, weights, targets, or generated banks.  Its only dynamic
tests execute standard-library runner helpers on synthetic temporary files and
ordinary-import the token validator without a mint channel.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SOURCE_PACKAGE = EXPERIMENTS / "public_q24_signed_pair_three_bank_source_v1"
PARENT_PACKAGE = EXPERIMENTS / "public_q24_sync_minimized_d4_parent_authority_v1"
RUNTIME_V1 = EXPERIMENTS / "public_q24_signed_pair_three_bank_runtime_v1"
RUNTIME_V1_REVIEW = EXPERIMENTS / "public_q24_signed_pair_three_bank_runtime_review_v1"
RUNTIME_V2_REVIEW = EXPERIMENTS / "public_q24_signed_pair_three_bank_runtime_review_v2"
RUNTIME_V4 = EXPERIMENTS / "public_q24_signed_pair_three_bank_runtime_v4"
INCIDENT_AUDIT = (
    EXPERIMENTS
    / "public_q24_signed_pair_lean_v4_flopscope_accumulator_incident_audit_v1"
    / "INDEPENDENT_INCIDENT_ROOT_CAUSE_V1.json"
)

TOKEN = HERE / "runtime_token.py"
PARENT = HERE / "parent_capture.py"
LITERAL = HERE / "signed_pair_literal.py"
REFERENCE = HERE / "signed_pair_reference.py"
ADAPTER = HERE / "capture_adapter.py"
RUNNER = HERE / "runtime_runner.py"
BUNDLE_MANIFEST = HERE / "CALLABLE_BUNDLE_MANIFEST.json"
MANIFEST_SCAN = HERE / "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json"
MANIFEST_FIXTURE = HERE / "ROW_MANIFEST_PREFLIGHT_FIXTURE_V4.json"
V3_TERMINAL_LINEAGE = HERE / "V3_TERMINAL_FAILURE_LINEAGE_V4.json"
V4_TERMINAL_LINEAGE = HERE / "V4_TERMINAL_FAILURE_LINEAGE_V5.json"
RESET_VERIFIER = HERE / "verify_reset.py"
RESET_VERIFICATION = HERE / "FLOPSCOPE_RESET_VERIFICATION_V5.json"

MANIFEST_SCAN_SHA256 = "052842efefda1fd73e49d15f17256411436900393cbaf6ae61dac8098310ce0a"
MANIFEST_FIXTURE_SHA256 = "28e7c706c092fca59b419d88aff5893cb72cb17e5f3a2858341641de8222a6aa"
V3_TERMINAL_LINEAGE_SHA256 = "b0756fd217c6bef5a0df473502109d18bb948fe8b12d0c69c4f15ab02f27ea70"
V3_CLAIM_SHA256 = "bfa9e8d800a5b1efc0ec4e47a62b53d70176e0bf8b9ec73f9981056974abd4e9"
V3_FAILURE_SHA256 = "e64f998fc1513f906e90d2869340e44ce2ca548b91e31667aade69da9af1d50d"
V3_CONSUMED_CAPABILITY_SHA256 = "f7cd31ab124c7914d1bd2b413289fb84ced4a38d2e93e3f3d3d9a9e711d38a39"
V4_TERMINAL_LINEAGE_SHA256 = "0dfdfff19826c163958bfc0b6866b31a9f334a7a2f62262a447b5baa3f5d7ded"
V4_CLAIM_SHA256 = "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"
V4_FAILURE_SHA256 = "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"
V4_CONSUMED_CAPABILITY_SHA256 = "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"
INCIDENT_AUDIT_SHA256 = "060e7bab3094a7929d246f4bdf5fbe6790ccb3aa9c3b05734395a6376b24cd6c"
RESET_VERIFICATION_SHA256 = "9253d354914da94d193dadd43e118ef7b6afaa9a2b11769c8d599cbfd62b26af"

SOURCE_HASHES = {
    "INDEPENDENT_STATIC_REVIEW_V1.json": "e5bbea1e38a9c299ba7f708d7a2160c638473c86a3059970f516b535d26f1b4d",
    "PACKAGE_MANIFEST.json": "ab3bc3b700f29dedb27c073c9cf8db995f1f0f275c31cc23c162810b2fbd3e90",
    "FREEZE_VERIFICATION.json": "708889d5db98ba27c642f15488bfd2551765dfc266472f9bd22ce4d7415fa6ea",
    "FREEZE_RECEIPT.json": "5ae256185e0b6c635f4fe291a30f3cd89bd728d0e5bedd2ac7a853062a9e7a72",
    "signed_pair_three_bank_source.py": "aaf76a5aff890cca1b4f6a5816f9e73f99b015467598034aaf4178b698aad307",
}
PARENT_HASHES = {
    "estimator_sync_minimized.py": "c9603babb7b61ee4fc2ff8b3d743b9129bbcfd71eb55940c39aa2a84168aaba1",
    "SYNC_MINIMIZED_D4_AUTHORITY_V1.json": "8750590c9f4f49b374ce7f4969b1725dd0ff9dd0d7e29c037d3c38f27c2e9c79",
    "FREEZE_RECEIPT_V1.json": "fcf327baf5428001429d5695285e61aee761fe5a2b284c97ffd2939b7609697b",
    "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json": "d72429d3456069f3b841a47dcbc5930247d494a397d340da2b82feb26c21abad",
}
PRIOR_RUNTIME_V1_HASHES = {
    "parent_successor.py": "14c4ceccaeaef16220b5e57f9e26539b6a8bdb2d4c59ec867289f47e3d8f9afc",
    "verify_static.py": "82beecb8fe5f881d949a5c46a7befb054826a29b12ec9c267e1469d27cc89bf7",
    "STATIC_VERIFICATION.json": "ad555c10f6a50126ec5ace7fba9ff4ec5a97f8ef403755b16cbf196b313f168d",
}
PRIOR_RUNTIME_V1_HOLD_SHA256 = (
    "49e51e300c158229f8401a5df032feebc6ae5cc8ae48ab1f2e55b2a407646d32"
)
PRIOR_RUNTIME_V2_GO_SHA256 = (
    "7bf6664b2e27d863e5fc4a556373a217d05628ec681141757a4d0beec46c0b03"
)
CALLABLE_NAMES = (
    "runtime_token.py",
    "parent_capture.py",
    "signed_pair_literal.py",
    "signed_pair_reference.py",
    "capture_adapter.py",
)
PRESERVED_V4_BYTES = CALLABLE_NAMES + (
    "CALLABLE_BUNDLE_MANIFEST.json",
    "EXECUTION_STACK_MANIFEST.json",
    "ATOMIC_BUNDLE_CONTRACT.json",
    "ORDERED_688_SCOPE_CONTRACT.json",
)
STATIC_DOCUMENTS = (
    "DESIGN.md",
    "PREREGISTRATION.json",
    "INDEPENDENT_REVIEW_CONTRACT.json",
    "INDEPENDENT_REVIEW_SCAFFOLD.json",
    "AUTHORITY_ANCHOR_SCAFFOLD.json",
    "SINGLE_USE_CAPABILITY_SCAFFOLD.json",
    "FINAL_EXECUTION_RELEASE_SCAFFOLD.json",
    "ORDERED_688_SCOPE_CONTRACT.json",
    "ATOMIC_BUNDLE_CONTRACT.json",
    "EXECUTION_STACK_CONTRACT.md",
    "MANIFEST_COMPATIBILITY_SCAN_RECEIPT_V4.json",
    "ROW_MANIFEST_PREFLIGHT_FIXTURE_V4.json",
    "V3_TERMINAL_FAILURE_LINEAGE_V4.json",
    "V4_TERMINAL_FAILURE_LINEAGE_V5.json",
    "FLOPSCOPE_RESET_VERIFICATION_V5.json",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def top_function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("missing top function " + name)


def top_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError("missing top class " + name)


def class_method(tree, class_name, method_name):
    parent = top_class(tree, class_name)
    for node in parent.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError("missing class method " + class_name + "." + method_name)


def assignment(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise AssertionError("missing assignment " + name)


def literal(tree, name):
    return ast.literal_eval(assignment(tree, name))


def body_after_docstring(function):
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


def is_guard(statement, argument=None):
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "require_runtime_token"
        and len(statement.value.args) == 1
    ):
        return False
    return argument is None or ast.unparse(statement.value.args[0]) == argument


def require_first_guard(function, argument=None):
    body = body_after_docstring(function)
    require(body and is_guard(body[0], argument), "token guard not first: " + function.name)


def top_imports(tree):
    result = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module)
    return result


def calls_named(root, name):
    return [
        node
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class RenameIdentifiers(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


def normalized_guarded_function(function, token_name):
    function = copy.deepcopy(function)
    body = body_after_docstring(function)
    removable = (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Call)
        and isinstance(body[0].value.func, ast.Name)
        and body[0].value.func.id in {"_require_release", "require_runtime_token"}
    )
    require(removable, "expected removable guard: " + function.name)
    function.body.remove(body[0])
    RenameIdentifiers({token_name: "authority_token"}).visit(function)
    ast.fix_missing_locations(function)
    return function


def validate_bindings():
    for name, expected in SOURCE_HASHES.items():
        require(sha256(SOURCE_PACKAGE / name) == expected, "source binding drift: " + name)
    for name, expected in PARENT_HASHES.items():
        require(sha256(PARENT_PACKAGE / name) == expected, "parent binding drift: " + name)
    for name, expected in PRIOR_RUNTIME_V1_HASHES.items():
        require(sha256(RUNTIME_V1 / name) == expected, "runtime V1 drift: " + name)
    hold = RUNTIME_V1_REVIEW / "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"
    require(sha256(hold) == PRIOR_RUNTIME_V1_HOLD_SHA256, "runtime V1 HOLD drift")
    require(read_json(hold).get("decision") == "HOLD", "runtime V1 HOLD verdict drift")
    prior_go = RUNTIME_V2_REVIEW / "INDEPENDENT_STATIC_REVIEW_V2.json"
    require(sha256(prior_go) == PRIOR_RUNTIME_V2_GO_SHA256, "runtime V2 GO drift")
    require(read_json(prior_go).get("status") == "GO", "runtime V2 GO verdict drift")
    return {
        "source": SOURCE_HASHES,
        "parent": PARENT_HASHES,
        "runtime_v1": PRIOR_RUNTIME_V1_HASHES,
        "runtime_v1_hold_sha256": PRIOR_RUNTIME_V1_HOLD_SHA256,
        "runtime_v2_go_sha256": PRIOR_RUNTIME_V2_GO_SHA256,
    }


def validate_v4_terminal_and_reset_repair(runner_tree, adapter_tree):
    preserved = {}
    for name in PRESERVED_V4_BYTES:
        current = sha256(HERE / name)
        prior = sha256(RUNTIME_V4 / name)
        require(current == prior, "V4 preserved byte drift: " + name)
        preserved[name] = current

    require(sha256(INCIDENT_AUDIT) == INCIDENT_AUDIT_SHA256, "incident audit drift")
    require(sha256(V4_TERMINAL_LINEAGE) == V4_TERMINAL_LINEAGE_SHA256, "V4 lineage drift")
    require(sha256(RESET_VERIFICATION) == RESET_VERIFICATION_SHA256, "reset proof drift")
    lineage = read_json(V4_TERMINAL_LINEAGE)
    require(lineage["claim_sha256"] == V4_CLAIM_SHA256, "V4 claim lineage")
    require(lineage["terminal_failure_sha256"] == V4_FAILURE_SHA256, "V4 failure lineage")
    require(
        lineage["consumed_capability_sha256"] == V4_CONSUMED_CAPABILITY_SHA256,
        "V4 consumed-capability lineage",
    )
    require(lineage["no_retry"] is True, "V4 no-retry lineage")
    require(
        lineage["independent_incident_root_cause_sha256"] == INCIDENT_AUDIT_SHA256,
        "V4 incident lineage",
    )
    incident = read_json(INCIDENT_AUDIT)
    require(
        incident["status"]
        == "PASS_ROOT_CAUSE_CONFIRMED__V5_PER_ROW_POST_COMMIT_RESET_REQUIRED",
        "incident verdict drift",
    )
    reset = read_json(RESET_VERIFICATION)
    require(
        reset["status"] == "PASS_STATIC_AND_SYNTHETIC_NO_REAL_EXECUTION",
        "reset verification verdict",
    )
    placement = reset["static_placement"]
    invariance = reset["synthetic_invariance"]
    adversarial = reset["synthetic_loop_adversaries"]
    require(placement["runner_reset_call_sites"] == 1, "reset site count")
    require(placement["adapter_reset_call_sites"] == 0, "adapter reset present")
    require(placement["reset_inside_meter"] is False, "reset inside meter")
    require(placement["reset_is_final_loop_statement"] is True, "reset not loop tail")
    require(invariance["completed_contexts"] == 1000, "synthetic context count")
    require(invariance["budget_reset_calls"] == 1000, "synthetic reset count")
    require(invariance["max_global_accumulator_records_before_reset"] == 1, "record bound")
    require(invariance["max_global_accumulator_records_after_reset"] == 0, "post-reset record bound")
    require(invariance["final_global_accumulator_records"] == 0, "final record count")
    require(invariance["detached_complete_flops_values"] == [6], "detached meter drift")
    require(invariance["context_complete_flops_after_reset_values"] == [6], "context meter drift")
    require(invariance["output_values_after_reset"] == [3.0], "synthetic output drift")
    require(
        invariance["memory_bounded_against_per_row_accumulator_growth"] is True,
        "diagnostic retention unbounded",
    )
    require(adversarial["failed_or_unvalidated_row_reset_calls"] == 0, "failed-row reset")
    require(adversarial["progress_boundary_rank7_reset_calls"] == 1, "progress-row reset")
    require(adversarial["reset_failure_stops_before_next_generate"] is True, "reset fail-open")

    runner_resets = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "flopscope.budget_reset"
    ]
    adapter_resets = [
        node
        for node in ast.walk(adapter_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "budget_reset"
    ]
    require(len(runner_resets) == 1 and not adapter_resets, "runtime reset AST drift")
    v4_runner_tree = ast.parse(
        (RUNTIME_V4 / "runtime_runner.py").read_text(encoding="utf-8")
    )
    preserved_runner_functions = (
        "_validate_row_result",
        "_validate_final_artifacts",
        "_validate_complete_cross_rung_prefixes",
        "_validate_exact_meter_vector_alignment",
        "_validate_row_A15_hash_alignment",
        "_write_receipt",
        "_publish_bundle",
    )
    for name in preserved_runner_functions:
        require(
            ast.dump(top_function(runner_tree, name), include_attributes=False)
            == ast.dump(top_function(v4_runner_tree, name), include_attributes=False),
            "V4 runner science/receipt function drift: " + name,
        )
    preserved_runner_assignments = (
        "EXPECTED_RANKS",
        "SENTINEL_RANKS",
        "EXPECTED_CAPTURE_LAYERS",
        "EXPECTED_H",
        "INTEGER_METER_NAMES",
        "FLOAT_METER_NAMES",
        "ROW_METER_REQUIRED_FIELDS",
        "ROW_RESULT_REQUIRED_FIELDS",
    )
    for name in preserved_runner_assignments:
        require(
            ast.dump(assignment(runner_tree, name), include_attributes=False)
            == ast.dump(assignment(v4_runner_tree, name), include_attributes=False),
            "V4 runner science/receipt assignment drift: " + name,
        )
    return {
        "preserved_v4_sha256": preserved,
        "v4_terminal_lineage_sha256": V4_TERMINAL_LINEAGE_SHA256,
        "v4_claim_sha256": V4_CLAIM_SHA256,
        "v4_failure_sha256": V4_FAILURE_SHA256,
        "v4_consumed_capability_sha256": V4_CONSUMED_CAPABILITY_SHA256,
        "independent_incident_root_cause_sha256": INCIDENT_AUDIT_SHA256,
        "reset_verification_sha256": RESET_VERIFICATION_SHA256,
        "one_reset_site_after_validated_completed_row": True,
        "no_reset_inside_meter": True,
        "generation_receipt_schema_and_keyset_preserved": True,
        "preserved_v4_runner_science_and_receipt_functions": len(
            preserved_runner_functions
        ),
        "preserved_v4_runner_science_and_receipt_assignments": len(
            preserved_runner_assignments
        ),
    }


def validate_bundle_manifest():
    manifest = read_json(BUNDLE_MANIFEST)
    require(manifest["schema"] == "public-q24-three-bank-callable-bundle-v4", "bundle schema")
    require(manifest["status"] == "FROZEN_IMMUTABLE_CALLABLE_CODE_ROOT", "bundle status")
    require(set(manifest["components"]) == set(CALLABLE_NAMES), "bundle components")
    for name in CALLABLE_NAMES:
        require(
            manifest["components"][name]["sha256"] == sha256(HERE / name),
            "bundle component hash drift: " + name,
        )
    require(manifest["source_lineage"] == {
        "accepted_law_sha256": SOURCE_HASHES["signed_pair_three_bank_source.py"],
        "freeze_receipt_sha256": SOURCE_HASHES["FREEZE_RECEIPT.json"],
        "freeze_verification_sha256": SOURCE_HASHES["FREEZE_VERIFICATION.json"],
        "independent_static_go_sha256": SOURCE_HASHES["INDEPENDENT_STATIC_REVIEW_V1.json"],
        "package_manifest_sha256": SOURCE_HASHES["PACKAGE_MANIFEST.json"],
    }, "bundle source lineage")
    require(manifest["parent_lineage"] == {
        "authority_sha256": PARENT_HASHES["SYNC_MINIMIZED_D4_AUTHORITY_V1.json"],
        "freeze_receipt_sha256": PARENT_HASHES["FREEZE_RECEIPT_V1.json"],
        "predecessor_sha256": PARENT_HASHES["estimator_sync_minimized.py"],
        "static_hold_sha256": PARENT_HASHES["INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"],
    }, "bundle parent lineage")
    require(
        manifest["prior_runtime_v1_hold_sha256"] == PRIOR_RUNTIME_V1_HOLD_SHA256,
        "bundle prior HOLD binding",
    )
    require(
        manifest["prior_runtime_v2_go_sha256"] == PRIOR_RUNTIME_V2_GO_SHA256,
        "bundle prior V2 GO binding",
    )
    expected_properties = {
        "base_capture_parent_comparator_is_dynamic_at_sentinels": True,
        "bundle_contains_transaction_hashes": False,
        "captured_parent_and_literal_source_execute_all_rows": True,
        "h256_prefix_derivation_executes_all_rows": True,
        "hostile_interpreter_security_claim": False,
        "literal_reference_source_execution_is_separate_at_sentinels": True,
        "ordinary_import_exposes_mint": False,
        "public_numerical_entrypoints_require_process_local_token": True,
        "resource_thresholds_can_veto_accuracy_generation": False,
        "sentinel_ranks_are_exactly_0_343_687": True,
        "manifest_firewall_allows_only_three_exact_zero_audit_paths": True,
        "manifest_root_schema_and_decoded_columns_are_exact": True,
        "v3_capability_is_terminally_consumed_and_not_retried": True,
    }
    require(manifest["properties"] == expected_properties, "bundle properties")
    require(
        manifest["manifest_compatibility_scan_receipt_sha256"] == MANIFEST_SCAN_SHA256,
        "bundle manifest-scan binding",
    )
    require(
        manifest["prior_runtime_v3_terminal_failure_lineage_sha256"]
        == V3_TERMINAL_LINEAGE_SHA256,
        "bundle V3 terminal-lineage binding",
    )
    return {"sha256": sha256(BUNDLE_MANIFEST), "components": manifest["components"]}


def validate_token(token_text, token_tree):
    imports = set(top_imports(token_tree))
    require(imports == {"__future__"}, "token gained non-future import")
    public_function_defs = [
        node.name for node in token_tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    require(not public_function_defs, "token exposes a public function definition")
    assigned_names = {
        child.id
        for node in token_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name)
    }
    require(
        {"require_runtime_token", "runtime_token_receipt_fields"} <= assigned_names,
        "token public API assignments missing",
    )
    require("_mint_runtime_token" not in token_text, "token exposes old mint")
    require("hostile interpreter" in token_text.lower(), "token threat boundary missing")

    module_name = "q24_v2_ordinary_direct_import_static_test"
    specification = importlib.util.spec_from_file_location(module_name, TOKEN)
    require(specification is not None and specification.loader is not None, "token spec")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    exposed_mint = [
        name for name in module.__dict__
        if "mint" in name.lower() and not name.startswith("__")
    ]
    require(not exposed_mint, "ordinary token import exposes mint")
    rejects = 0
    for fake in (None, object(), {"claim_sha256": "0" * 64}):
        try:
            module.require_runtime_token(fake)
        except module.RuntimeTokenRequired:
            rejects += 1
        else:
            raise AssertionError("ordinary token import accepted fake")
    return {
        "ordinary_import_exposed_mint_attributes": exposed_mint,
        "ordinary_fake_tokens_rejected": rejects,
        "nonserializable_token_implementation_present": "__reduce_ex__" in token_text,
        "hostile_interpreter_security_claim": False,
    }


def validate_literal(literal_tree):
    accepted_tree = ast.parse(
        (SOURCE_PACKAGE / "signed_pair_three_bank_source.py").read_text(encoding="utf-8")
    )
    constant_names = (
        "WIDTH", "CAPTURE_LAYERS", "SECTORS", "H16", "H64", "H256",
        "RECEIVERS_H16", "RECEIVERS_H64", "RECEIVERS_H256",
        "PAIR_SCALE_F32", "TAU_FLOOR_F32", "MGS_DOT_FLOOR_F64",
        "MGS_RMS_FLOOR_F64", "CAPTURE_LAYERS_ZERO_BASED",
        "ACTION_TAPE_SLOT_BY_LAYER_ZERO_BASED", "ZERO_ACTION_LAYER_ZERO_BASED",
        "PROBE_SELECTOR_BY_SECTOR", "MONOMIAL_EXPONENTS", "DEGREE_ONE_UNIT",
        "H16_PREFIX_INDEX_MAP",
    )
    for name in constant_names:
        require(
            ast.dump(assignment(literal_tree, name), include_attributes=False)
            == ast.dump(assignment(accepted_tree, name), include_attributes=False),
            "literal constant drift: " + name,
        )
    function_names = (
        "candidate_exponents_for_sector", "allocate_three_bank_workspace",
        "_build_monomial", "_orthogonalize_one_receiver",
        "_build_receiver_dictionary", "capture_one_layer", "bank_views",
    )
    for name in function_names:
        accepted = normalized_guarded_function(top_function(accepted_tree, name), "release_runner_sha256")
        current = normalized_guarded_function(top_function(literal_tree, name), "runtime_token")
        require(
            ast.dump(accepted, include_attributes=False) == ast.dump(current, include_attributes=False),
            "literal accepted-law AST drift: " + name,
        )
    require(
        literal(literal_tree, "LITERAL_IMPLEMENTATION_ID") == "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2",
        "literal implementation ID drift",
    )
    for name in function_names:
        require_first_guard(top_function(literal_tree, name), "runtime_token")
    return {
        "accepted_law_sha256": SOURCE_HASHES["signed_pair_three_bank_source.py"],
        "normalized_function_ast_equal": len(function_names),
        "call_ledger_inherited": {"direct_pre_dictionary_calls": 44, "calls_per_capture": 38050, "calls_per_row": 494678},
    }


def validate_reference(reference_text, reference_tree, literal_tree):
    shared_constants = (
        "WIDTH", "CAPTURE_LAYERS", "SECTORS", "H16", "H64", "H256",
        "RECEIVERS_H16", "RECEIVERS_H64", "RECEIVERS_H256",
        "PAIR_SCALE_F32", "TAU_FLOOR_F32", "MGS_DOT_FLOOR_F64",
        "MGS_RMS_FLOOR_F64", "CAPTURE_LAYERS_ZERO_BASED",
        "PROBE_SELECTOR_BY_SECTOR", "MONOMIAL_EXPONENTS", "DEGREE_ONE_UNIT",
    )
    for name in shared_constants:
        require(
            ast.dump(assignment(reference_tree, name), include_attributes=False)
            == ast.dump(assignment(literal_tree, name), include_attributes=False),
            "reference constant drift: " + name,
        )
    require("signed_pair_literal" not in reference_text, "reference imports/calls literal")
    require(sha256(REFERENCE) != sha256(LITERAL), "reference bytes alias literal")
    require(
        literal(reference_tree, "REFERENCE_IMPLEMENTATION_ID") == "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2",
        "reference implementation ID drift",
    )
    while_count = sum(isinstance(node, ast.While) for node in ast.walk(reference_tree))
    literal_while_count = sum(isinstance(node, ast.While) for node in ast.walk(literal_tree))
    require(while_count >= 7 and literal_while_count == 0, "reference loop structure not independent")
    require("receiver_index::RECEIVERS_H256" not in reference_text, "reference retained literal stride pack")
    require("Independent sector-block packing" in reference_text, "reference sector pack missing")
    functions = [node for node in reference_tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        require_first_guard(function, "runtime_token")
    return {
        "sha256_distinct_from_literal": True,
        "imports_literal": False,
        "reference_while_nodes": while_count,
        "literal_while_nodes": literal_while_count,
        "sector_block_pack_distinct_from_receiver_stride_pack": True,
        "runtime_byte_comparator_required": True,
    }


def _top_assignments(tree):
    output = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                output[target.id] = node
    return output


def _rename_function(function, name, mapping=None, remove_guard=False):
    function = copy.deepcopy(function)
    function.name = name
    if remove_guard:
        body = body_after_docstring(function)
        require(body and is_guard(body[0]), "expected token guard: " + name)
        function.body.remove(body[0])
    if mapping:
        RenameIdentifiers(mapping).visit(function)
    ast.fix_missing_locations(function)
    return function


def validate_parent(parent_text, parent_tree):
    prior_path = RUNTIME_V1 / "parent_successor.py"
    prior_tree = ast.parse(prior_path.read_text(encoding="utf-8"))
    require(sha256(prior_path) == PRIOR_RUNTIME_V1_HASHES["parent_successor.py"], "prior parent drift")

    current_assignments = _top_assignments(parent_tree)
    prior_assignments = _top_assignments(prior_tree)
    shared_assignments = set(current_assignments) & set(prior_assignments)
    for name in shared_assignments:
        require(
            ast.dump(current_assignments[name], include_attributes=False)
            == ast.dump(prior_assignments[name], include_attributes=False),
            "parent frozen assignment drift: " + name,
        )

    changed = {
        "_predict",
        "predict_with_native_tape",
        "_predict_with_native_tape",
        "materialize_parameters",
        "_materialize_parameters",
        "transport_correction",
        "_transport_correction",
        "predict_with_signed_pair_capture",
        "_predict_corrected",
    }
    current_functions = {
        node.name: node for node in parent_tree.body if isinstance(node, ast.FunctionDef)
    }
    prior_functions = {
        node.name: node for node in prior_tree.body if isinstance(node, ast.FunctionDef)
    }
    unchanged = (set(current_functions) & set(prior_functions)) - changed
    for name in unchanged:
        require(
            ast.dump(current_functions[name], include_attributes=False)
            == ast.dump(prior_functions[name], include_attributes=False),
            "parent frozen helper drift: " + name,
        )

    current_predict = _rename_function(
        current_functions["_predict"],
        "_predict",
        {"runtime_token": "release_runner_sha256"},
        remove_guard=True,
    )
    prior_predict = copy.deepcopy(prior_functions["_predict"])
    require(
        ast.dump(current_predict, include_attributes=False)
        == ast.dump(prior_predict, include_attributes=False),
        "parent recurrence/capture-hook AST drift from reviewed V1 successor",
    )

    guarded_functions = (
        "_predict",
        "_predict_with_native_tape",
        "_materialize_parameters",
        "_transport_correction",
        "_five_components",
        "predict_base_five_components",
        "predict_with_signed_pair_capture",
        "_predict_corrected",
    )
    for name in guarded_functions:
        require_first_guard(current_functions[name], "runtime_token")
    require_first_guard(class_method(parent_tree, "Estimator", "predict"), "runtime_token")

    prediction = current_functions["_predict"]
    hooks = [
        node for node in ast.walk(prediction)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "signed_pair_source is not None and layer_index >= 2"
    ]
    require(len(hooks) == 1, "parent capture-hook count drift")
    hook_call = hooks[0].body[0].value
    require(
        ast.unparse(hook_call.func) == "signed_pair_source.capture_one_layer",
        "parent capture-hook target drift",
    )
    expected_args = [
        "fnp", "layer_index", "covariance", "k21_target", "k4_target",
        "k31_target", "k22_target", "repaired_variance", "alpha",
        "signed_pair_workspace", "runtime_token",
    ]
    require([ast.unparse(value) for value in hook_call.args] == expected_args, "parent capture args drift")
    loop = next(
        node for node in ast.walk(prediction)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "layer_index"
        and ast.unparse(node.iter) == "range(1, mlp.depth)"
    )
    reset_index = next(
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_full_relu_reset"
    )
    hook_index = loop.body.index(hooks[0])
    row_index = next(
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "rows.append"
    )
    gate_index = next(
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "gate" for target in node.targets)
    )
    require(reset_index < hook_index < row_index < gate_index, "parent capture seam order drift")
    sync_count = sum(
        isinstance(node, ast.Call) and ast.unparse(node) == "float(fnp.min(variance))"
        for node in ast.walk(parent_tree)
    )
    require(sync_count == 1, "parent law-sync source-site count drift")
    for token in ("fit(", "target.npy", "ground_truth", "leaderboard", "submit("):
        require(token not in parent_text.lower(), "parent forbidden surface: " + token)
    return {
        "prior_reviewed_successor_sha256": sha256(prior_path),
        "shared_frozen_assignments_equal": len(shared_assignments),
        "shared_frozen_helpers_equal": len(unchanged),
        "full_recurrence_and_hook_ast_equal_after_token_rename": True,
        "capture_hook_count": 1,
        "capture_layers": list(range(2, 15)),
        "hook_order": "full_reset_then_capture_then_rows_then_reset_stars",
        "guarded_prediction_entrypoints": len(guarded_functions) + 1,
        "depth16_dynamic_law_sync_count": 14,
    }


def validate_manifest_firewall():
    """Execute metadata-only V4 firewall tests on the exact real-row fixture."""
    require(sha256(MANIFEST_SCAN) == MANIFEST_SCAN_SHA256, "manifest scan drift")
    require(sha256(MANIFEST_FIXTURE) == MANIFEST_FIXTURE_SHA256, "manifest fixture drift")
    require(
        sha256(V3_TERMINAL_LINEAGE) == V3_TERMINAL_LINEAGE_SHA256,
        "V3 terminal lineage drift",
    )
    scan = read_json(MANIFEST_SCAN)
    fixture = read_json(MANIFEST_FIXTURE)
    lineage = read_json(V3_TERMINAL_LINEAGE)
    require(scan["row_count"] == 688, "manifest scan row count")
    require(scan["rank0_manifest_sha256"] == MANIFEST_FIXTURE_SHA256, "fixture binding")
    require(scan["root_keys"] == sorted(fixture), "manifest scan root-keyset drift")
    require(scan["manifest_schema"] == fixture["schema"], "manifest scan schema drift")
    require(scan["decoded_columns"] == fixture["decoded_columns"], "decoded-column drift")
    require(lineage["claim_sha256"] == V3_CLAIM_SHA256, "V3 claim lineage")
    require(lineage["terminal_failure_sha256"] == V3_FAILURE_SHA256, "V3 failure lineage")
    require(
        lineage["consumed_capability_sha256"] == V3_CONSUMED_CAPABILITY_SHA256,
        "V3 consumed capability lineage",
    )
    require(lineage["no_retry"] is True, "V3 no-retry lineage")

    adapter_tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    required_assignments = {
        "EXPECTED_ROW_MANIFEST_SCHEMA",
        "EXPECTED_ROW_MANIFEST_ROOT_KEYS",
        "EXPECTED_DECODED_COLUMNS",
        "FORBIDDEN_MANIFEST_KEY_TOKENS",
        "ALLOWED_ZERO_AUDIT_PATHS",
    }
    selected = []
    for node in adapter_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in required_assignments
            for target in node.targets
        ):
            selected.append(copy.deepcopy(node))
        elif isinstance(node, ast.ClassDef) and node.name == "AdapterIntegrityError":
            selected.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_reject_forbidden_manifest",
            "_validate_manifest_metadata",
        }:
            selected.append(copy.deepcopy(node))
    firewall_namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(ADAPTER), "exec"),
        firewall_namespace,
    )
    require(required_assignments <= set(firewall_namespace), "firewall constants missing")
    require(
        list(firewall_namespace["EXPECTED_ROW_MANIFEST_ROOT_KEYS"]) == scan["root_keys"],
        "adapter root-keyset contract drift",
    )
    require(
        firewall_namespace["EXPECTED_ROW_MANIFEST_SCHEMA"] == scan["manifest_schema"],
        "adapter manifest schema contract drift",
    )
    require(
        list(firewall_namespace["EXPECTED_DECODED_COLUMNS"]) == scan["decoded_columns"],
        "adapter decoded-column contract drift",
    )
    require(
        list(firewall_namespace["ALLOWED_ZERO_AUDIT_PATHS"])
        == scan["allowed_zero_audit_paths"],
        "adapter audit-counter allowlist drift",
    )
    validate = firewall_namespace["_validate_manifest_metadata"]
    validate(fixture)
    passed = ["exact_real_rank0_manifest_fixture_accepted"]

    for name in ("scores", "target_columns_projected", "target_values_decoded"):
        for value, suffix in ((1, "nonzero"), (-1, "negative"), (True, "bool")):
            mutated = copy.deepcopy(fixture)
            mutated[name] = value
            passed.append(_expect_reject(
                lambda mutated=mutated: validate(mutated),
                name + "_" + suffix + "_rejected",
            ))
    for name in ("fits", "optimizers_constructed"):
        for value, suffix in ((1, "nonzero"), (True, "bool")):
            mutated = copy.deepcopy(fixture)
            mutated[name] = value
            passed.append(_expect_reject(
                lambda mutated=mutated: validate(mutated),
                name + "_" + suffix + "_rejected",
            ))

    adversaries = []
    mutated = copy.deepcopy(fixture)
    mutated["target_column_projected"] = 0
    adversaries.append((mutated, "renamed_target_counter_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["files"]["innocent"] = {"scores": 0}
    adversaries.append((mutated, "nested_scores_zero_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["files"]["innocent"] = {"board_metric": 0}
    adversaries.append((mutated, "nested_board_field_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["mse_proxy"] = 0
    adversaries.append((mutated, "root_mse_field_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["extra_metadata"] = 0
    adversaries.append((mutated, "extra_safe_root_key_rejected"))
    mutated = copy.deepcopy(fixture)
    del mutated["scores"]
    adversaries.append((mutated, "missing_required_root_key_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["decoded_columns"] = fixture["decoded_columns"] + ["target"]
    adversaries.append((mutated, "decoded_target_column_rejected"))
    mutated = copy.deepcopy(fixture)
    mutated["schema"] = "different-target-blind-schema"
    adversaries.append((mutated, "wrong_manifest_schema_rejected"))
    for mutated, label in adversaries:
        passed.append(_expect_reject(
            lambda mutated=mutated: validate(mutated),
            label,
        ))
    return {
        "manifest_scan_receipt_sha256": MANIFEST_SCAN_SHA256,
        "exact_real_fixture_sha256": MANIFEST_FIXTURE_SHA256,
        "v3_terminal_lineage_sha256": V3_TERMINAL_LINEAGE_SHA256,
        "positive_fixture_count": 1,
        "adversarial_rejections": passed[1:],
        "adversarial_rejection_count": len(passed) - 1,
        "numerical_modules_imported": 0,
        "weights_opened": 0,
        "targets_opened": 0,
    }


def validate_adapter(adapter_text, adapter_tree):
    allowed = {
        "__future__", "hashlib", "importlib", "json", "math", "os",
        "pathlib", "resource", "stat", "sys", "time",
        "q24_three_bank_runtime_token_v2",
    }
    require(set(top_imports(adapter_tree)) <= allowed, "adapter top-level import drift")
    for name in (
        "begin_target_blind_three_bank_pass",
        "generate_one_target_blind_row",
        "finish_target_blind_three_bank_pass",
        "_validate_record_and_load_mlp",
    ):
        require_first_guard(top_function(adapter_tree, name), "runtime_token")
    require(
        ast.unparse(assignment(adapter_tree, "DIAGNOSTIC_METER_CEILING"))
        == "(1 << 63) - 1",
        "diagnostic ceiling drift",
    )
    fixed_bound = literal(adapter_tree, "FIXED_ACCURACY_ROW_FLOP_UPPER_BOUND")
    require(fixed_bound == 4_000_000_000_000, "fixed-row upper bound drift")
    require((1 << 63) - 1 > 1_000_000 * fixed_bound, "diagnostic margin insufficient")
    generate = top_function(adapter_tree, "generate_one_target_blind_row")
    contexts = [
        node for node in ast.walk(generate)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "flops.BudgetContext"
    ]
    require(len(contexts) == 1, "adapter BudgetContext call count drift")
    require(
        {item.arg for item in contexts[0].keywords} == {"flop_budget", "quiet", "namespace"},
        "adapter BudgetContext keyword drift",
    )
    require("wall_time_limit_s" not in adapter_text, "adapter uses unsupported wall deadline")
    require(literal(adapter_tree, "SENTINEL_RANKS") == (0, 343, 687), "sentinel ranks drift")
    required_tokens = (
        "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE",
        "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE",
        "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2",
        "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2",
        "RESOURCE_AND_RESIDUAL_OBSERVATIONAL_DEPLOYMENT_ONLY",
        "Accuracy bytes are copied before deployment-only resource classification",
        "literal/reference action bytes differ: H",
        "if not _bytes_equal(np, base[name], captured[name])",
        "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256",
        'sentinel = expected_rank in SENTINEL_RANKS',
        'if sentinel:',
        'reference_source=(session["reference_source"] if sentinel else None)',
        '"source_action_comparator": source_comparator',
        '"five_component_parent_comparator": parent_comparator',
        '"sentinel_derived_matches_literal_and_reference"',
        'MEMMAP_FLUSH_INTERVAL_ROWS = 8',
        "accuracy_veto_allowed\": False",
        "allowed audit counter must be exact built-in int zero",
        "row manifest root keyset mismatch",
        "row manifest decoded columns mismatch",
    )
    for token in required_tokens:
        require(token in adapter_text, "adapter comparator/resource token missing: " + token)
    resource_name_loads = [
        node for node in ast.walk(generate)
        if isinstance(node, ast.Name)
        and node.id == "resource_observation"
        and isinstance(node.ctx, ast.Load)
    ]
    require(len(resource_name_loads) == 1, "resource observation controls accuracy flow")
    for node in ast.walk(generate):
        if isinstance(node, (ast.If, ast.While, ast.Assert)):
            require(
                "complete_flops_within_deployment_cap" not in ast.unparse(node.test),
                "deployment cap became an accuracy branch",
            )
    forbidden = ("target.npy", "targets.npy", "ground_truth.npy", "leaderboard_api", "submit(")
    require(not any(token in adapter_text.lower() for token in forbidden), "adapter forbidden surface")
    return {
        "top_level_heavy_imports": 0,
        "token_dominated_effectful_entrypoints": 4,
        "official_BudgetContext_keywords": ["flop_budget", "quiet", "namespace"],
        "diagnostic_meter_ceiling": (1 << 63) - 1,
        "fixed_accuracy_row_flop_upper_bound": fixed_bound,
        "ceiling_to_bound_ratio_gt_one_million": True,
        "all_row_captured_parent_and_literal_source": True,
        "sentinel_ranks": [0, 343, 687],
        "literal_reference_dynamic_bytes_compared_at_sentinels": True,
        "base_capture_five_components_dynamic_bytes_compared_at_sentinels": True,
        "h256_prefix_derivation_all_rows": True,
        "non_sentinel_duplicate_source_and_parent_passes": 0,
        "memmap_flush_interval_rows": 8,
        "resource_thresholds_can_veto_accuracy_generation": False,
        "manifest_firewall_semantic_allowlist_exact": True,
    }


def validate_execution_stack():
    path = HERE / "EXECUTION_STACK_MANIFEST.json"
    manifest = read_json(path)
    expected_stack = literal(ast.parse(RUNNER.read_text(encoding="utf-8")), "EXECUTION_STACK_MANIFEST_SHA256")
    require(sha256(path) == expected_stack, "stack manifest drift")
    require(manifest["schema"] == "public-q24-three-bank-execution-stack-v3", "stack schema")
    require(manifest["status"] == "FROZEN_OFFICIAL_SHIPPING_LANE", "stack status")
    require(manifest["lane"] == "OFFICIAL_FULL_NATIVE_SHIPPING_STACK", "stack lane")
    interpreter = manifest["interpreter"]
    invocation = Path(interpreter["invocation_path"])
    prefix = Path(interpreter["venv_prefix"])
    require(".client-venv" not in str(invocation), "minimal client lane selected")
    require(invocation.is_symlink(), "frozen invocation is not symlink")
    require(os.readlink(invocation) == interpreter["invocation_symlink_target"], "interpreter link drift")
    require(sha256(invocation.resolve()) == interpreter["resolved_executable_sha256"], "interpreter hash drift")
    for relative, expected in manifest["module_files"].items():
        require(sha256(prefix / relative) == expected, "stack file drift: " + relative)
    lock = manifest["requirements_lock"]
    require(sha256(Path(lock["path"])) == lock["sha256"], "requirements lock drift")
    require(manifest["distribution_versions"] == {
        "flopscope": "0.12.0",
        "flopscope-server": "0.12.0",
        "numpy": "2.4.6",
        "whestbench": "0.16.0",
    }, "stack versions drift")
    contract = manifest["budget_context_contract"]
    require(contract == {
        "constructor_keywords": ["flop_budget", "quiet", "namespace"],
        "diagnostic_meter_ceiling": (1 << 63) - 1,
        "fixed_accuracy_row_flop_upper_bound": 4_000_000_000_000,
        "no_wall_deadline_by_omission": True,
        "remote_client_lane_allowed": False,
    }, "stack budget contract drift")
    budget_path = prefix / "lib/python3.11/site-packages/flopscope/_budget.py"
    budget_tree = ast.parse(budget_path.read_text(encoding="utf-8"))
    budget_class = top_class(budget_tree, "BudgetContext")
    constructor = next(
        node for node in budget_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameters = [item.arg for item in constructor.args.args + constructor.args.kwonlyargs]
    require(
        {"flop_budget", "quiet", "namespace"} <= set(parameters),
        "shipping BudgetContext lacks frozen keyword subset",
    )
    return {
        "manifest_sha256": sha256(path),
        "lane": manifest["lane"],
        "interpreter": interpreter["version"],
        "distributions": manifest["distribution_versions"],
        "bound_file_count": len(manifest["module_files"]) + 2,
        "BudgetContext_required_subset_present": True,
        "target_free_protocol_sanity_reported_by_root": True,
    }


def validate_runner(runner_text, runner_tree, bundle_sha256, stack_sha256):
    allowed = {
        "__future__", "argparse", "datetime", "hashlib", "importlib.util",
        "json", "math", "os", "pathlib", "platform", "stat", "sys",
    }
    require(set(top_imports(runner_tree)) <= allowed, "runner gained non-stdlib top import")
    require(literal(runner_tree, "CALLABLE_BUNDLE_MANIFEST_SHA256") == bundle_sha256, "runner bundle root drift")
    require(literal(runner_tree, "EXECUTION_STACK_MANIFEST_SHA256") == stack_sha256, "runner stack root drift")
    require(literal(runner_tree, "SENTINEL_RANKS") == (0, 343, 687), "runner sentinel ranks drift")
    capability_fields = set(literal(runner_tree, "CAPABILITY_REQUIRED_FIELDS"))
    require("authority_anchor_sha256" in capability_fields, "capability misses anchor")
    require("execution_stack_manifest_sha256" in capability_fields, "capability misses stack")
    require("final_release_sha256" not in capability_fields, "capability binds final release")
    release_fields = set(literal(runner_tree, "RELEASE_REQUIRED_FIELDS"))
    require({"authority_anchor_sha256", "single_use_capability_sha256"} <= release_fields, "release DAG edge drift")
    review_fields = set(literal(runner_tree, "INDEPENDENT_REVIEW_REQUIRED_FIELDS"))
    require("reviewed_execution_stack_manifest_sha256" in review_fields, "review misses stack")
    require(
        {
            "prior_runtime_v4_terminal_failure_lineage_sha256",
            "prior_runtime_v4_claim_sha256",
            "prior_runtime_v4_terminal_failure_sha256",
            "prior_runtime_v4_consumed_capability_sha256",
            "independent_incident_root_cause_sha256",
            "flopscope_reset_verification_sha256",
            "r6_flopscope_reset_placement_and_invariance_verified",
            "v4_terminal_no_retry_closed",
        }
        <= review_fields,
        "review misses V4 terminal/reset bindings",
    )

    main = top_function(runner_tree, "main")
    positions = {
        name: sorted(node.lineno for node in calls_named(main, name))
        for name in (
            "_create_claim_exclusive", "_consume_capability", "_load_token_and_mint",
            "_load_module", "_write_receipt", "_publish_bundle",
        )
    }
    require(len(positions["_create_claim_exclusive"]) == 1, "claim call count")
    require(len(positions["_consume_capability"]) == 1, "consume call count")
    require(len(positions["_load_token_and_mint"]) == 1, "token mint call count")
    require(len(positions["_load_module"]) == 4, "runtime module load count")
    require(
        positions["_create_claim_exclusive"][0]
        < positions["_consume_capability"][0]
        < positions["_load_token_and_mint"][0]
        < min(positions["_load_module"]),
        "claim/consume/token/import chronology drift",
    )
    require(
        len(positions["_write_receipt"]) == len(positions["_publish_bundle"]) == 1
        and positions["_write_receipt"][0] < positions["_publish_bundle"][0],
        "receipt/publication chronology drift",
    )
    required_tokens = (
        'claim_ownership = {"acquired": False}',
        'claim_ownership.get("acquired") is True',
        "os.O_WRONLY | os.O_CREAT | os.O_EXCL",
        "capability must never bind the final release",
        '"execution_stack_manifest_sha256": execution_stack_manifest_sha256',
        "atomic staging inventory is not exactly seven files",
        "_validate_exact_meter_vector_alignment",
        "_validate_row_A15_hash_alignment",
        "_validate_complete_cross_rung_prefixes",
        "fourteen_parent_law_synchronizations_allowed_for_accuracy_science",
        "sentinel_duplicate_timing_is_deployment_only",
        "literal_reference_bytes_equal_sentinels",
        "base_capture_five_components_equal_sentinels",
        "h256_prefix_derivation_complete_all_rows",
        "if completed_rows % 8 == 0 or completed_rows == len(EXPECTED_RANKS):",
        '"lean-bank-progress completed_rows="',
        "flush=True",
        '"hostile_interpreter_security_claim") is not False',
        'flopscope = session["flops"]',
        "flopscope.budget_reset()",
    )
    for token in required_tokens:
        require(token in runner_text, "runner invariant token missing: " + token)
    return {
        "top_level_imports_stdlib_only": True,
        "callable_bundle_sha256": bundle_sha256,
        "execution_stack_manifest_sha256": stack_sha256,
        "authority_graph": "bundle_and_stack_to_runner__review_to_anchor__anchor_to_capability__release_to_anchor_and_capability",
        "capability_binds_final_release": False,
        "acyclic": True,
        "claim_before_consume": True,
        "consume_before_token_mint": True,
        "token_before_numerical_imports": True,
        "numerical_dynamic_import_count": 4,
        "owner_only_terminal_failure": True,
        "atomic_bundle_file_count": 7,
        "hostile_interpreter_security_claim": False,
        "one_unconditional_row_loop_tail_budget_reset_site": True,
    }


def runtime_mock_namespace(runner_tree):
    tree = copy.deepcopy(runner_tree)
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "__name__ == '__main__'"
        )
    ]
    ast.fix_missing_locations(tree)
    namespace = {
        "__name__": "q24_runtime_v5_static_transaction_mock",
        "__file__": str(RUNNER),
    }
    exec(compile(tree, str(RUNNER), "exec"), namespace)
    return namespace


def _expect_reject(callable_, label):
    try:
        callable_()
    except Exception:
        return label
    raise AssertionError("adversarial fixture unexpectedly accepted: " + label)


def _synthetic_row(namespace, rank=0):
    sentinel = rank in namespace["SENTINEL_RANKS"]
    digest = "a" * 64
    other = "b" * 64
    component_sha256 = {
        "runtime_token.py": "1" * 64,
        "parent_capture.py": "2" * 64,
        "signed_pair_literal.py": "3" * 64,
        "signed_pair_reference.py": "4" * 64,
        "capture_adapter.py": "5" * 64,
    }
    pair_hashes = {
        str(h): {"action_sha256": digest, "A15_zero_sha256": other}
        for h in (16, 64, 256)
    }
    source = {
        "comparator": "SEPARATE_LITERAL_AND_REFERENCE_EXECUTIONS_BYTEWISE",
        "literal_implementation_id": "ACCEPTED_LAW_RECEIVER_STRIDE_LITERAL_V2",
        "reference_implementation_id": "INDEPENDENT_LOOP_AND_SECTOR_PACK_REFERENCE_V2",
        "literal_source_sha256": component_sha256["signed_pair_literal.py"],
        "reference_source_sha256": component_sha256["signed_pair_reference.py"],
        "literal": copy.deepcopy(pair_hashes),
        "reference": copy.deepcopy(pair_hashes),
        "all_action_and_A15_bytes_equal": True,
    }
    component_hashes = {name: digest for name in namespace["FIVE_COMPONENT_PARENT_NAMES"]}
    parent = {
        "comparator": "UNHOOKED_BASE_VS_CAPTURED_PARENT_BYTEWISE",
        "component_names": list(namespace["FIVE_COMPONENT_PARENT_NAMES"]),
        "base_component_sha256": dict(component_hashes),
        "capture_component_sha256": dict(component_hashes),
        "all_five_component_bytes_equal": True,
    }
    meters = {
        name: 1 for name in namespace["INTEGER_METER_NAMES"]
    }
    meters.update({name: 1.0 for name in namespace["FLOAT_METER_NAMES"]})
    meters["source_hook_capture_count"] = 13
    meters["reference_source_hook_capture_count"] = 13 if sentinel else 0
    if not sentinel:
        for name in (
            "base_parent_flops", "base_parent_calls",
            "reference_source_hook_flops", "reference_source_hook_calls",
            "reference_workspace_flops", "reference_workspace_calls",
        ):
            meters[name] = 0
    meters["meter_scope"] = "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS"
    banks = {
        str(h): {
            "shape": [13, 1024, h],
            "dtype": "float32",
            "finite": True,
            "action_bytes_sha256": digest,
            "layer_action_bytes_sha256": [digest] * 13,
            "A15_zero_bytes_sha256": other,
        }
        for h in (16, 64, 256)
    }
    record = {
        "rank": rank,
        "mlp_id": 17,
        "row_manifest_sha256": "6" * 64,
        "weights_sha256": "7" * 64,
        "weights_schema_sha256": "8" * 64,
    }
    result = {
        **record,
        "sentinel_rank": sentinel,
        "comparator_mode": (
            "SENTINEL_DUAL_PARITY" if sentinel else "LEAN_CAPTURE_LITERAL_ONLY"
        ),
        "capture_layers_zero_based": list(range(2, 15)),
        "same_layer_preactivation_alpha": True,
        "capture_after_full_reset_before_reset_stars": True,
        "A15_exact_zero_float32": True,
        "cross_rung_H16_prefix_byte_identical": True,
        "h256_prefix_derivation": {
            "method": "EIGHT_SECTOR_RECEIVER_PREFIX_FROM_H256",
            "source_h": 256,
            "derived_h": [16, 64],
            "all_derived_bytes_match_written_row": True,
            "sentinel_derived_matches_literal_and_reference": (
                True if sentinel else None
            ),
        },
        "source_action_comparator": source if sentinel else None,
        "five_component_parent_comparator": parent if sentinel else None,
        "banks": banks,
        "exact_meters": meters,
        "resource_observation": {
            "classification": "OBSERVATIONAL_DEPLOYMENT_ONLY",
            "deployment_flop_cap": 2_199_023_255_552,
            "complete_flops_within_deployment_cap": False,
            "peak_rss_bytes": 1,
            "wall_seconds": 1.0,
            "residual_seconds": 1.0,
            "accuracy_veto_allowed": False,
        },
    }
    return record, result, component_sha256


def adversarial_integrity_mocks(runner_tree):
    namespace = runtime_mock_namespace(runner_tree)
    passed = []
    record, result, components = _synthetic_row(namespace)
    namespace["_validate_row_result"](result, record, components)
    passed.append("sentinel_row_validates_with_dynamic_parity")
    plain_record, plain_result, plain_components = _synthetic_row(namespace, rank=1)
    namespace["_validate_row_result"](plain_result, plain_record, plain_components)
    passed.append("non_sentinel_row_validates_without_duplicate_work")

    mutated = copy.deepcopy(result)
    mutated["source_action_comparator"]["literal_source_sha256"] = "0" * 64
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "wrong_literal_source_binding_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["source_action_comparator"]["reference"]["16"]["action_sha256"] = "0" * 64
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "literal_reference_action_mismatch_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["five_component_parent_comparator"]["base_component_sha256"]["host"] = True
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "truthy_nonhash_parent_component_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["five_component_parent_comparator"]["capture_component_sha256"]["host"] = "0" * 64
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "base_capture_component_mismatch_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["exact_meters"]["wall_seconds"] = float("nan")
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "nonfinite_exact_meter_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["resource_observation"]["accuracy_veto_allowed"] = True
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "resource_accuracy_veto_flag_rejected",
    ))
    mutated = copy.deepcopy(plain_result)
    mutated["source_action_comparator"] = copy.deepcopy(result["source_action_comparator"])
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, plain_record, plain_components),
        "non_sentinel_comparator_overclaim_rejected",
    ))
    mutated = copy.deepcopy(plain_result)
    mutated["exact_meters"]["base_parent_calls"] = 1
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, plain_record, plain_components),
        "non_sentinel_duplicate_parent_meter_rejected",
    ))
    mutated = copy.deepcopy(result)
    mutated["sentinel_rank"] = False
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_result"](mutated, record, components),
        "sentinel_identity_mutation_rejected",
    ))

    rows = []
    A15_DIGESTS = {16: "1" * 64, 64: "4" * 64, 256: "f" * 64}
    for rank in range(688):
        meters = {
            name: rank + index
            for index, name in enumerate(namespace["INTEGER_METER_NAMES"])
        }
        meters["source_hook_capture_count"] = 13
        meters["reference_source_hook_capture_count"] = 13 if rank in (0, 343, 687) else 0
        meters.update({
            name: float(rank + index + 1)
            for index, name in enumerate(namespace["FLOAT_METER_NAMES"])
        })
        meters["meter_scope"] = "CAPTURE_LITERAL_ALL_ROWS_BASE_REFERENCE_SENTINELS"
        rows.append({
            "rank": rank,
            "exact_meters": meters,
            "banks": {
                str(h): {"A15_zero_bytes_sha256": A15_DIGESTS[h]}
                for h in (16, 64, 256)
            },
        })
    vectors = {
        vector: [rows[rank]["exact_meters"][scalar] for rank in range(688)]
        for vector, scalar in namespace["EXACT_METER_VECTOR_SCALAR_PAIRS"]
    }
    namespace["_validate_exact_meter_vector_alignment"](vectors, rows)
    passed.append("exact_meter_vectors_align_rank_by_rank")
    wrong_vectors = copy.deepcopy(vectors)
    wrong_vectors["complete_flops_by_rank"][0], wrong_vectors["complete_flops_by_rank"][1] = (
        wrong_vectors["complete_flops_by_rank"][1], wrong_vectors["complete_flops_by_rank"][0]
    )
    passed.append(_expect_reject(
        lambda: namespace["_validate_exact_meter_vector_alignment"](wrong_vectors, rows),
        "reordered_meter_vector_rejected",
    ))
    final = {
        "banks": {
            str(h): {"A15_zero_sha256": A15_DIGESTS[h]}
            for h in (16, 64, 256)
        }
    }
    namespace["_validate_row_A15_hash_alignment"](final, rows)
    passed.append("all_2064_row_A15_hashes_align_to_canonical_files")
    wrong_rows = copy.deepcopy(rows)
    wrong_rows[2]["banks"]["64"]["A15_zero_bytes_sha256"] = "f" * 64
    passed.append(_expect_reject(
        lambda: namespace["_validate_row_A15_hash_alignment"](final, wrong_rows),
        "mismatched_row_A15_hash_rejected",
    ))
    return passed


def adversarial_transaction_mocks(runner_tree):
    namespace = runtime_mock_namespace(runner_tree)
    passed = []
    with tempfile.TemporaryDirectory(prefix="q24-runtime-v5-static-", dir="/private/tmp") as temporary:
        root = Path(temporary) / "race"
        root.mkdir()
        owner_state = {"acquired": False}
        paths, claim_sha256 = namespace["_create_claim_exclusive"](
            root, {"schema": "synthetic-claim", "status": "OWNER"}, owner_state
        )
        require(owner_state["acquired"] is True, "claim owner bit not set")
        before = sorted((item.name, item.read_bytes()) for item in root.iterdir())
        loser_state = {"acquired": False}
        passed.append(_expect_reject(
            lambda: namespace["_create_claim_exclusive"](
                root, {"schema": "synthetic-claim", "status": "LOSER"}, loser_state
            ),
            "lock_race_loser_rejected",
        ))
        after = sorted((item.name, item.read_bytes()) for item in root.iterdir())
        require(before == after and loser_state == {"acquired": False}, "lock-race loser mutated winner")
        passed.append("lock_race_loser_mutation_free")

        paths["claim"].write_bytes(b"{partial")
        namespace["_write_owned_failure"](
            paths, claim_sha256, RuntimeError("synthetic post-claim failure")
        )
        failure = json.loads(paths["failure"].read_text(encoding="ascii"))
        require(failure["claim_complete"] is False, "partial claim not terminally receipted")
        passed.append("post_claim_partial_owner_terminal_failure")

        cap_root = Path(temporary) / "cap"
        cap_root.mkdir()
        payload = b'{"synthetic":true}\n'
        capability = cap_root / "capability.json"
        capability.write_bytes(payload)
        alias = cap_root / "alias.json"
        os.link(capability, alias)
        passed.append(_expect_reject(
            lambda: namespace["_consume_capability"](capability, payload),
            "preexisting_capability_hardlink_rejected",
        ))
        require(capability.exists() and alias.exists(), "hardlink reject mutated capability")
        passed.append("hardlink_reject_mutation_free")

        clean_root = Path(temporary) / "clean-cap"
        clean_root.mkdir()
        clean = clean_root / "capability.json"
        clean.write_bytes(payload)
        consumed, inode = namespace["_consume_capability"](clean, payload)
        require(not clean.exists() and consumed.is_file(), "capability was not consumed")
        require(inode["initial_link_count"] == inode["consumed_link_count"] == 1, "one-use inode ledger")
        passed.append("single_inode_capability_consumption")

        publish_root = Path(temporary) / "publish"
        publish_root.mkdir()
        publish_paths, _ = namespace["_create_claim_exclusive"](
            publish_root, {"schema": "synthetic-claim", "status": "PUBLISH"}, {"acquired": False}
        )
        publish_paths["staging"].mkdir()
        names = {
            "actions_H16.f32le", "actions_H64.f32le", "actions_H256.f32le",
            "A15_zero_H16.f32le", "A15_zero_H64.f32le", "A15_zero_H256.f32le",
            "THREE_BANK_GENERATION_RECEIPT.json",
        }
        for name in names:
            (publish_paths["staging"] / name).write_bytes(b"synthetic")
        namespace["_publish_bundle"](publish_paths)
        require({item.name for item in publish_paths["bundle"].iterdir()} == names, "seven-file publish drift")
        passed.append("exact_seven_file_atomic_publish")

        bad_root = Path(temporary) / "bad-publish"
        bad_root.mkdir()
        bad_paths, _ = namespace["_create_claim_exclusive"](
            bad_root, {"schema": "synthetic-claim", "status": "BAD"}, {"acquired": False}
        )
        bad_paths["staging"].mkdir()
        for name in names | {"orphan.tmp"}:
            (bad_paths["staging"] / name).write_bytes(b"synthetic")
        passed.append(_expect_reject(lambda: namespace["_publish_bundle"](bad_paths), "orphan_file_rejected"))
        require(not bad_paths["bundle"].exists(), "orphan staging was published")
        passed.append("orphan_reject_prepublication")
    return passed


def authority_construction_mock(bundle_sha256, stack_sha256, runner_sha256):
    review = {
        "runner": runner_sha256,
        "bundle": bundle_sha256,
        "stack": stack_sha256,
        "verdict": "GO",
    }
    review_sha256 = hashlib.sha256(
        json.dumps(review, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    anchor = {
        "runner": runner_sha256,
        "bundle": bundle_sha256,
        "stack": stack_sha256,
        "review": review_sha256,
        "scope": "6" * 64,
    }
    anchor_bytes = json.dumps(anchor, sort_keys=True, separators=(",", ":")).encode("ascii")
    anchor_sha256 = hashlib.sha256(anchor_bytes).hexdigest()
    capability = {"authority_anchor_sha256": anchor_sha256, "one_use": True}
    capability_bytes = json.dumps(capability, sort_keys=True, separators=(",", ":")).encode("ascii")
    capability_sha256 = hashlib.sha256(capability_bytes).hexdigest()
    release = {
        "authority_anchor_sha256": anchor_sha256,
        "single_use_capability_sha256": capability_sha256,
    }
    release_sha256 = hashlib.sha256(
        json.dumps(release, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    require("final_release_sha256" not in capability, "capability/release cycle")
    require(release_sha256.encode("ascii") not in capability_bytes, "release hash leaked backward")
    return {
        "construction_order": [
            "callable_bundle_and_stack", "runner", "independent_review",
            "authority_anchor", "single_use_capability", "final_release",
        ],
        "all_dynamic_hashes_constructed": all(
            len(value) == 64
            for value in (review_sha256, anchor_sha256, capability_sha256, release_sha256)
        ),
        "capability_contains_final_release_hash": False,
        "acyclic": True,
    }


def validate_contracts(bundle_sha256, stack_sha256, runner_sha256):
    prereg = read_json(HERE / "PREREGISTRATION.json")
    review_contract = read_json(HERE / "INDEPENDENT_REVIEW_CONTRACT.json")
    review = read_json(HERE / "INDEPENDENT_REVIEW_SCAFFOLD.json")
    anchor = read_json(HERE / "AUTHORITY_ANCHOR_SCAFFOLD.json")
    capability = read_json(HERE / "SINGLE_USE_CAPABILITY_SCAFFOLD.json")
    release = read_json(HERE / "FINAL_EXECUTION_RELEASE_SCAFFOLD.json")
    scope = read_json(HERE / "ORDERED_688_SCOPE_CONTRACT.json")
    atomic = read_json(HERE / "ATOMIC_BUNDLE_CONTRACT.json")
    require(prereg["status"] == "STATIC_HOLD_NO_EXECUTION", "prereg unexpectedly live")
    require(prereg["callable_bundle_manifest_sha256"] == bundle_sha256, "prereg bundle binding")
    require(prereg["execution_stack_manifest_sha256"] == stack_sha256, "prereg stack binding")
    require(prereg["runner_sha256"] == runner_sha256, "prereg runner binding")
    require(prereg["schema"] == "public-q24-three-bank-runtime-preregistration-v5", "prereg schema")
    require(
        prereg["manifest_compatibility_scan_receipt_sha256"] == MANIFEST_SCAN_SHA256,
        "prereg manifest-scan binding",
    )
    require(
        prereg["prior_runtime_v3_terminal_failure_lineage_sha256"]
        == V3_TERMINAL_LINEAGE_SHA256,
        "prereg V3 terminal lineage binding",
    )
    require(
        prereg["prior_runtime_v4_terminal_failure_lineage_sha256"]
        == V4_TERMINAL_LINEAGE_SHA256,
        "prereg V4 terminal lineage binding",
    )
    require(
        prereg["independent_incident_root_cause_sha256"] == INCIDENT_AUDIT_SHA256,
        "prereg incident binding",
    )
    require(
        prereg["flopscope_reset"]["reset_verification_sha256"]
        == RESET_VERIFICATION_SHA256,
        "prereg reset proof binding",
    )
    require(
        review_contract["acceptance"]["hostile_interpreter_security_claim"] is False,
        "review contract threat overclaim",
    )
    require(review["status"] == "HOLD_NULL_REVIEW_SCAFFOLD", "review scaffold live")
    require(
        set(review) == set(literal(ast.parse(RUNNER.read_text(encoding="utf-8")), "INDEPENDENT_REVIEW_REQUIRED_FIELDS")),
        "review scaffold field inventory drift",
    )
    require(review["schema"] == "public-q24-three-bank-runtime-static-review-scaffold-v5", "review scaffold schema")
    require(review["reviewed_runner_sha256"] is None, "review scaffold binds runner")
    require(
        review["manifest_compatibility_scan_receipt_sha256"] == MANIFEST_SCAN_SHA256,
        "review scaffold manifest-scan binding",
    )
    require(anchor["status"] == "HOLD_NULL_AUTHORITY_ANCHOR", "anchor scaffold live")
    require(anchor["schema"] == "public-q24-three-bank-runtime-authority-anchor-scaffold-v5", "anchor scaffold schema")
    require(anchor["authority_anchor_sha256"] is None, "anchor scaffold hash live")
    require(capability["status"] == "HOLD_NULL_SINGLE_USE_CAPABILITY", "capability scaffold live")
    require(capability["schema"] == "public-q24-signed-pair-bank-capability-scaffold-v5", "capability scaffold schema")
    require("single_use_capability_sha256" not in capability, "capability scaffold self-hash field")
    require("final_release_sha256" not in capability, "capability has release back-edge")
    require(release["status"] == "HOLD_NULL_FINAL_EXECUTION_RELEASE", "release scaffold live")
    require(release["schema"] == "public-q24-three-bank-runtime-final-release-scaffold-v5", "release scaffold schema")
    require(release["final_release_sha256"] is None, "release scaffold hash live")
    require(scope["status"] == "HOLD_NULL_ORDERED_688_SCOPE", "scope scaffold live")
    require(scope["ordered_688_row_scope_sha256"] is None, "scope scaffold hash live")
    require(scope["ordered_688_record_sha256"] is None, "ordered-record scaffold hash live")
    require(len(atomic["staging_inventory"]) == 7, "atomic inventory count drift")
    require(atomic["publish_operation"] == "ONE_ATOMIC_DIRECTORY_RENAME", "atomic publish law drift")
    return {
        "review_null": True,
        "anchor_null": True,
        "capability_null": True,
        "release_null": True,
        "ordered_scope_null": True,
        "public_runner_reaches_claim_with_scaffolds": False,
        "scope_rows_opened": 0,
        "row_manifests_opened": 0,
        "weight_arrays_opened": 0,
        "target_arrays_opened": 0,
    }


def main():
    bindings = validate_bindings()
    token_text = TOKEN.read_text(encoding="utf-8")
    parent_text = PARENT.read_text(encoding="utf-8")
    literal_text = LITERAL.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    token_tree = ast.parse(token_text)
    parent_tree = ast.parse(parent_text)
    literal_tree = ast.parse(literal_text)
    reference_tree = ast.parse(reference_text)
    adapter_tree = ast.parse(adapter_text)
    runner_tree = ast.parse(runner_text)

    bundle = validate_bundle_manifest()
    stack = validate_execution_stack()
    runner_sha256 = sha256(RUNNER)
    transaction = adversarial_transaction_mocks(runner_tree)
    integrity = adversarial_integrity_mocks(runner_tree)
    manifest_firewall = validate_manifest_firewall()
    result = {
        "schema": "public-q24-signed-pair-three-bank-runtime-static-verification-v5",
        "status": "PASS_STATIC_HOLD_NO_EXECUTION",
        "lineage_bindings": bindings,
        "callable_bundle": bundle,
        "execution_stack": stack,
        "runtime_sha256": {
            name: sha256(HERE / name)
            for name in CALLABLE_NAMES + ("runtime_runner.py", "verify_static.py")
        },
        "token": validate_token(token_text, token_tree),
        "parent": validate_parent(parent_text, parent_tree),
        "literal_source": validate_literal(literal_tree),
        "reference_source": validate_reference(reference_text, reference_tree, literal_tree),
        "adapter": validate_adapter(adapter_text, adapter_tree),
        "v4_terminal_and_flopscope_reset_repair": (
            validate_v4_terminal_and_reset_repair(runner_tree, adapter_tree)
        ),
        "manifest_firewall": manifest_firewall,
        "runner": validate_runner(
            runner_text, runner_tree, bundle["sha256"], stack["manifest_sha256"]
        ),
        "adversarial_transaction_mocks": transaction,
        "adversarial_integrity_mocks": integrity,
        "adversarial_fixture_count": (
            len(transaction)
            + len(integrity)
            + manifest_firewall["adversarial_rejection_count"]
            + manifest_firewall["positive_fixture_count"]
        ),
        "authority_construction_mock": authority_construction_mock(
            bundle["sha256"], stack["manifest_sha256"], runner_sha256
        ),
        "contracts": validate_contracts(
            bundle["sha256"], stack["manifest_sha256"], runner_sha256
        ),
        "static_document_sha256": {
            name: sha256(HERE / name) for name in STATIC_DOCUMENTS
        },
        "access_counters": {
            "numerical_runtime_modules_imported": 0,
            "estimator_or_source_calls": 0,
            "ordered_scope_files_opened": 0,
            "row_manifests_opened": 0,
            "weight_arrays_opened": 0,
            "targets_opened": 0,
            "models_created": 0,
            "banks_created": 0,
            "claims_created_outside_temporary_mocks": 0,
            "capabilities_consumed_outside_temporary_mocks": 0,
            "training_runs": 0,
            "scores_computed": 0,
            "packages_built": 0,
            "submissions_made": 0,
        },
        "security_boundary": {
            "ordinary_direct_import_and_public_API_governance_fail_closed": True,
            "hostile_same_interpreter_arbitrary_source_or_introspection_in_scope": False,
            "hostile_interpreter_security_claim": False,
        },
        "verdict": "GO_REQUEST_INDEPENDENT_STATIC_REVIEW__EXECUTION_REMAINS_HOLD",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
