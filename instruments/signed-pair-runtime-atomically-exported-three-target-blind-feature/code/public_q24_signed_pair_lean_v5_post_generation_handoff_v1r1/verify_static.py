#!/usr/bin/env python3
"""Static and synthetic verification for the V5 post-generation handoff V1R1."""

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


sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
P2 = HERE.parent
HELPER = HERE / "postseal_handoff.py"
COMMANDS = HERE / "COMMAND_PLAN.json"
CONTRACT = HERE / "CONTRACT.json"
STATIC = HERE / "STATIC_VERIFICATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helper():
    specification = importlib.util.spec_from_file_location(
        "q24_v5_postseal_handoff_static_test", HELPER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def function_calls(tree: ast.AST, name: str) -> set[str]:
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def call_counts(nodes: list[ast.stmt]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for statement in nodes:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            counts[name] = counts.get(name, 0) + 1
    return counts


def expect_handoff_error(helper, operation, label: str) -> None:
    try:
        operation()
    except helper.HandoffError:
        return
    raise AssertionError(f"adversarial fixture was accepted: {label}")


def atomic_gate_fixtures(helper) -> dict[str, bool]:
    original = {
        name: getattr(helper, name)
        for name in ("TRANSACTION_ROOT", "BUNDLE", "STAGING", "FAILURE", "CLAIM")
    }

    def bind(root: Path) -> None:
        helper.TRANSACTION_ROOT = root
        helper.BUNDLE = root / "THREE_BANK_BUNDLE"
        helper.STAGING = root / ".THREE_BANK_BUNDLE.staging"
        helper.FAILURE = root / "THREE_BANK_RUN.FAILURE.json"
        helper.CLAIM = root / "THREE_BANK_RUN.CLAIM.json"

    def regular(path: Path) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        os.close(descriptor)

    def published(root: Path, names: set[str] | None = None) -> None:
        bundle = root / "THREE_BANK_BUNDLE"
        bundle.mkdir()
        for name in sorted(helper.EXPECTED_INVENTORY if names is None else names):
            regular(bundle / name)

    try:
        with tempfile.TemporaryDirectory(
            dir="/private/tmp", prefix="q24_v5_postseal_v1r1_"
        ) as temporary:
            parent = Path(temporary)

            missing = parent / "missing_bundle"
            missing.mkdir()
            bind(missing)
            regular(helper.CLAIM)
            expect_handoff_error(
                helper,
                lambda: helper._require_current_atomic_ready("synthetic missing"),
                "missing bundle",
            )

            staging = parent / "staging"
            staging.mkdir()
            bind(staging)
            regular(helper.CLAIM)
            helper.STAGING.mkdir()
            expect_handoff_error(
                helper,
                lambda: helper._require_current_atomic_ready("synthetic staging"),
                "staging state",
            )

            failure = parent / "failure"
            failure.mkdir()
            bind(failure)
            regular(helper.CLAIM)
            regular(helper.FAILURE)
            expect_handoff_error(
                helper,
                lambda: helper._require_current_atomic_ready("synthetic failure"),
                "failure state",
            )

            transaction_drift = parent / "transaction_drift"
            transaction_drift.mkdir()
            bind(transaction_drift)
            regular(helper.CLAIM)
            published(transaction_drift)
            regular(transaction_drift / "UNEXPECTED")
            expect_handoff_error(
                helper,
                helper.transaction_status,
                "transaction inventory mutation",
            )

            bundle_drift = parent / "bundle_drift"
            bundle_drift.mkdir()
            bind(bundle_drift)
            regular(helper.CLAIM)
            published(bundle_drift, set(helper.EXPECTED_INVENTORY) - {helper.RECEIPT_NAME})
            expect_handoff_error(
                helper,
                helper.transaction_status,
                "bundle inventory mutation",
            )

            ready = parent / "ready"
            ready.mkdir()
            bind(ready)
            regular(helper.CLAIM)
            published(ready)
            assert helper._require_current_atomic_ready("synthetic ready")["status"] == (
                "READY_ATOMIC_BUNDLE_PUBLISHED"
            )
    finally:
        for name, value in original.items():
            setattr(helper, name, value)
    return {
        "synthetic_missing_bundle_rejected": True,
        "synthetic_staging_state_rejected": True,
        "synthetic_failure_state_rejected": True,
        "synthetic_transaction_inventory_mutation_rejected": True,
        "synthetic_bundle_inventory_mutation_rejected": True,
        "synthetic_exact_ready_inventory_accepted": True,
    }


def synthetic_full_verification(helper, meter_names: set[str]) -> dict:
    receipt_sha = "1" * 64
    file_sha256 = {name: "2" * 64 for name in helper.EXPECTED_INVENTORY}
    file_sha256[helper.RECEIPT_NAME] = receipt_sha
    chain = {
        "source_authority_sha256": helper.STATIC_ROOTS["source_authority"][1],
        "source_authority_review_sha256": helper.STATIC_ROOTS[
            "source_authority_review"
        ][1],
        "generation_claim_sha256": helper.STATIC_ROOTS["generation_claim"][1],
        "generation_receipt_sha256": receipt_sha,
        "generation_execution_anchor_sha256": helper.STATIC_ROOTS[
            "generation_anchor"
        ][1],
        "generation_final_release_sha256": helper.STATIC_ROOTS[
            "generation_release"
        ][1],
        "generation_single_use_capability_sha256": helper.STATIC_ROOTS[
            "consumed_generation_capability"
        ][1],
        "pre_generation_v4_terminal_successor_record_sha256": helper.STATIC_ROOTS[
            "v4_terminal_successor_record"
        ][1],
        "v4_terminal_failure_lineage_sha256": helper.STATIC_ROOTS[
            "v4_terminal_failure_lineage"
        ][1],
        "independent_incident_root_cause_sha256": helper.STATIC_ROOTS[
            "independent_incident_root_cause"
        ][1],
        "flopscope_reset_verification_sha256": helper.STATIC_ROOTS[
            "flopscope_reset_verification"
        ][1],
        "v4_exclusive_claim_sha256": (
            "d690aa5b42dd0bca12705795004640413368a4384bce467bad2a16c6560fd2b7"
        ),
        "v4_terminal_failure_sha256": (
            "fbac37a97c64983fbc56627b67fdebb59dca07d0f43f3762798495022ed74dae"
        ),
        "v4_consumed_capability_sha256": (
            "24db92e64b4129d3a8c002620017aa516e2287cb9183c5edf9c99ec35c6c4518"
        ),
        "manifest_compatibility_scan_receipt_sha256": helper.STATIC_ROOTS[
            "manifest_compatibility_scan_receipt"
        ][1],
        "generation_launch_command_sha256": helper.STATIC_ROOTS[
            "generation_launch_command"
        ][1],
    }
    compatibility = {
        "receipt_schema_exact_v4": True,
        "receipt_top_level_keyset_exact_v4": True,
        "runtime_profile_exact_lean_v5": True,
        "nested_reset_field_absent": True,
        "post_commit_reset_law_authenticated": True,
        "flopscope_reset_verification_sha256": helper.STATIC_ROOTS[
            "flopscope_reset_verification"
        ][1],
        "reset_execution_independently_observed": False,
        "authorized_interpreter_version": list(helper.AUTHORIZED_INTERPRETER_VERSION),
        "authorized_interpreter_sha256": helper.AUTHORIZED_INTERPRETER_SHA256,
        "oof_v4r1_science_identity_uses_exact_source_bytes": True,
        "interpreter_dependent_ast_dump_used_as_authority": False,
    }
    meter_names_sorted = sorted(meter_names)
    meter_evidence = {
        "meter_vector_names": meter_names_sorted,
        "meter_vector_count": len(meter_names_sorted),
        "meter_vector_length": helper.ROWS,
        "meter_vector_row_scalar_comparisons": len(meter_names_sorted) * helper.ROWS,
        "finite_nonnegative_float_meter_values": 2 * helper.ROWS,
        "nonnegative_integer_meter_values": (len(meter_names_sorted) - 2)
        * helper.ROWS,
        "all_meter_vectors_align_with_row_scalars": True,
    }
    verification = {
        "schema": "public-q24-lean-v5-post-generation-handoff-verification-v1",
        "status": "PASS_EXACT_ATOMIC_V5_BUNDLE_READY_FOR_CANONICAL_SEAL",
        "issued_by": "postseal_handoff_verifier",
        "issued_utc": "2026-08-28T00:00:00Z",
        "atomic_publication": {
            "transaction_root": str(helper.TRANSACTION_ROOT),
            "published_bundle": str(helper.BUNDLE),
            "claim_present": True,
            "staging_absent": True,
            "failure_absent": True,
            "exact_seven_file_inventory": True,
            "publish_law": "ONE_ATOMIC_DIRECTORY_RENAME",
        },
        "static_roots": {
            name: expected for name, (_, expected) in helper.STATIC_ROOTS.items()
        },
        "generation_chain": chain,
        "receipt_compatibility": compatibility,
        "bundle_evidence": {
            "file_sha256": file_sha256,
            "complete_bank_bytes_read": {
                str(h): helper.EXPECTED_FILES[h][1] for h in helper.EXPECTED_H
            },
            "A15_bytes_read": {
                str(h): helper.EXPECTED_FILES[h][3] for h in helper.EXPECTED_H
            },
            "layer_hashes_recomputed": helper.ROWS
            * helper.LAYERS
            * len(helper.EXPECTED_H),
            "row_hashes_recomputed": helper.ROWS * len(helper.EXPECTED_H),
            "finite_float32_values_checked": helper.ROWS
            * helper.LAYERS
            * helper.WIDTH
            * sum(helper.EXPECTED_H),
            "vectorized_sector_prefix_checks": helper.ROWS * helper.LAYERS * 3,
            "logical_unit_sector_prefix_comparisons": helper.ROWS
            * helper.LAYERS
            * helper.WIDTH
            * helper.SECTORS
            * 3,
            "all_float32_finite": True,
            "all_A15_bytes_zero": True,
            "all_row_hashes_match_receipt": True,
            "all_layer_hashes_match_receipt": True,
            "all_sector_prefixes_byte_identical": True,
            "maximum_bank_layer_payload_bytes_buffered": helper.WIDTH
            * sum(helper.EXPECTED_H)
            * 4,
            "bank_hash_block_bytes": 8 << 20,
            "maximum_canonical_json_bytes": helper.MAX_CANONICAL_JSON_BYTES,
            "resource_bound": (
                "BANK_PAYLOAD_STREAMING__ONE_THREE_H_LAYER__"
                "CANONICAL_JSON_CAPPED_128MIB"
            ),
        },
        "sentinel_evidence": {
            "sentinel_ranks": [0, 343, 687],
            "sentinel_receipt_claims_authenticated": True,
            "sentinel_bank_digest_crosswalk_verified": True,
            "non_sentinel_rows_with_null_dual_parity_evidence": 685,
            "all_row_dual_parity_claimed": False,
            "sentinel_comparators_reexecuted_by_handoff_verifier": False,
        },
        "meter_evidence": meter_evidence,
        "access_accounting": {
            "bundle_files_opened": 7,
            "staging_entries_opened": 0,
            "target_artifacts_opened": 0,
            "target_values_decoded": 0,
            "weight_arrays_opened": 0,
            "row_manifests_opened": 0,
            "scores_opened": 0,
            "training_runs": 0,
            "estimator_runs": 0,
            "submissions": 0,
        },
        "limitations": {
            "runtime_target_freedom": (
                "AUTHENTICATED_RECEIPT_AND_FROZEN_CONTROL_FLOW_"
                "NOT_EXTERNAL_PROCESS_TRACING"
            ),
            "sentinel_parity": (
                "AUTHENTICATED_RECEIPT_DIGEST_CROSSWALK_"
                "NOT_COMPARATOR_REEXECUTION"
            ),
            "hostile_interpreter_security_claim": False,
        },
    }
    helper._validate_verification(verification)
    return verification


def canonical_replay_tamper_fixtures(helper, meter_names: set[str]) -> int:
    verification = synthetic_full_verification(helper, meter_names)
    bundle = verification["bundle_evidence"]
    chain = verification["generation_chain"]
    evidence = {
        "receipt_sha256": chain["generation_receipt_sha256"],
        "claim_sha256": chain["generation_claim_sha256"],
        "source_authority_sha256": chain["source_authority_sha256"],
        "source_authority_review_sha256": chain[
            "source_authority_review_sha256"
        ],
        "bank_evidence": {
            "complete_bank_sha256": {
                str(h): bundle["file_sha256"][helper.EXPECTED_FILES[h][0]]
                for h in helper.EXPECTED_H
            },
            "A15_zero_sha256": {
                str(h): bundle["file_sha256"][helper.EXPECTED_FILES[h][2]]
                for h in helper.EXPECTED_H
            },
            "sector_prefix_comparisons": bundle[
                "logical_unit_sector_prefix_comparisons"
            ],
        },
        "handoff_meter_evidence": verification["meter_evidence"],
        "handoff_receipt_compatibility": verification["receipt_compatibility"],
    }
    helper._crosscheck_canonical_sealer_evidence(verification, evidence)
    mutations = []
    receipt_tamper = copy.deepcopy(evidence)
    receipt_tamper["receipt_sha256"] = "3" * 64
    mutations.append(("receipt", receipt_tamper))
    bank_tamper = copy.deepcopy(evidence)
    bank_tamper["bank_evidence"]["complete_bank_sha256"]["16"] = "3" * 64
    mutations.append(("bank", bank_tamper))
    prefix_tamper = copy.deepcopy(evidence)
    prefix_tamper["bank_evidence"]["sector_prefix_comparisons"] -= 1
    mutations.append(("prefix", prefix_tamper))
    meter_tamper = copy.deepcopy(evidence)
    meter_tamper["handoff_meter_evidence"][
        "meter_vector_row_scalar_comparisons"
    ] -= 1
    mutations.append(("meter", meter_tamper))
    reset_tamper = copy.deepcopy(evidence)
    reset_tamper["handoff_receipt_compatibility"][
        "flopscope_reset_verification_sha256"
    ] = "3" * 64
    mutations.append(("reset", reset_tamper))
    for label, mutation in mutations:
        expect_handoff_error(
            helper,
            lambda mutation=mutation: helper._crosscheck_canonical_sealer_evidence(
                verification, mutation
            ),
            f"canonical replay {label} tamper",
        )
    return len(mutations)


def main() -> None:
    helper_source = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(helper_source)
    helper = load_helper()
    commands = json.loads(COMMANDS.read_text(encoding="ascii"))
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))

    static_roots = helper.verify_static_roots()
    assert static_roots == {
        name: expected for name, (_, expected) in helper.STATIC_ROOTS.items()
    }
    assert len(static_roots) == 42
    assert (
        static_roots["predecessor_v4_postseal_independent_review"]
        == "05d74c21d19a4b15a210d9c9b530eca109579223d019b951f216284f734432f7"
    )
    assert static_roots["runtime_v5_package_manifest"] == (
        "6307642b0ebe6397e9ffb7c5d163fe29970b5644fb39f76a240fe59afbd4935d"
    )
    assert static_roots["runtime_v5_runner"] == (
        "db30d0743940b654116d94542278b970caa3eece8d9da0d641f46e0291e0c3c4"
    )
    assert static_roots["runtime_v5_independent_review"] == (
        "26af3b9cf9cb4320b9dfff1917723655ee7ba4046dc99a302fcd2f1608cb2f8e"
    )
    assert static_roots["generation_launch_command"] == (
        "8110b966be52e865e34b9470e16423444a513ef428927dfd97ee24fd29e90bd2"
    )
    assert static_roots["oof_v4r1_package_manifest"] == (
        "eaa79cc6da52140ab13d452ff95c46f4beee3d5261f207b4b708fd9ec0daecca"
    )
    assert static_roots["oof_v4r1_authority_bindings"] == (
        "6b213850137f6610a359850144d0b5857eab31c27b68cee41ff674f5ee129fac"
    )
    assert static_roots["oof_v4r1_static_verifier"] == (
        "aefb0ae4360750a085adeee1b311cdc7ac807acf8646afee09f14173b51f490d"
    )
    assert static_roots["held_v1_postseal_package_manifest"] == (
        "ceda2efca3599cbd017795013ce83994a5bebf5d3a6547e9f175356fac4fdc20"
    )
    assert static_roots["held_v1_postseal_freeze_receipt"] == (
        "52305fb1da405de88ed96d1b62553fe56e406df837b08264283a08f30ba3327e"
    )
    assert static_roots["held_v1_postseal_freeze_verification"] == (
        "6f6633de0f73c8d13df8b485a1ea68cc174e5451249833ce1f2c2be48ef7dc4d"
    )
    assert static_roots["held_v1_postseal_static_verification"] == (
        "b6c89c45b6bf49a0938c8025582075f0f7ba857da87e50e00b402e991f889970"
    )
    assert static_roots["held_v1_postseal_helper"] == (
        "5d06d2de99591f0a79239fb397a99526d2246ac0a9a3cd50dbaa725370113421"
    )
    assert static_roots["held_v1_postseal_independent_review"] == (
        "e0f6aa3f87f549c14272be886746a95ee4235f4dd1de2eb453aebb0e55bce201"
    )
    assert helper.verify_authorized_interpreter() == {
        "implementation": "CPython",
        "version": [3, 11, 5],
        "invocation_path": str(P2 / "data_gate" / ".venv" / "bin" / "python"),
        "resolved_binary_sha256": (
            "6116d7bcb852d61f4384daa1c43488771e6794636d959f2a7065b54c92a070a1"
        ),
    }

    status_calls = function_calls(tree, "transaction_status")
    forbidden_status_calls = {
        "_read_file",
        "_read_canonical",
        "_sha256",
        "open",
        "read",
        "read_bytes",
        "verify_static_roots",
    }
    assert status_calls.isdisjoint(forbidden_status_calls)
    assert "os.scandir(root)" in helper_source
    assert "os.scandir(bundle_root)" in helper_source
    assert "os.scandir(STAGING)" not in helper_source
    assert "STAGING.iterdir" not in helper_source

    main_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    issuance_branch = next(
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "issue_review_request"
    )
    review_request_calls = call_counts(issuance_branch.body)
    consumer_handoff_calls = call_counts(issuance_branch.orelse)
    for calls, builder in (
        (review_request_calls, "build_review_request"),
        (consumer_handoff_calls, "build_consumer_handoff"),
    ):
        assert calls.get("_require_current_atomic_ready") == 2
        assert calls.get("_canonical_sealer_evidence") == 1
        assert calls.get("_crosscheck_canonical_sealer_evidence") == 1
        assert calls.get("_load_frozen") == 1
        assert calls.get(builder) == 1

    assert commands["commands_are_direct_argv_without_shell"] is True
    assert commands["schema"] == (
        "public-q24-lean-v5-post-generation-command-plan-v1r1"
    )
    assert commands["environment"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    steps = commands["ordered_commands"]
    assert [step["step"] for step in steps] == [1, 2, 3, 4, 5, 6]
    assert steps[0]["required_status"] == "READY_ATOMIC_BUNDLE_PUBLISHED"
    assert steps[1]["resource_bound"] == (
        "BANK_PAYLOAD_STREAMING__ONE_THREE_H_LAYER__CANONICAL_JSON_CAPPED_128MIB"
    )
    assert steps[4]["must_be_run_by_reviewer_different_from_seal_issuer"] is True
    assert steps[4]["must_be_run_by_agent_different_from_seal_issuer"] is True
    assert "--review-request" in steps[4]["argv"]
    assert "or args.review_request is None" in helper_source
    for step in steps:
        assert type(step["argv"]) is list and step["argv"]
        assert all(type(item) is str and item for item in step["argv"])
        assert step["argv"][1] == str(HELPER)
    assert contract["access_firewall"]["target_access_allowed"] is False
    assert contract["schema"] == (
        "public-q24-lean-v5-post-generation-handoff-contract-v1r1"
    )
    assert contract["narrow_successor_lineage"]["held_v1_review_sha256"] == (
        "e0f6aa3f87f549c14272be886746a95ee4235f4dd1de2eb453aebb0e55bce201"
    )
    assert set(contract["protected_issuance_replay"]) == {
        "review_request",
        "consumer_handoff",
    }
    assert all(
        "CURRENT_EXACT_ATOMIC_READY_BEFORE_AND_AFTER_CANONICAL_SEALER_"
        "RECEIPT_BANK_PREFIX_METER_RESET_REPLAY"
        == value
        for value in contract["protected_issuance_replay"].values()
    )
    assert contract["independent_full_bundle_checks"]["all_seven_files_sha256"] is True
    assert contract["independent_full_bundle_checks"]["meter_vector_alignment"] is True
    assert contract["independent_full_bundle_checks"]["reset_compatible_receipt"] is True
    assert contract["independent_full_bundle_checks"][
        "maximum_canonical_json_bytes"
    ] == 128 << 20
    assert contract["independent_full_bundle_checks"]["streaming_resource_bound"] == (
        "BANK_PAYLOAD_STREAMING__ONE_THREE_H_LAYER__CANONICAL_JSON_CAPPED_128MIB"
    )
    assert contract["oof_handoff_profile"] == (
        "EXACT_CORRECTED_FROZEN_NESTED_ACCURACY_OOF_V4R1_BANK_BINDING_INVENTORY"
    )
    assert helper.OOF.name == "public_q24_signed_pair_nested_accuracy_oof_v4r1"
    assert helper.STATIC_ROOTS["oof_v4r1_independent_static_review"][0].parts[-2:] == (
        "public_q24_signed_pair_nested_accuracy_oof_review_v4r1",
        "INDEPENDENT_STATIC_REVIEW_V4R1.json",
    )
    bindings = json.loads(
        helper.STATIC_ROOTS["oof_v4r1_authority_bindings"][0].read_text(
            encoding="ascii"
        )
    )
    assert bindings["schema"] == (
        "public-q24-signed-pair-nested-accuracy-oof-v4r1-authority-bindings-v1"
    )
    assert bindings["status"] == (
        "READY_PINNED_CPYTHON_3_11_5_STATIC_SUCCESSOR__EXECUTION_HOLD"
    )
    assert bindings["authorized_interpreter"]["version"] == [3, 11, 5]
    assert bindings["science_identity"]["ast_dump_digest_used_as_authority"] is False
    assert bindings["predecessor_hold"]["science_drift_detected"] is False
    assert bindings["authorization"]["nested_anchor_review_issue"] is False

    atomic_results = atomic_gate_fixtures(helper)

    import numpy as np

    width = 3
    h256 = np.arange(width * 256, dtype=np.float32).reshape(width, 8, 32)
    payload = {
        16: h256[:, :, :2].reshape(width, 16).astype("<f4").tobytes(),
        64: h256[:, :, :8].reshape(width, 64).astype("<f4").tobytes(),
        256: h256.reshape(width, 256).astype("<f4").tobytes(),
    }
    assert helper.verify_layer_payloads(payload, np=np, width=width) == 3

    prefix_mutation = dict(payload)
    changed = bytearray(prefix_mutation[16])
    changed[0] ^= 1
    prefix_mutation[16] = bytes(changed)
    try:
        helper.verify_layer_payloads(prefix_mutation, np=np, width=width)
    except helper.HandoffError:
        pass
    else:
        raise AssertionError("prefix mutation was accepted")

    nonfinite = dict(payload)
    changed_floats = np.frombuffer(nonfinite[256], dtype="<f4").copy()
    changed_floats[-1] = np.inf
    nonfinite[256] = changed_floats.tobytes()
    try:
        helper.verify_layer_payloads(nonfinite, np=np, width=width)
    except helper.HandoffError:
        pass
    else:
        raise AssertionError("nonfinite mutation was accepted")

    meter_names = {
        "complete_flops_by_rank",
        "complete_calls_by_rank",
        "base_parent_flops_by_rank",
        "base_parent_calls_by_rank",
        "capture_parent_flops_by_rank",
        "capture_parent_calls_by_rank",
        "literal_source_hook_flops_by_rank",
        "literal_source_hook_calls_by_rank",
        "reference_source_hook_flops_by_rank",
        "reference_source_hook_calls_by_rank",
        "literal_workspace_flops_by_rank",
        "literal_workspace_calls_by_rank",
        "reference_workspace_flops_by_rank",
        "reference_workspace_calls_by_rank",
        "peak_rss_bytes_by_rank",
        "wall_seconds_by_rank",
        "residual_seconds_by_rank",
        "reference_source_hook_capture_count_by_rank",
    }
    rows = [{"exact_meters": {}} for _ in range(3)]
    vectors = {}
    for offset, name in enumerate(sorted(meter_names)):
        scalar = name.removesuffix("_by_rank")
        if name in {"wall_seconds_by_rank", "residual_seconds_by_rank"}:
            vector = [float(offset + rank) for rank in range(3)]
        else:
            vector = [offset + rank for rank in range(3)]
        vectors[name] = vector
        for rank, value in enumerate(vector):
            rows[rank]["exact_meters"][scalar] = value
    meter_receipt = {
        "row_receipts": rows,
        "complete_banks": {"exact_meters": vectors},
    }
    meter_result = helper.verify_meter_vector_alignment(
        meter_receipt, expected_rows=3
    )
    assert meter_result["meter_vector_row_scalar_comparisons"] == 54
    meter_mutation = copy.deepcopy(meter_receipt)
    meter_mutation["complete_banks"]["exact_meters"][
        "complete_flops_by_rank"
    ][1] += 1
    try:
        helper.verify_meter_vector_alignment(meter_mutation, expected_rows=3)
    except helper.HandoffError:
        pass
    else:
        raise AssertionError("meter-vector misalignment was accepted")

    canonical_tamper_rejections = canonical_replay_tamper_fixtures(
        helper, meter_names
    )
    assert canonical_tamper_rejections == 5

    oof_runner = helper._load_frozen(
        helper.OOF_RUNNER,
        helper.STATIC_ROOTS["oof_v4r1_runner"][1],
        "q24_v5_postseal_oof_v4_inventory_test",
    )
    assert oof_runner.GENERATION_RECEIPT_PROFILE == "EXACT_LEAN_RUNTIME_V5"
    assert oof_runner.BANK_BINDING_KEYS == helper.OOF_BANK_BINDING_KEYS
    assert helper.OOF_STATIC_PACKAGE_BINDING_KEYS == {
        "package_manifest_sha256",
        "freeze_receipt_sha256",
        "freeze_verification_sha256",
        "static_verification_sha256",
        "runner_sha256",
        "post_generation_sealer_sha256",
        "post_generation_seal_contract_sha256",
        "v5_provenance_fixtures_sha256",
        "v4r1_authority_bindings_sha256",
        "static_verifier_sha256",
        "independent_static_review_sha256",
    }
    assert helper._validate_generation_command() == static_roots[
        "generation_launch_command"
    ]

    result = {
        "schema": "public-q24-lean-v5-post-generation-handoff-static-verification-v1r1",
        "status": "PASS_STATIC_SYNTHETIC_NO_LIVE_BANK_OR_TARGET_OPEN",
        "helper_sha256": sha256(HELPER),
        "contract_sha256": sha256(CONTRACT),
        "command_plan_sha256": sha256(COMMANDS),
        "static_roots_verified": len(static_roots),
        "status_mode_metadata_only": True,
        "live_staging_opened": False,
        "live_bank_opened": False,
        "target_artifacts_opened": 0,
        "synthetic_valid_prefix_accepted": True,
        "synthetic_prefix_mutation_rejected": True,
        "synthetic_nonfinite_mutation_rejected": True,
        "synthetic_valid_meter_alignment_accepted": True,
        "synthetic_meter_misalignment_rejected": True,
        **atomic_results,
        "canonical_replay_tamper_mutations_rejected": canonical_tamper_rejections,
        "reset_compatible_v5_receipt_profile_bound": True,
        "generation_launch_command_bound": True,
        "canonical_review_request_required": True,
        "streaming_resource_bound_verified": True,
        "oof_v4r1_pinned_interpreter_successor_bound": True,
        "protected_modes_require_authorized_interpreter": True,
        "oof_anchor_bank_binding_inventory_exact": True,
        "held_v1_independent_hold_review_bound": True,
        "review_request_branch_atomic_replay_gate_reachable": True,
        "consumer_handoff_branch_atomic_replay_gate_reachable": True,
    }
    if STATIC.exists():
        assert json.loads(STATIC.read_text(encoding="ascii")) == result
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
