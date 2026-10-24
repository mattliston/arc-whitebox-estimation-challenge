#!/usr/bin/env python3
"""Additive one-use V2 driver for the frozen full-budget joint scorer.

V1 is permanently consumed and is never modified, retried, resumed, or called
through its execution entry point.  V2 binds V1's exact science source and
entire release/claim/failure chain, imports that frozen scoring core, and
changes only the PyArrow 11 RecordBatch schema-name access from
``batch.column_names`` to ``batch.schema.names``.  The default path is
target-free and creates no state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterator, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
V2_BINDINGS_PATH = HERE / "V2_BINDINGS.json"
SYNTHETIC_RECEIPT_PATH = HERE / "SYNTHETIC_ITERATOR_RECEIPT_V2.json"
STATIC_RECEIPT_PATH = HERE / "STATIC_PREPARATION_RECEIPT_V2.json"
INDEPENDENT_AUDIT_PATH = HERE / "INDEPENDENT_SCORER_AUDIT_V2.json"
ROOT_RELEASE_PATH = HERE / "ROOT_SINGLE_TARGET_OPEN_RELEASE_V2.json"
ONE_USE_CLAIM_PATH = (
    HERE / "receipts" / "claims" / "fullbudget-joint-mini100-v2.one-use-target-claim.json"
)
ONE_USE_RESULT_PATH = (
    HERE / "receipts" / "results" / "fullbudget-joint-mini100-v2.joint-raw-mse.json"
)
EXECUTION_TOKEN = "OPEN_FULLBUDGET_JOINT_MINI100_TARGETS_ONCE_V2"
EXECUTION_TOKEN_SHA256 = hashlib.sha256(EXECUTION_TOKEN.encode("utf-8")).hexdigest()
EXPECTED_ROSTER = (
    "incumbent_cp1792",
    "sobol_raw_n64512",
    "iid_paired_whitened_n64000",
    "cap16384",
    "cap16384_layer2_mean_carry",
    "cap16384_recursive_mean_carry",
)
TARGET_COLUMNS = ("mlp_id", "mlp_name", "mlp_seed", "all_layer_means")
N_MLPS = 100
DEPTH = 16
WIDTH = 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ZERO_FIELDS = (
    "target_rows_decoded",
    "target_values_decoded",
    "fits",
    "refits",
    "sweeps",
    "retries",
    "scores",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _workspace_file(relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AssertionError(f"invalid workspace-relative file: {relative!r}")
    lexical = WORKSPACE / relative
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(WORKSPACE.resolve()):
        raise AssertionError(f"bound file escaped workspace: {relative}")
    if lexical.is_symlink() or not resolved.is_file():
        raise AssertionError(f"bound file missing or symlinked: {relative}")
    return resolved


def _assert_exact_environment(bindings: dict[str, Any]) -> dict[str, str]:
    expected = bindings.get("exact_environment", {})
    conda_record_path = Path(str(expected.get("pyarrow_conda_record_path", "")))
    observed = {
        "python_executable_resolved": str(Path(sys.executable).resolve()),
        "required_invocation_executable": "/Users/matt/anaconda3/bin/python3",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "pyarrow_conda_record_path": str(conda_record_path),
        "pyarrow_conda_record_sha256": (
            _sha256_file(conda_record_path) if conda_record_path.is_file() else "MISSING"
        ),
    }
    if observed != expected:
        raise AssertionError(f"V2 exact execution environment mismatch: {observed} != {expected}")
    return observed


def _assert_zero_receipt(receipt: Mapping[str, Any], *, where: str) -> None:
    if receipt.get("claim_created") is not False:
        raise AssertionError(f"{where}: prior V2 claim exists or field is missing")
    if receipt.get("target_iterator_constructed") is not False:
        raise AssertionError(f"{where}: target iterator was constructed")
    for field in ZERO_FIELDS:
        if receipt.get(field) != 0:
            raise AssertionError(f"{where}: nonzero or missing {field}")
    if receipt.get("this_receipt_is_execution_or_target_authority") is not False:
        raise AssertionError(f"{where}: receipt improperly grants authority")


def _verify_self_hash(payload: dict[str, Any], field: str, expected: object) -> None:
    core = dict(payload)
    recorded = core.pop(field, None)
    actual = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    if recorded != expected or recorded != actual:
        raise AssertionError(f"canonical self-hash mismatch: {field}")


def _load_bindings() -> tuple[dict[str, Any], list[str]]:
    bindings = _read_json(V2_BINDINGS_PATH)
    if bindings.get("schema") != "phase2-fullbudget-joint-mini100-v2-bindings-v1":
        raise AssertionError("unexpected V2 binding schema")
    _assert_exact_environment(bindings)
    science = bindings.get("science_contract", {})
    if tuple(science.get("candidate_roster", ())) != EXPECTED_ROSTER:
        raise AssertionError("V2 candidate roster changed")
    if tuple(science.get("target_projection", ())) != TARGET_COLUMNS:
        raise AssertionError("V2 target projection changed")
    if science.get("target_shape_per_row") != [DEPTH, WIDTH] or science.get(
        "target_dtype"
    ) != "float32":
        raise AssertionError("V2 target geometry/dtype changed")
    if science.get("raw_mse_primary") is not True or science.get(
        "official_0p1_adjustment_primary"
    ) is not False:
        raise AssertionError("V2 raw-MSE primary contract changed")
    if science.get("fit_refit_sweep_retry_authority") is not False:
        raise AssertionError("V2 adaptive-work authority changed")

    chain = bindings.get("v1_frozen_chain", {})
    hashed_descriptors = (
        "scorer",
        "bundle_bindings",
        "protocol",
        "static_verifier",
        "static_receipt",
        "independent_audit",
    )
    for name in hashed_descriptors:
        descriptor = chain.get(name, {})
        if not _is_sha256(descriptor.get("sha256")):
            raise AssertionError(f"V1 {name} digest binding missing")
        path = _workspace_file(descriptor.get("path"))
        if _sha256_file(path) != descriptor["sha256"]:
            raise AssertionError(f"frozen V1 artifact changed: {name}")
    for name in ("root_release", "consumed_claim"):
        descriptor = chain.get(name, {})
        if not _is_sha256(descriptor.get("file_sha256")):
            raise AssertionError(f"V1 {name} file binding missing")
        path = _workspace_file(descriptor.get("path"))
        if _sha256_file(path) != descriptor["file_sha256"]:
            raise AssertionError(f"frozen V1 artifact changed: {name}")

    release_descriptor = chain["root_release"]
    release = _read_json(_workspace_file(release_descriptor["path"]))
    _verify_self_hash(release, "release_payload_sha256", release_descriptor["payload_sha256"])
    if release.get("release_id") != release_descriptor.get("release_id"):
        raise AssertionError("V1 release id changed")
    claim_descriptor = chain["consumed_claim"]
    claim = _read_json(_workspace_file(claim_descriptor["path"]))
    _verify_self_hash(claim, "claim_payload_sha256", claim_descriptor["payload_sha256"])
    if claim.get("release_id") != release.get("release_id"):
        raise AssertionError("V1 claim/release mismatch")
    if claim.get("status") != "SINGLE_TARGET_OPEN_IRREVERSIBLY_CLAIMED__NO_RETRY_OR_RESUME":
        raise AssertionError("V1 claim is not the consumed terminal claim")
    if claim.get("scorer_sha256") != chain["scorer"]["sha256"]:
        raise AssertionError("V1 claim/scorer mismatch")
    if claim.get("root_release_file_sha256") != release_descriptor["file_sha256"]:
        raise AssertionError("V1 claim/release file mismatch")
    result_relative = chain.get("result_path")
    result_path = WORKSPACE / str(result_relative)
    if chain.get("result_must_be_absent") is not True or result_path.exists():
        raise AssertionError("V1 failed opening unexpectedly has a published result")

    blockers: list[str] = []
    failure_descriptor = chain.get("terminal_failure_receipt", {})
    if not all(
        (
            isinstance(failure_descriptor.get("path"), str),
            _is_sha256(failure_descriptor.get("sha256")),
            isinstance(failure_descriptor.get("status"), str),
        )
    ):
        blockers.append("missing_frozen_v1_terminal_failure_receipt")
    else:
        failure_path = _workspace_file(failure_descriptor["path"])
        if _sha256_file(failure_path) != failure_descriptor["sha256"]:
            raise AssertionError("V1 terminal failure receipt hash mismatch")
        failure = _read_json(failure_path)
        if failure.get("status") != failure_descriptor["status"] or not str(
            failure.get("status", "")
        ).startswith("PASS"):
            raise AssertionError("V1 terminal failure receipt is not the bound PASS audit")
        frozen = failure.get("frozen_bindings", {})
        if frozen.get("scorer_sha256") != chain["scorer"]["sha256"]:
            raise AssertionError("V1 failure receipt/scorer mismatch")
        if frozen.get("one_use_claim_file_sha256") != claim_descriptor["file_sha256"]:
            raise AssertionError("V1 failure receipt/claim mismatch")
        materialization = failure.get("physical_materialization_boundary", {})
        expected_materialization = {
            "parquet_record_batches_materialized": 1,
            "parquet_logical_rows_materialized": 1,
            "all_layer_means_float32_values_materialized_in_arrow": DEPTH * WIDTH,
            "identity_python_scalars_extracted": 0,
            "all_layer_means_numpy_arrays_extracted": 0,
            "all_layer_means_python_scalars_extracted": 0,
            "target_rows_yielded_by_scorer": 0,
            "target_rows_consumed_by_metric_loop": 0,
        }
        for field, expected in expected_materialization.items():
            if materialization.get(field) != expected:
                raise AssertionError(f"V1 failure materialization mismatch: {field}")
        scientific = failure.get("scientific_boundary", {})
        for field in (
            "prediction_target_differences_computed",
            "squared_error_arrays_computed",
            "metric_accumulator_updates",
            "candidate_metrics_computed",
            "fits",
            "refits",
            "sweeps",
            "retries",
        ):
            if scientific.get(field) != 0:
                raise AssertionError(f"V1 failure scientific boundary mismatch: {field}")
        terminal = failure.get("persistence_and_terminal_state", {})
        if terminal.get("result_file_created") is not False:
            raise AssertionError("V1 failure receipt does not bind absent result")
        if terminal.get("authorized_retry_or_resume_path") is not False:
            raise AssertionError("V1 failure receipt accidentally authorizes V1 reuse")

    observation = bindings.get("v1_terminal_observation", {})
    expected_observation = {
        "exit_code": 1,
        "exception_type": "AttributeError",
        "exception_message": "'pyarrow.lib.RecordBatch' object has no attribute 'column_names'",
        "failing_expression_v1": "tuple(batch.column_names)",
        "replacement_expression_v2": "tuple(batch.schema.names)",
        "failure_stage": "first_projected_record_batch_schema_order_check",
        "projected_record_batches_materialized": 1,
        "projected_rows_materialized": 1,
        "arrow_target_float32_values_materialized": DEPTH * WIDTH,
        "identity_python_scalars_extracted": 0,
        "numpy_target_arrays_extracted": 0,
        "numpy_target_scalars_extracted": 0,
        "target_rows_yielded": 0,
        "target_rows_consumed_by_metric_loop": 0,
        "metric_accumulator_updates": 0,
        "target_values_persisted": 0,
        "result_published": False,
        "v1_claim_consumed": True,
        "v1_retry_or_resume_authorized": False,
    }
    if observation != expected_observation:
        raise AssertionError("V1 terminal observation binding changed")
    if not blockers and bindings.get("status") != "BOUND_V1_TERMINAL_FAILURE_AND_EXACT_V2_SCIENCE":
        raise AssertionError("V2 binding status was not finalized")
    return bindings, blockers


def _load_frozen_v1_core(bindings: dict[str, Any]) -> Any:
    descriptor = bindings["v1_frozen_chain"]["scorer"]
    path = _workspace_file(descriptor["path"])
    if _sha256_file(path) != descriptor["sha256"]:
        raise AssertionError("V1 scoring core changed before import")
    module_name = "_frozen_fullbudget_joint_score_v1_core_for_v2"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("could not construct frozen V1 science import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _decode_exact_record_batch(core: Any, batch: Any, ordinal: int) -> Any:
    """PyArrow-11-compatible exact schema/order/dtype/shape decoder."""

    if batch.num_rows != 1 or tuple(batch.schema.names) != TARGET_COLUMNS:
        raise AssertionError("physical target projection violated exact four-column contract")
    identity = (
        int(batch.column(batch.schema.get_field_index("mlp_id"))[0].as_py()),
        str(batch.column(batch.schema.get_field_index("mlp_name"))[0].as_py()),
        int(batch.column(batch.schema.get_field_index("mlp_seed"))[0].as_py()),
    )
    values = core._nested_float32_view(batch, "all_layer_means", 2)
    if values.dtype != np.dtype("float32") or values.size != DEPTH * WIDTH:
        raise AssertionError("target RecordBatch dtype/shape changed")
    targets = values.reshape(DEPTH, WIDTH)
    if not np.isfinite(targets).all():
        raise AssertionError(f"target row {ordinal} contains a non-finite value")
    return core.TargetRow(ordinal=ordinal, identity=identity, all_layer_means=targets)


def _iter_pinned_exact_targets_v2(core: Any, authority_binding: dict[str, Any]) -> Iterator[Any]:
    """Sole production target iterator; entered only after the fresh V2 claim."""

    import pyarrow.parquet as pq  # Exact-env local import, deliberately post-claim.

    authority_path = core._workspace_path(authority_binding["path"], file=True)
    if core._sha256_file(authority_path) != authority_binding["sha256"]:
        raise AssertionError("post-claim authority hash mismatch")
    authority = core._read_json(authority_path)
    snapshot = (
        authority_path.parent
        / ".hf-cache"
        / "hub"
        / "datasets--aicrowd--arc-whestbench-public-2026"
        / "snapshots"
        / authority["pinned_commit_sha"]
    )
    ordinal = 0
    for expected in authority["splits"]["mini"]["shards"]:
        resolved = core._attest_shard_pointer(snapshot / expected["path"], expected)
        batches = pq.ParquetFile(resolved).iter_batches(
            batch_size=1,
            columns=list(TARGET_COLUMNS),
            use_threads=False,
        )
        for batch in batches:
            yield _decode_exact_record_batch(core, batch, ordinal)
            ordinal += 1
    if ordinal != N_MLPS:
        raise AssertionError(f"pinned target iterator yielded {ordinal}, expected {N_MLPS}")


def audit_target_free() -> tuple[Any | None, tuple[Any, ...], dict[str, Any]]:
    bindings, blockers = _load_bindings()
    core = _load_frozen_v1_core(bindings)
    verified, v1_bundle_audit = core.audit_prediction_bundles()
    if v1_bundle_audit.get("status") != "READY_FOR_INDEPENDENT_SCORER_AUDIT_AND_ROOT_RELEASE":
        raise AssertionError("frozen V1 science bundle audit no longer passes")
    roster = tuple(candidate.name for bundle in verified for candidate in bundle.candidates)
    if roster != EXPECTED_ROSTER:
        raise AssertionError("frozen V1 science returned a changed roster")
    report = {
        "schema": "phase2-fullbudget-joint-mini100-v2-target-free-audit-v1",
        "status": (
            "READY_FOR_V2_SYNTHETIC_STATIC_AND_INDEPENDENT_AUDIT"
            if not blockers
            else "BLOCKED_AWAITING_FROZEN_V1_TERMINAL_FAILURE_RECEIPT"
        ),
        "v2_scorer_sha256": _sha256_file(Path(__file__).resolve()),
        "v2_bindings_sha256": _sha256_file(V2_BINDINGS_PATH),
        "v1_scoring_core_sha256": bindings["v1_frozen_chain"]["scorer"]["sha256"],
        "v1_consumed_claim_sha256": bindings["v1_frozen_chain"]["consumed_claim"][
            "file_sha256"
        ],
        "exact_environment": bindings["exact_environment"],
        "exact_candidate_roster": list(roster),
        "target_projection": list(TARGET_COLUMNS),
        "blockers": blockers,
        "target_iterator_constructed": False,
        "target_rows_decoded": 0,
        "target_values_decoded": 0,
        "claim_created": False,
        "fits": 0,
        "refits": 0,
        "sweeps": 0,
        "retries": 0,
        "scores": 0,
    }
    return core, verified, report


def _execution_gate(report: dict[str, Any]) -> dict[str, Any]:
    blockers = list(report.get("blockers", []))
    if report.get("status") != "READY_FOR_V2_SYNTHETIC_STATIC_AND_INDEPENDENT_AUDIT":
        blockers.append("v2_target_free_audit_not_ready")
    for label, path in (
        ("synthetic_iterator_receipt", SYNTHETIC_RECEIPT_PATH),
        ("static_preparation_receipt", STATIC_RECEIPT_PATH),
        ("independent_scorer_audit", INDEPENDENT_AUDIT_PATH),
        ("root_single_target_open_release", ROOT_RELEASE_PATH),
    ):
        if not path.is_file():
            blockers.append(f"missing_{label}")
    return {
        "execution_releasable": not blockers,
        "blockers": blockers,
        "v2_claim_path": str(ONE_USE_CLAIM_PATH),
        "v2_claim_created": ONE_USE_CLAIM_PATH.exists(),
        "target_iterator_constructed": False,
        "target_values_decoded": 0,
    }


def _load_v2_release(report: dict[str, Any], bindings: dict[str, Any]) -> tuple[dict[str, Any], str]:
    gate = _execution_gate(report)
    if gate["blockers"]:
        raise PermissionError(f"fresh V2 target opening is blocked: {gate['blockers']}")
    for path in (
        SYNTHETIC_RECEIPT_PATH,
        STATIC_RECEIPT_PATH,
        INDEPENDENT_AUDIT_PATH,
        ROOT_RELEASE_PATH,
    ):
        if path.is_symlink() or path.parent.resolve() != HERE or not path.is_file():
            raise AssertionError(f"invalid V2 gate artifact: {path}")
    scorer_sha = _sha256_file(Path(__file__).resolve())
    bindings_sha = _sha256_file(V2_BINDINGS_PATH)
    synthetic_sha = _sha256_file(SYNTHETIC_RECEIPT_PATH)
    static_sha = _sha256_file(STATIC_RECEIPT_PATH)
    audit_sha = _sha256_file(INDEPENDENT_AUDIT_PATH)
    synthetic = _read_json(SYNTHETIC_RECEIPT_PATH)
    static = _read_json(STATIC_RECEIPT_PATH)
    independent = _read_json(INDEPENDENT_AUDIT_PATH)
    expected_receipts = (
        (synthetic, "PASS_EXACT_ENV_PYARROW11_RECORD_BATCH_ITERATOR_SYNTHETIC_V2", "synthetic"),
        (static, "PASS_STATIC_TARGET_FREE_FULLBUDGET_JOINT_SCORER_V2", "static"),
        (independent, "PASS_INDEPENDENT_ONE_USE_FULLBUDGET_JOINT_SCORER_V2", "independent"),
    )
    for receipt, status, where in expected_receipts:
        if receipt.get("status") != status:
            raise AssertionError(f"{where} V2 receipt is not the exact PASS")
        if receipt.get("v2_scorer_sha256") != scorer_sha:
            raise AssertionError(f"{where} receipt/scorer mismatch")
        if receipt.get("v2_bindings_sha256") != bindings_sha:
            raise AssertionError(f"{where} receipt/bindings mismatch")
        _assert_zero_receipt(receipt, where=where)
    chain = bindings["v1_frozen_chain"]
    failure_sha = chain["terminal_failure_receipt"]["sha256"]
    expected_frozen = {
        "v2_scorer_sha256": scorer_sha,
        "v2_bindings_sha256": bindings_sha,
        "v1_scoring_core_sha256": chain["scorer"]["sha256"],
        "v1_consumed_claim_sha256": chain["consumed_claim"]["file_sha256"],
        "v1_terminal_failure_receipt_sha256": failure_sha,
        "authority_sha256": "b33aea75ee83be5cc7c2778aefd4cc35d731dcd702d8b258269fc2c895ef4ec5",
        "synthetic_iterator_receipt_sha256": synthetic_sha,
        "static_preparation_receipt_sha256": static_sha,
        "independent_scorer_audit_sha256": audit_sha,
        "execution_token_sha256": EXECUTION_TOKEN_SHA256,
    }
    release = _read_json(ROOT_RELEASE_PATH)
    if release.get("schema") != "phase2-fullbudget-joint-mini100-root-target-open-release-v2":
        raise AssertionError("unexpected V2 root release schema")
    if release.get("status") != "ROOT_RELEASE_EXACTLY_ONE_FRESH_FULLBUDGET_JOINT_MINI100_TARGET_OPEN_V2":
        raise AssertionError("unexpected V2 root release status")
    core = dict(release)
    recorded = core.pop("release_payload_sha256", None)
    if recorded != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise AssertionError("V2 root release canonical self-hash mismatch")
    if release.get("frozen_bindings") != expected_frozen:
        raise AssertionError("V2 root release frozen binding map mismatch")
    if release.get("issued_by") != "/root" or release.get(
        "this_release_is_fresh_v2_target_open_authority_only"
    ) is not True:
        raise AssertionError("V2 root release issuer/purpose mismatch")
    if release.get("candidate_roster") != list(EXPECTED_ROSTER) or release.get(
        "target_projection"
    ) != list(TARGET_COLUMNS):
        raise AssertionError("V2 root release science roster/projection mismatch")
    if int(release.get("authorized_fresh_v2_target_opens", -1)) != 1 or int(
        release.get("authorized_target_rows", -1)
    ) != N_MLPS:
        raise AssertionError("V2 root release is not one fresh mini100 opening")
    for key in (
        "v1_retry_or_resume_authorized",
        "v2_retry_or_resume_authorized",
        "fit_refit_sweep_authority",
    ):
        if release.get(key) is not False:
            raise AssertionError(f"V2 root release accidentally grants {key}")
    release_id = release.get("release_id")
    if not _is_sha256(release_id):
        raise AssertionError("V2 root release id is not SHA-256")
    expected_zero = {
        "v2_claim_created": False,
        "target_iterator_constructed": False,
        "target_rows_decoded": 0,
        "target_values_decoded": 0,
        "fits": 0,
        "refits": 0,
        "sweeps": 0,
        "retries": 0,
        "scores": 0,
    }
    if release.get("access_before_release") != expected_zero:
        raise AssertionError("V2 root release zero-access map mismatch")
    return release, str(recorded)


def _score_v2_once(
    core: Any,
    verified: tuple[Any, ...],
    report: dict[str, Any],
    release: dict[str, Any],
    release_payload_sha256: str,
    claim_payload_sha256: str,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    original_iterator = core._iter_pinned_exact_targets
    original_release_path = core.ROOT_RELEASE_PATH
    core._iter_pinned_exact_targets = lambda authority: _iter_pinned_exact_targets_v2(
        core, authority
    )
    core.ROOT_RELEASE_PATH = ROOT_RELEASE_PATH
    try:
        result = core._score_claimed_once(
            verified,
            report,
            release,
            release_payload_sha256,
            ONE_USE_CLAIM_PATH,
            claim_payload_sha256,
        )
    finally:
        core._iter_pinned_exact_targets = original_iterator
        core.ROOT_RELEASE_PATH = original_release_path
    chain = bindings["v1_frozen_chain"]
    result.update(
        {
            "schema": "phase2-fullbudget-joint-mini100-single-target-open-result-v2",
            "status": "COMPLETE_EXACTLY_ONE_FRESH_V2_FULLBUDGET_JOINT_MINI100_RAW_MSE_SCORE",
            "scorer_sha256": _sha256_file(Path(__file__).resolve()),
            "v2_bindings_sha256": _sha256_file(V2_BINDINGS_PATH),
            "frozen_v1_scoring_core_sha256": chain["scorer"]["sha256"],
            "v1_consumed_claim_sha256": chain["consumed_claim"]["file_sha256"],
            "v1_terminal_failure_receipt_sha256": chain["terminal_failure_receipt"][
                "sha256"
            ],
            "exact_execution_environment": bindings["exact_environment"],
            "v1_retry_or_resume_performed": False,
            "fresh_v2_openings_performed": 1,
        }
    )
    return result


def execute_once(token: str) -> dict[str, Any]:
    if token != EXECUTION_TOKEN:
        raise PermissionError("explicit fresh V2 target-open token mismatch")
    core, verified, report = audit_target_free()
    assert core is not None
    bindings, blockers = _load_bindings()
    if blockers:
        raise PermissionError(f"V2 bindings incomplete: {blockers}")
    release, release_payload_sha = _load_v2_release(report, bindings)
    if ONE_USE_CLAIM_PATH.exists():
        raise FileExistsError("fresh V2 opening already consumed; retry/resume forbidden")
    if ONE_USE_RESULT_PATH.exists():
        raise FileExistsError("unexpected pre-existing fixed V2 result")
    chain = bindings["v1_frozen_chain"]
    claim_payload = {
        "schema": "phase2-fullbudget-joint-mini100-one-use-target-claim-v2",
        "status": "FRESH_V2_TARGET_OPEN_IRREVERSIBLY_CLAIMED__NO_V1_OR_V2_RETRY",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": release["release_id"],
        "root_release_file_sha256": _sha256_file(ROOT_RELEASE_PATH),
        "root_release_payload_sha256": release_payload_sha,
        "v2_scorer_sha256": _sha256_file(Path(__file__).resolve()),
        "v2_bindings_sha256": _sha256_file(V2_BINDINGS_PATH),
        "v1_scoring_core_sha256": chain["scorer"]["sha256"],
        "v1_consumed_claim_sha256": chain["consumed_claim"]["file_sha256"],
        "v1_terminal_failure_receipt_sha256": chain["terminal_failure_receipt"]["sha256"],
        "candidate_roster": list(EXPECTED_ROSTER),
        "target_projection": list(TARGET_COLUMNS),
        "expected_result_path": str(ONE_USE_RESULT_PATH),
        "v1_retry_or_resume": False,
        "v2_retry_or_resume": False,
        "target_iterator_constructed_before_claim": False,
        "target_values_decoded_before_claim": 0,
        "fits": 0,
        "refits": 0,
        "sweeps": 0,
        "retries": 0,
    }
    claim_payload_sha = core._write_exclusive_json(
        ONE_USE_CLAIM_PATH, claim_payload, "claim_payload_sha256"
    )
    result = _score_v2_once(
        core,
        verified,
        report,
        release,
        release_payload_sha,
        claim_payload_sha,
        bindings,
    )
    result_payload_sha = core._publish_atomic_no_replace(ONE_USE_RESULT_PATH, result)
    return {
        "status": result["status"],
        "claim_path": str(ONE_USE_CLAIM_PATH),
        "result_path": str(ONE_USE_RESULT_PATH),
        "result_payload_sha256": result_payload_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-single-fresh-v2-target-open",
        metavar="TOKEN",
        help="irreversibly consume the independently audited/root-released fresh V2 opening",
    )
    args = parser.parse_args()
    if args.execute_single_fresh_v2_target_open is not None:
        print(
            json.dumps(
                execute_once(args.execute_single_fresh_v2_target_open),
                indent=2,
                sort_keys=True,
            )
        )
        return
    _, _, report = audit_target_free()
    print(json.dumps({**report, "execution_gate": _execution_gate(report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
