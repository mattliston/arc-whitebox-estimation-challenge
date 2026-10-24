#!/usr/bin/env python3
"""One-use CPU launcher for the Phase-2 V4 observer fit/selection session.

The command-line surface is deliberately preflight-only.  A later audited
launcher may call ``execute_authorized_once`` with separately issued reader
and training credentials.  The irreversible claim is durably created before
the target reader, feature provider, training core, converter, any target
iteration, or any optimizer construction.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from feature_store_contract import preflight as feature_contract_preflight
from feature_store_contract import validate_future_manifest
from sealed_bundle_gate import preflight as bundle_preflight
from sealed_bundle_gate import sha256_file, validate_activation


HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parents[1]
DATA_GATE = PHASE2 / "data_gate"
V3 = HERE.parent / "v3"
CLAIM_PATH = HERE / "PHASE2_OBSERVER_V4_TRAIN_ONCE.CLAIM"
OUTPUT_PATH = HERE / "authorized_fit_selection_result"
STAGING_PATH = HERE / "authorized_fit_selection_result.staging"
TORCH_PYTHON = Path("/Users/matt/anaconda3/bin/python")
READER_SOURCE = DATA_GATE / "sealed_role_targets_v2.py"
TRAINING_CORE_SOURCE = HERE / "cpu_training_core.py"
CONVERTER_SOURCE = HERE / "checkpoint_converter.py"
FEATURE_CONTRACT_SOURCE = HERE / "feature_store_contract.py"
EXPECTED_READER_SHA256 = "43cdc72d6bcd7b534d6c661d670b5e718af75bf482aff954ca6507d8d640cdbe"
EXPECTED_OBSERVER_SHA256 = "e70fa563bbbfe6f4e3b8952ede47eeba2c9e1ca0c2507cbc96ce541c0547bbd5"
EXPECTED_SELECTION_SHA256 = "806a2933afebd369040413a17f3e8f2cee939b0e870da7ab728697ba64eed516"
EXPECTED_ENVIRONMENT = {
    "python": "3.11.5",
    "numpy": "1.26.4",
    "torch": "2.7.0",
    "device": "cpu",
    "mps_authorized": False,
}


def _load_after_claim(name: str, path: Path):
    if not CLAIM_PATH.is_file() or CLAIM_PATH.is_symlink():
        raise PermissionError("one-use training claim is absent")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load authorized post-claim module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _consume_claim(training_authorization: Path, activation: dict[str, Any]) -> dict[str, Any]:
    authorization_sha = sha256_file(training_authorization)
    authorization = json.loads(training_authorization.read_text(encoding="utf-8"))
    payload = {
        "schema": "phase2-observer-v4-one-use-training-claim-v1",
        "status": "CONSUMED_BEFORE_TARGET_READER_IMPORT_FEATURE_PROVIDER_IMPORT_OR_OPTIMIZER",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorization_sha256": authorization_sha,
        "authorization_nonce": authorization["nonce"],
        "activation_bindings": activation,
        "target_values_decoded_before_claim": 0,
        "optimizers_constructed_before_claim": 0,
        "fits_before_claim": 0,
        "retry_authorized": False,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(CLAIM_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _directory_fsync(CLAIM_PATH.parent)
    return {**payload, "claim_sha256": hashlib.sha256(encoded).hexdigest()}


def preflight() -> dict[str, Any]:
    if platform.python_version() != EXPECTED_ENVIRONMENT["python"]:
        # The repository-shell interpreter is not the training authority; this
        # check describes rather than imports the exact Torch environment.
        shell_note = f"repository preflight interpreter is {platform.python_version()}"
    else:
        shell_note = "repository preflight interpreter matches training authority"
    bundle = bundle_preflight()
    features = feature_contract_preflight()
    if sha256_file(READER_SOURCE) != EXPECTED_READER_SHA256:
        raise AssertionError("sealed reader source changed")
    if sha256_file(V3 / "observer_model.py") != EXPECTED_OBSERVER_SHA256:
        raise AssertionError("frozen CPU observer changed")
    if sha256_file(V3 / "selection_and_promotion.py") != EXPECTED_SELECTION_SHA256:
        raise AssertionError("frozen selection law changed")
    return {
        "status": "PASS_TARGET_FREE_CPU_TRAINER_SCAFFOLD__HOLD_EXECUTION",
        "bundle_preflight": bundle["status"],
        "feature_contract_preflight": features["status"],
        "training_environment": EXPECTED_ENVIRONMENT,
        "shell_interpreter_note": shell_note,
        "primary": {"cap": 1536, "hidden_channels": 128, "cell_width": 256},
        "control": {"cap": 1792, "hidden_channels": 64, "cell_width": 96},
        "roles": {"observer_fit": 688, "observer_selection": 88},
        "layer1_correction": "structurally_zero",
        "layer2_non_regression": "mandatory_selection_gate",
        "training_claim_exists": CLAIM_PATH.exists(),
        "training_output_exists": OUTPUT_PATH.exists(),
        "launch_authority_present": False,
        "feature_store_present": False,
        "target_reader_imported": False,
        "role_parquet_files_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "mps_used": False,
    }


def execute_authorized_once(
    *,
    independent_post_seal_receipt: Path,
    independent_launch_audit_receipt: Path,
    reader_authorization: Path,
    training_authorization: Path,
    implementation_manifest: Path,
    feature_manifest: Path,
    feature_provider_source: Path,
) -> dict[str, Any]:
    """Execute once after future authority; never called by this V4 freeze."""

    if Path(os.path.realpath(os.sys.executable)) != Path(os.path.realpath(TORCH_PYTHON)):
        raise PermissionError("training must use the exact CPU Torch interpreter")
    observed_environment = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "torch": importlib.metadata.version("torch"),
        "device": "cpu",
        "mps_authorized": False,
    }
    if observed_environment != EXPECTED_ENVIRONMENT:
        raise PermissionError("CPU training interpreter/package authority changed")
    if CLAIM_PATH.exists() or OUTPUT_PATH.exists() or STAGING_PATH.exists():
        raise FileExistsError("training attempt/output/staging path is not fresh")
    feature_metadata = validate_future_manifest(
        manifest_path=feature_manifest,
        provider_source=feature_provider_source,
    )
    activation = validate_activation(
        independent_post_seal_receipt=independent_post_seal_receipt,
        independent_launch_audit_receipt=independent_launch_audit_receipt,
        reader_authorization=reader_authorization,
        training_authorization=training_authorization,
        trainer_source=Path(__file__),
        training_core_source=TRAINING_CORE_SOURCE,
        converter_source=CONVERTER_SOURCE,
        feature_contract_source=FEATURE_CONTRACT_SOURCE,
        feature_provider_source=feature_provider_source,
        implementation_manifest=implementation_manifest,
        feature_manifest=feature_manifest,
    )
    authorization = json.loads(training_authorization.read_text(encoding="utf-8"))
    exact_paths = {
        "claim_path": str(CLAIM_PATH),
        "output_path": str(OUTPUT_PATH),
        "staging_path": str(STAGING_PATH),
    }
    for key, value in exact_paths.items():
        if authorization.get(key) != value:
            raise PermissionError(f"training authorization path mismatch: {key}")
    for key in ("authorized_by", "authorized_at_utc", "nonce"):
        if not isinstance(authorization.get(key), str) or not authorization[key]:
            raise PermissionError(f"training authorization lacks {key}")
    claim = _consume_claim(training_authorization, activation)

    # Every potentially target/fit-capable import is intentionally below the
    # durable O_EXCL claim.  Failure from this point permanently consumes it.
    reader_module = _load_after_claim("v4_authorized_sealed_reader", READER_SOURCE)
    feature_provider = _load_after_claim("v4_authorized_feature_provider", feature_provider_source)
    training_core = _load_after_claim("v4_authorized_cpu_training_core", TRAINING_CORE_SOURCE)
    converter = _load_after_claim("v4_authorized_checkpoint_converter", CONVERTER_SOURCE)
    result = training_core.run_training_session(
        reader_module=reader_module,
        reader_authorization=reader_authorization,
        bundle_manifest=DATA_GATE / "sealed_role_target_bundle" / "manifest.json",
        feature_provider_module=feature_provider,
        feature_manifest=feature_manifest,
    )

    STAGING_PATH.mkdir(mode=0o700)
    receipt = {
        **result.receipt,
        "training_claim_sha256": claim["claim_sha256"],
        "training_authorization_sha256": sha256_file(training_authorization),
        "feature_manifest_sha256": feature_metadata["feature_manifest_sha256"],
        "postfit_parity_status": "HOLD",
        "postfit_common_driver_meter_status": "HOLD",
        "protected_audit_release_authorized": False,
    }
    if result.selected_model is not None:
        checkpoint = converter.export_model_bundle(
            model=result.selected_model,
            centers=result.selected_centers,
            scales=result.selected_scales,
            output_dir=STAGING_PATH / "selected_checkpoint",
            hidden_channels=result.selected_model.hidden_channels,
            cell_width=result.selected_model.cell_width,
            provenance="authorized_v4_cpu_fit_selection__unreleased_pending_postfit_gates",
        )
        receipt["selected_checkpoint"] = {
            "manifest_sha256": checkpoint["manifest_sha256"],
            "tensor_bundle_sha256": checkpoint["torch_tensor_bundle_sha256"],
        }
    receipt_path = STAGING_PATH / "FIT_SELECTION_RECEIPT.json"
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with receipt_path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    for path in STAGING_PATH.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
        elif path.is_dir():
            path.chmod(0o500)
    STAGING_PATH.chmod(0o500)
    _directory_fsync(STAGING_PATH)
    os.replace(STAGING_PATH, OUTPUT_PATH)
    _directory_fsync(OUTPUT_PATH.parent)
    return {
        "status": "COMPLETE_CPU_FIT_SELECTION__HOLD_ALL_POSTFIT_RELEASES",
        "claim_sha256": claim["claim_sha256"],
        "output_receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "postfit_parity_required": True,
        "postfit_common_driver_meter_required": True,
        "audit_or_lockbox_access_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))
