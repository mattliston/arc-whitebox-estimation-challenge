#!/usr/bin/env python3
"""Verify retained hashes and invariants for a synthetic meter run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def _workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("could not locate workspace root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_fingerprint(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for relative in sorted(distribution.files or (), key=str):
        if str(relative).endswith((".pyc", ".pyo")):
            continue
        resolved = Path(distribution.locate_file(relative))
        if not resolved.is_file():
            continue
        payload = resolved.read_bytes()
        encoded_name = str(relative).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        file_count += 1
        total_bytes += len(payload)
    return {
        "version": distribution.version,
        "installed_distribution_sha256": digest.hexdigest(),
        "hashed_file_count": file_count,
        "hashed_bytes": total_bytes,
    }


def _resolve_recorded_path(recorded: str, workspace: Path) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else workspace / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing receipt.json and raw artifacts")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = _workspace_root()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = workspace / run_dir
    receipt_path = run_dir / "receipt.json"
    receipt: dict[str, Any] = json.loads(receipt_path.read_text())
    checks: dict[str, bool] = {}

    checks["receipt_schema"] = (
        receipt.get("schema") == "phase2-target-blind-synthetic-meter-receipt-v1"
    )
    checks["receipt_status"] = receipt.get("status", "").startswith("PASS_")
    for name, record in receipt.get("raw_artifacts", {}).items():
        artifact = run_dir / name
        checks[f"artifact_exists:{name}"] = artifact.is_file()
        checks[f"artifact_sha256:{name}"] = artifact.is_file() and _sha256(artifact) == record.get(
            "sha256"
        )
        checks[f"artifact_bytes:{name}"] = artifact.is_file() and artifact.stat().st_size == record.get(
            "bytes"
        )

    estimator_record = receipt["estimator"]
    estimator = _resolve_recorded_path(estimator_record["path"], workspace)
    checks["estimator_exists"] = estimator.is_file()
    checks["estimator_sha256"] = estimator.is_file() and _sha256(estimator) == estimator_record.get(
        "sha256"
    )
    provenance = receipt["provenance"]
    driver = _resolve_recorded_path(provenance["meter_driver_path"], workspace)
    verifier = _resolve_recorded_path(provenance["receipt_verifier_path"], workspace)
    lock = _resolve_recorded_path(provenance["requirements_lock_path"], workspace)
    checks["meter_driver_sha256"] = driver.is_file() and _sha256(driver) == provenance.get(
        "meter_driver_sha256"
    )
    checks["receipt_verifier_sha256"] = verifier.is_file() and _sha256(verifier) == provenance.get(
        "receipt_verifier_sha256"
    )
    checks["requirements_lock_sha256"] = lock.is_file() and _sha256(lock) == provenance.get(
        "requirements_lock_sha256"
    )
    for name, recorded in receipt.get("software", {}).items():
        checks[f"installed_distribution:{name}"] = _distribution_fingerprint(name) == recorded

    site_root = Path(importlib.metadata.distribution("whestbench").locate_file(""))
    for relative, recorded_hash in provenance.get("installed_source_sha256", {}).items():
        source = site_root / relative
        checks[f"installed_source:{relative}"] = source.is_file() and _sha256(source) == recorded_hash

    raw_report = json.loads((run_dir / "raw_whest_stdout.json").read_text())
    invocation = json.loads((run_dir / "invocation.json").read_text())
    checks["embedded_invocation_matches_artifact"] = receipt.get("invocation") == invocation
    checks["target_blind_no_dataset_argument"] = "--dataset" not in invocation.get("whest_argv", [])
    checks["target_blind_scope"] = (
        receipt.get("scope", {}).get("public_phase2_target_or_truth_values_accessed") == 0
    )
    checks["raw_run_geometry"] = raw_report.get("run_config", {}).get("width") == 1024 and raw_report.get(
        "run_config", {}
    ).get("depth") == 16
    checks["raw_run_no_failures"] = raw_report.get("results", {}).get("n_failed_mlps") == 0
    output_contract = receipt.get("measured", {}).get("output_contract", {})
    checks["output_contract"] = (
        output_contract.get("shape") == [16, 1024]
        and output_contract.get("materialized_scored_dtype") == "float32"
        and output_contract.get("all_values_finite") is True
    )
    raw_flops = [
        int(record.get("flops_used", 0))
        for record in raw_report.get("results", {}).get("per_mlp", [])
    ]
    stored_flops = receipt.get("measured", {}).get("flops_used_per_mlp", [])
    checks["raw_flops_match_receipt"] = raw_flops == stored_flops
    stored_floor = receipt.get("measured", {}).get("score_floor", {})
    expected_floor_records = [
        {
            "mlp_index": index,
            "flops_used": flops_used,
            "within_0p1_score_floor": 10 * flops_used <= 2**41,
            "integer_flop_margin_to_floor_max": (2**41 // 10) - flops_used,
        }
        for index, flops_used in enumerate(raw_flops)
    ]
    checks["exact_score_floor_classification"] = (
        stored_floor.get("exact_threshold_expression") == "flops_used * 10 <= 2^41"
        and stored_floor.get("max_integer_flops_inclusive") == 2**41 // 10
        and stored_floor.get("per_mlp") == expected_floor_records
        and stored_floor.get("all_mlps_within_floor")
        == all(record["within_0p1_score_floor"] for record in expected_floor_records)
    )
    rss_samples = [
        json.loads(line)
        for line in (run_dir / "raw_process_tree_rss.jsonl").read_text().splitlines()
        if line.strip()
    ]
    memory = receipt.get("measured", {}).get("memory", {})
    sampler = memory.get("sampler", {})
    checks["rss_samples_present"] = len(rss_samples) > 0
    checks["rss_sample_count"] = sampler.get("sample_count") == len(rss_samples)
    rss_fields = (
        "root_rss_bytes",
        "direct_children_aggregate_rss_bytes",
        "descendants_aggregate_rss_bytes",
        "tree_aggregate_rss_bytes",
    )
    for field in rss_fields:
        checks[f"rss_peak:{field}"] = sampler.get(f"peak_{field}") == max(
            (int(sample[field]) for sample in rss_samples), default=0
        )
    checks["rss_sample_internal_sums"] = all(
        int(sample["tree_aggregate_rss_bytes"])
        == int(sample["root_rss_bytes"]) + int(sample["descendants_aggregate_rss_bytes"])
        and int(sample["root_rss_bytes"])
        == sum(
            int(process["rss_bytes"])
            for process in sample["processes"]
            if int(process["depth"]) == 0
        )
        and int(sample["descendants_aggregate_rss_bytes"])
        == sum(
            int(process["rss_bytes"])
            for process in sample["processes"]
            if int(process["depth"]) > 0
        )
        for sample in rss_samples
    )
    checks["rss_descendant_peak_observed"] = int(
        sampler.get("peak_descendants_aggregate_rss_bytes", 0)
    ) > 0
    checks["rss_descendant_peak_below_limit"] = int(
        sampler.get("peak_descendants_aggregate_rss_bytes", 0)
    ) < int(memory.get("memory_limit_bytes", 0))
    checks["rss_sampler_error_free"] = sampler.get("sampling_errors") == []
    checks["stored_assertions_all_true"] = all(receipt.get("assertions", {}).values())

    passed = all(checks.values())
    print(
        json.dumps(
            {
                "status": "PASS_METER_RECEIPT_VERIFICATION" if passed else "FAIL_METER_RECEIPT_VERIFICATION",
                "run_dir": str(run_dir),
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
