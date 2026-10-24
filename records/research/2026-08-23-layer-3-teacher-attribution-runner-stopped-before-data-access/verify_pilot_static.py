#!/usr/bin/env python3
"""Dataset-free static/math verifier for the staged teacher pilot.

This verifier imports no dataset module, reads no public row, creates no pilot
output, and cannot grant execution authority.  It emits a candidate audit
payload for independent review; the root release must bind the final bytes.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
DESIGN_DIR = HERE.parent
DATA_GATE = DESIGN_DIR.parent / "data_gate"
FOOTHOLD_DIR = DESIGN_DIR.parent / "exact_layer_foothold"
sys.path.insert(0, str(HERE))
import pilot_core as core  # noqa: E402
import build_probe_asset_v1law as probe_builder  # noqa: E402


ZERO_COUNTERS = {
    "target_values_decoded": 0,
    "target_metrics_computed": 0,
    "fits": 0,
    "retries": 0,
    "sweeps": 0,
    "variants": 0,
    "submissions": 0,
}
FROZEN = {
    "predecessor_design": DESIGN_DIR / "DESIGN.md",
    "predecessor_static_resource": DESIGN_DIR / "STATIC_RESOURCE_ESTIMATE.json",
    "predecessor_independent_hold_audit": DESIGN_DIR / "INDEPENDENT_DESIGN_AUDIT.md",
    "sealed_foothold_diagnostic": FOOTHOLD_DIR / "DIAGNOSTIC.md",
    "sealed_foothold_verifier": FOOTHOLD_DIR / "verify_exact_layer_foothold.py",
    "design_v2_preserved": DESIGN_DIR / "DESIGN_V2.md",
    "design_v2_audited_law_correction": DESIGN_DIR / "DESIGN_V2_AUDITED_LAW_CORRECTION.md",
    "static_resource_v2": DESIGN_DIR / "STATIC_RESOURCE_ESTIMATE_V2.json",
    "preregistration_preserved": HERE / "PREREGISTRATION.md",
    "preregistration_v1law_correction": HERE / "PREREGISTRATION_V1LAW_CORRECTION.md",
    "seed_split_authority_v1": HERE / "SEED_SPLIT_AUTHORITY_V1.json",
    "interface_preserved": HERE / "PILOT_INTERFACE.json",
    "interface_v1law": HERE / "PILOT_INTERFACE_V1LAW.json",
    "pilot_resource": HERE / "PILOT_RESOURCE_ESTIMATE.json",
    "probe_generator_preserved": HERE / "build_probe_asset.py",
    "probe_spec_preserved": HERE / "PROBE_SPEC.json",
    "probe_bytes_preserved": HERE / "probe32.f32le",
    "probe_generator_v1law": HERE / "build_probe_asset_v1law.py",
    "probe_spec_v1law": HERE / "PROBE_SPEC_V1LAW.json",
    "probe_bytes_v1law": HERE / "probe32_v1law.f32le",
    "pilot_core": HERE / "pilot_core.py",
    "pilot_driver": HERE / "run_pilot.py",
    "static_verifier": HERE / "verify_pilot_static.py",
    "receipts_parent": HERE / "receipts" / "README.md",
    "runs_parent": HERE / "runs" / "README.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_driver_without_execution():
    spec = importlib.util.spec_from_file_location("teacher_pilot_driver_audit", HERE / "run_pilot.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load driver spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_probe() -> dict[str, object]:
    spec = json.loads((HERE / "PROBE_SPEC_V1LAW.json").read_text(encoding="utf-8"))
    frozen = (HERE / "probe32_v1law.f32le").read_bytes()
    rebuilt = probe_builder.construct().tobytes(order="C")
    if rebuilt != frozen:
        raise AssertionError("probe reconstruction is not byte-exact")
    if hashlib.sha256(frozen).hexdigest() != spec["sha256"]:
        raise AssertionError("probe spec hash mismatch")
    indices, signs = probe_builder.column_schedule()
    if indices != spec["column_indices"] or signs != spec["column_signs"]:
        raise AssertionError("probe column schedule mismatch")
    probe = np.frombuffer(frozen, dtype="<f4").reshape(1024, 32)
    if not np.array_equal(probe.astype(np.float64).T @ probe, np.eye(32)):
        raise AssertionError("probe orthogonality failed")
    return {
        "payload_sha256": spec["sha256"],
        "generator_sha256": sha(HERE / "build_probe_asset_v1law.py"),
        "byte_exact_reconstruction": True,
        "orthonormal": True,
    }


def fake_host_teacher() -> tuple[core.HostTape, core.TeacherTape]:
    depth, width, rank = 2, 3, 2
    host = core.HostTape(
        pre_mean=np.full((depth, width), 0.1),
        pre_covariance=np.stack([np.eye(width), np.eye(width)]),
        pre_variance=np.full((depth, width), 1.1),
        pre_K3=np.full((depth, width), 0.03),
        pre_K21_probe=np.zeros((depth, width, rank)),
        pre_K4=np.full((depth, width), -0.02),
        post_mean=np.zeros((depth, width)),
        post_covariance=np.stack([np.eye(width), np.eye(width)]),
    )
    teacher = core.TeacherTape(
        count=8,
        pre_mean=np.full((depth, width), 0.2),
        pre_covariance=np.stack([2.0 * np.eye(width), 2.0 * np.eye(width)]),
        pre_variance=np.full((depth, width), 2.0),
        pre_K3=np.full((depth, width), -0.04),
        pre_K4=np.full((depth, width), 0.05),
        pre_K21=np.zeros((depth, width, width)),
        post_mean=np.zeros((depth, width)),
        post_covariance=np.stack([2.0 * np.eye(width), 2.0 * np.eye(width)]),
    )
    return host, teacher


def check_all_arms() -> dict[str, object]:
    host, teacher = fake_host_teacher()
    observed = core.factorial_predictions(host, teacher)
    if observed.shape != (16, 2, 3):
        raise AssertionError("factorial output shape changed")
    for index, mask in enumerate(core.ARM_MASKS):
        fields = []
        for bit, host_value, teacher_value in zip(
            mask,
            (host.pre_mean, host.pre_variance, host.pre_K3, host.pre_K4),
            (
                teacher.pre_mean,
                teacher.pre_variance,
                teacher.pre_K3,
                teacher.pre_K4,
            ),
            strict=True,
        ):
            fields.append(teacher_value if bit == "1" else host_value)
        literal = core.scalar_edgeworth4(*fields)
        if not np.array_equal(observed[index], literal):
            raise AssertionError(f"factorial mask mismatch: {mask}")
    if np.array_equal(observed[0], observed[-1]):
        raise AssertionError("host and all-teacher arms unexpectedly identical")
    return {"masks": list(core.ARM_MASKS), "field_order": list(core.FIELD_ORDER)}


def check_pooling_and_K21() -> dict[str, object]:
    width, depth = 3, 3
    weights = np.stack([np.eye(width), np.eye(width), np.eye(width)]).astype(np.float32)
    first_inputs = np.array(
        [[-2.0, 0.5, 1.0], [1.0, -1.0, 2.0], [0.25, 3.0, -0.5]],
        dtype=np.float32,
    )
    second_inputs = np.array(
        [[4.0, -0.25, 0.75], [-1.5, 2.5, 1.25]], dtype=np.float32
    )
    first = core.RawTape.zeros(depth, width)
    second = core.RawTape.zeros(depth, width)
    combined = core.RawTape.zeros(depth, width)
    core.accumulate_cpu_block(first, weights, first_inputs)
    core.accumulate_cpu_block(second, weights, second_inputs)
    core.accumulate_cpu_block(
        combined, weights, np.concatenate([first_inputs, second_inputs], axis=0)
    )
    pooled = core.merge_raw_tapes([first, second])
    for label in ("pre_powers", "post_sum", "post_raw2", "pre_sq_cross"):
        if not np.array_equal(getattr(pooled, label), getattr(combined, label)):
            raise AssertionError(f"raw pooling changed {label}")
    teacher = core.normalize_raw_tape(pooled, weights)
    layer2 = np.maximum(
        np.concatenate([first_inputs, second_inputs], axis=0), 0.0
    ).astype(np.float64)
    exact_mean = np.full(width, 1.0 / np.sqrt(2.0 * np.pi))
    _, _, exact_raw2 = core.zero_mean_gaussian_relu_moments(np.eye(width))
    raw_sq_cross = (layer2 * layer2).T @ layer2 / layer2.shape[0]
    layer2_literal_K21 = (
        raw_sq_cross
        - np.diag(exact_raw2)[:, None] * exact_mean[None, :]
        - 2.0 * exact_mean[:, None] * exact_raw2
        + 2.0 * (exact_mean * exact_mean)[:, None] * exact_mean[None, :]
    )
    diagonal = np.arange(width)
    layer2_literal_K21[diagonal, diagonal] = teacher.pre_K3[1]
    if not np.allclose(
        teacher.pre_K21[1], layer2_literal_K21, rtol=1e-14, atol=1e-14
    ):
        raise AssertionError("exact-source layer2 K21 expansion mismatch")
    sampled_mean = np.mean(layer2, axis=0)
    centered = layer2 - sampled_mean
    layer3_literal_K21 = (centered * centered).T @ centered / layer2.shape[0]
    if not np.allclose(
        teacher.pre_K21[2], layer3_literal_K21, rtol=1e-14, atol=1e-14
    ):
        raise AssertionError("bank-source layer3 K21 orientation mismatch")
    if not np.array_equal(np.diag(teacher.pre_K21[0]), teacher.pre_K3[0]):
        raise AssertionError("Gaussian source K21 diagonal mismatch")
    if not np.allclose(np.diag(teacher.pre_K21[1]), teacher.pre_K3[1], atol=0.0):
        raise AssertionError("K21 diagonal convention mismatch")
    return {
        "pooled_count": pooled.count,
        "layer1_teacher_source_K21_zero": bool(
            np.count_nonzero(teacher.pre_K21[0]) == 0
        ),
        "layer2_exact_source_K21_max_error": float(
            np.max(np.abs(teacher.pre_K21[1] - layer2_literal_K21))
        ),
        "layer3_bank_source_K21_max_error": float(
            np.max(np.abs(teacher.pre_K21[2] - layer3_literal_K21))
        ),
        "layer2_pre_mean_exact_error": float(
            np.max(np.abs(teacher.pre_mean[1] - exact_mean))
        ),
        "layer2_pre_covariance_exact_error": float(
            np.max(
                np.abs(
                    teacher.pre_covariance[1]
                    - core.zero_mean_gaussian_relu_moments(np.eye(width))[1]
                )
            )
        ),
    }


def check_actions() -> dict[str, object]:
    rng = np.random.default_rng(20260823)
    width, rank = 7, 3
    a = rng.standard_normal((width, width))
    delta = 0.5 * (a + a.T)
    teacher = rng.standard_normal((width, width))
    teacher = teacher @ teacher.T + np.eye(width)
    host = teacher + delta
    weight = rng.standard_normal((width, width))
    probe, _ = np.linalg.qr(rng.standard_normal((width, rank)))
    action = core.covariance_action_split(host, teacher, weight, probe)
    literal = weight.T @ delta @ weight
    diagonal_literal = (weight * weight).T @ np.diag(delta)
    if not np.allclose(action.total, np.diag(literal), rtol=1e-13, atol=1e-13):
        raise AssertionError("total covariance action mismatch")
    if not np.allclose(action.diagonal, diagonal_literal, rtol=1e-13, atol=1e-13):
        raise AssertionError("diagonal covariance action mismatch")
    if not np.allclose(
        action.offdiagonal, np.diag(literal) - diagonal_literal, rtol=1e-13, atol=1e-13
    ):
        raise AssertionError("off-diagonal covariance action mismatch")
    return {
        "total_max_error": float(np.max(np.abs(action.total - np.diag(literal)))),
        "split_identity_max_error": float(
            np.max(np.abs(action.diagonal + action.offdiagonal - action.total))
        ),
    }


def check_raw_bank_dependency_binding(driver) -> dict[str, object]:
    """Exercise the exact raw-bank/provenance dependency chain in scratch space."""

    level = 16
    weights = np.eye(2, dtype=np.float32)[None, :, :]
    inputs = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
    tape = core.RawTape.zeros(1, 2)
    core.accumulate_cpu_block(tape, weights, inputs)
    with tempfile.TemporaryDirectory(prefix="teacher-pilot-static-") as temporary:
        root = Path(temporary)
        observed: dict[str, object] = {}
        for bank in ("A", "B"):
            manifest = driver.seal_raw_tape(
                root / "rqmc" / f"bank_{bank}" / f"N{level}", tape.copy()
            )
            provenance = {
                "schema": "phase2-teacher-scramble-provenance-v1",
                "sampler": "rqmc",
                "bank": bank,
                "backend": "cpu",
                "replicates": [],
                "levels": {str(level): manifest},
                **ZERO_COUNTERS,
            }
            provenance_sha = driver.atomic_json(
                root / "rqmc" / f"bank_{bank}_provenance.json", provenance
            )
            binding = driver.raw_bank_binding(root, "rqmc", bank, level)
            if binding["manifest_file_sha256"] != manifest["manifest_sha256"]:
                raise AssertionError("derived/raw manifest file binding failed")
            if binding["provenance_file_sha256"] != provenance_sha:
                raise AssertionError("derived/raw provenance file binding failed")
            if binding["raw_tape_sha256"] != manifest["raw_tape_sha256"]:
                raise AssertionError("derived/raw semantic binding failed")
            if binding["count"] != tape.count:
                raise AssertionError("derived/raw count binding failed")
            observed[bank] = binding
    return {
        "banks": sorted(observed),
        "manifest_file_hash_bound": True,
        "provenance_file_hash_bound": True,
        "semantic_and_array_hashes_bound": True,
        "scratch_only": True,
    }


def check_nested_streams() -> dict[str, object]:
    expected_preimage = b"p2-teacher-v1|v2-phase2|-17|rqmc|3"
    if core.stream_seed_preimage(-17, "rqmc", 3) != expected_preimage:
        raise AssertionError("audited signed-decimal v1 seed preimage changed")
    authority = json.loads(
        (HERE / "SEED_SPLIT_AUTHORITY_V1.json").read_text(encoding="utf-8")
    )
    if authority["full_split"]["preimage_ascii_prefix"] != "p2-teacher-v1|v2-phase2|":
        raise AssertionError("audited full split prefix changed")
    if authority["pilot_role"]["full_split_roles_available"] or authority[
        "pilot_role"
    ]["target_roles_available"]:
        raise AssertionError("mini pilot acquired a full or target role")
    result = {}
    for sampler in ("rqmc", "iid"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            long_blocks = list(
                core.iter_normal_blocks(
                    width=4,
                    network_seed=17,
                    sampler=sampler,
                    replicate=3,
                    count=16,
                    batch_size=8,
                )
            )
            short_blocks = list(
                core.iter_normal_blocks(
                    width=4,
                    network_seed=17,
                    sampler=sampler,
                    replicate=3,
                    count=8,
                    batch_size=8,
                )
            )
        if not np.array_equal(long_blocks[0], short_blocks[0]):
            raise AssertionError(f"{sampler} prefix is not nested")
        seeds = [core.stream_seed(17, sampler, replicate) for replicate in range(8)]
        if len(set(seeds)) != 8:
            raise AssertionError(f"{sampler} replicate seed collision")
        result[sampler] = {
            "nested_prefix_sha256": core.payload_sha256(long_blocks[0]),
            "unique_seeds": True,
        }
    if core.stream_seed(17, "rqmc", 0) == core.stream_seed(17, "iid", 0):
        raise AssertionError("RQMC and IID seeds collided")
    result["audited_law"] = {
        "negative_seed_preimage": expected_preimage.decode("ascii"),
        "full_split_prefix": authority["full_split"]["preimage_ascii_prefix"],
        "pilot_role": authority["pilot_role"]["name"],
        "full_and_target_roles": 0,
    }
    return result


def check_driver_gate() -> dict[str, object]:
    source = (HERE / "run_pilot.py").read_text(encoding="utf-8")
    for executable_path in (
        HERE / "pilot_core.py",
        HERE / "run_pilot.py",
        HERE / "PILOT_INTERFACE_V1LAW.json",
        HERE / "SEED_SPLIT_AUTHORITY_V1.json",
        HERE / "build_probe_asset_v1law.py",
        HERE / "PROBE_SPEC_V1LAW.json",
    ):
        if "p2-teacher-v2" in executable_path.read_text(encoding="utf-8", errors="strict"):
            raise AssertionError(
                f"retired v2 seed/probe literal reached executable binding: {executable_path.name}"
            )
    if core.LITERAL != "p2-teacher-v1":
        raise AssertionError("core namespace is not audited v1")
    tree = ast.parse(source)
    forbidden = json.loads((DATA_GATE / "authority.json").read_text(encoding="utf-8"))[
        "target_seal"
    ]["forbidden_value_columns"]
    for literal in forbidden:
        if literal in source or literal in (HERE / "pilot_core.py").read_text(encoding="utf-8"):
            raise AssertionError(f"forbidden physical field literal in pilot source: {literal}")
    if "iter_target_blind_full" in source or "dataset_split\": \"full" in source:
        raise AssertionError("pilot source exposes a full-split iterator role")
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    if any("target_blind_mini" in ast.unparse(node) for node in imports):
        raise AssertionError("dataset loader imported statically")
    execute = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    execute_source = ast.get_source_segment(source, execute)
    claim_position = execute_source.index("write_one_use_claim(")
    row_position = execute_source.index("load_cases_after_claim(")
    output_position = execute_source.index("output.mkdir(")
    if not claim_position < row_position < output_position:
        raise AssertionError("claim/import/output chronology changed")
    derive = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "derive_and_seal_level"
    )
    derive_source = ast.get_source_segment(source, derive)
    binding_position = derive_source.index("source_raw_banks =")
    load_position = derive_source.index("raw_a = load_raw_tape(")
    manifest_position = derive_source.index('"source_raw_banks": source_raw_banks')
    if not binding_position < load_position < manifest_position:
        raise AssertionError("derived/raw dependency binding chronology changed")
    if "os.O_EXCL" not in source or "os.fsync" not in source:
        raise AssertionError("durable one-use claim primitives missing")
    if (HERE / "PILOT_RELEASE.json").exists():
        raise AssertionError("release unexpectedly present during static audit")
    driver = import_driver_without_execution()
    report = driver.preflight()
    if report["dataset_rows_decoded"] != 0 or report["dataset_module_imported"]:
        raise AssertionError("zero-row preflight contract failed")
    if report["status"] != "BLOCKED_PENDING_AUDITS_AND_ROOT_RELEASE":
        raise AssertionError("execution is not mechanically blocked")
    safe_reading = {
        "process_peak_rss_bytes": 100,
        "mps_current_allocated_bytes": 20,
        "mps_driver_allocated_bytes": 30,
        "combined_bytes": 150,
    }
    driver.enforce_resource_limit(
        "static_safe_injection", limit_bytes=151, reading=safe_reading
    )
    guard_raised = False
    try:
        driver.enforce_resource_limit(
            "static_abort_injection", limit_bytes=150, reading=safe_reading
        )
    except MemoryError:
        guard_raised = True
    if not guard_raised:
        raise AssertionError("resource guard did not abort at the exact threshold")
    if not (HERE / "receipts" / "README.md").is_file() or not (
        HERE / "runs" / "README.md"
    ).is_file():
        raise AssertionError("frozen execution parents are absent")
    return {
        "claim_before_dataset_import": True,
        "claim_before_output_creation": True,
        "O_CREAT_O_EXCL_and_fsync": True,
        "release_absent": True,
        "frozen_receipts_and_runs_parents": True,
        "derived_manifests_bind_exact_raw_banks": True,
        "resource_guard_safe_and_abort_injections": True,
        "zero_row_preflight_status": report["status"],
    }


def main() -> None:
    frozen_sha = {label: sha(path) for label, path in FROZEN.items()}
    expected_history = {
        "predecessor_design": "bb8ca6d673fef1c660c695e8baad14d9739a53efe9f4e22c7c52a15d4df25dd0",
        "predecessor_static_resource": "28d46b24db3eef68fce4f0fb96d66b59c774a647d75a0e6c9586cc2968b39da6",
        "predecessor_independent_hold_audit": "96254da6f8e05f494eef2b8e3fe3355e1b7427142f8d5c9c5a706eab28b903c4",
    }
    if any(frozen_sha[label] != digest for label, digest in expected_history.items()):
        raise AssertionError("preserved predecessor audit chain changed")
    interface = json.loads(
        (HERE / "PILOT_INTERFACE_V1LAW.json").read_text(encoding="utf-8")
    )
    resource_plan = json.loads(
        (HERE / "PILOT_RESOURCE_ESTIMATE.json").read_text(encoding="utf-8")
    )
    if interface["access_counters"] != ZERO_COUNTERS:
        raise AssertionError("interface counters changed")
    if resource_plan["peak_memory"]["conservative_sequential_process_peak_bytes"] >= resource_plan["peak_memory"]["abort_threshold_bytes"]:
        raise AssertionError("static peak estimate is not below abort threshold")
    report = {
        "schema": "phase2-teacher-pilot-static-verification-v1",
        "status": "PASS_LOCAL_STATIC_VERIFICATION__NOT_EXECUTION_AUTHORITY",
        "frozen_sha256": frozen_sha,
        "core_self_test": core.small_exact_self_test(),
        "probe": check_probe(),
        "factorial": check_all_arms(),
        "pooling_and_K21": check_pooling_and_K21(),
        "covariance_actions": check_actions(),
        "raw_bank_dependency_binding": check_raw_bank_dependency_binding(
            import_driver_without_execution()
        ),
        "nested_streams": check_nested_streams(),
        "driver_gate": check_driver_gate(),
        "resource_static": {
            "three_case_flop_upper_bound": resource_plan["nested_sampling"][
                "total_upper_bound_flops_all_three_cases"
            ],
            "required_free_workspace": resource_plan["persistent_bytes"][
                "required_free_workspace"
            ],
            "conservative_peak_bytes": resource_plan["peak_memory"][
                "conservative_sequential_process_peak_bytes"
            ],
            "abort_threshold_bytes": resource_plan["peak_memory"][
                "abort_threshold_bytes"
            ],
        },
        "MPS_runtime_tested": False,
        "dataset_module_imported": False,
        "dataset_rows_decoded": 0,
        **ZERO_COUNTERS,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
