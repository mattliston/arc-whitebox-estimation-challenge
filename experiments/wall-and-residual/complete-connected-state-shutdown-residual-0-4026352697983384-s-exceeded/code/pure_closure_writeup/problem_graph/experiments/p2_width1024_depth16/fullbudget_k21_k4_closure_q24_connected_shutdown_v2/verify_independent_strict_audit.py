#!/usr/bin/env python3
"""Independent target-free strict audit for frozen Q24-CS2 source bytes."""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "estimator.py"
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PREREGISTER = HERE / "PREREGISTER.md"
RUNNER = HERE / "run_source_probe.py"
RESULT = HERE / "runs" / "source_probe_v1" / "results.json"
SOURCE_HOLD = HERE / "SOURCE_HOLD_V2.json"
FIXTURE_HELPER = ROOT / "q24_two_point_moment_repair_probe_v1" / "run_probe.py"
DESIGN_PREREGISTER = (
    ROOT / "q24_connected_shutdown_pairing_complete_design_v1" / "PREREGISTER.md"
)

SOURCE_SHA256 = "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
PREREGISTER_SHA256 = "e83d1bfd6578556853af6ad1b6a9de3f35a2f45869a1262b99241d12653928c0"
RUNNER_SHA256 = "d9e9e423103037058254d8d06e837fb8426f46ab6b2898ea47c7dab91c448aea"
RESULT_SHA256 = "c4c8231c17a68f89916fea5fdf87f9dd0fa52be34ad47d03fbe3ab4d0e21bb7f"
SOURCE_HOLD_SHA256 = "c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c"
FIXTURE_HELPER_SHA256 = "2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d"
DESIGN_PREREGISTER_SHA256 = "3f95c1920ff609265e7d50567c713a891a6ecc315da801ef3cc5723eda37541a"
PROBE_BUDGET = 10**15
PHASE2_BUDGET = 1 << 41
PARENT_LITERAL_FLOPS = 2_131_280_786_389
PARENT_LITERAL_CALLS = 7_208


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def import_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(names)


def operation_count(budget) -> int:
    return sum(
        int(value.get("calls", 0))
        for value in budget.summary_dict(by_namespace=False)
        .get("operations", {}).values()
    )


def exact_margins(raw: tuple[float, float, float, float]) -> tuple[Fraction, Fraction]:
    r1, r2, r3, r4 = (Fraction.from_float(float(value)) for value in raw)
    r1sq = r1 * r1
    r2sq = r2 * r2
    margin_a = r1 * r3 - r2sq
    variance = r2 - r1sq
    centered = r3 - r1 * r2
    margin_b = (r4 - r2sq) * variance - centered * centered
    return margin_a, margin_b


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix).copy()
    np.fill_diagonal(result, 0.0)
    return result


def incident_nonzero(value: np.ndarray, invalid: np.ndarray) -> int:
    array = np.asarray(value)
    if array.ndim == 1:
        return int(np.count_nonzero(array[invalid]))
    valid = ~invalid
    valid_plane = valid[:, None] & valid[None, :]
    return int(np.count_nonzero(np.where(~valid_plane, array, 0.0)))


def cp3_read(lam: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if lam.size == 0:
        return np.zeros((p.shape[0], p.shape[0]), dtype=np.float64)
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    lam = np.asarray(lam, dtype=np.float32)
    first = np.einsum("ir,jr->ij", p * p, q * lam[None, :])
    second = np.einsum("ir,jr->ij", p * q, p * lam[None, :])
    return np.asarray(first + np.float32(2.0) * second, dtype=np.float64)


def cp4_planes(lam: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = int(u.shape[0])
    if lam.size == 0:
        return (
            np.zeros(width, dtype=np.float64),
            np.zeros((width, width), dtype=np.float64),
            np.zeros((width, width), dtype=np.float64),
        )
    lam = np.asarray(lam, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    u2 = u * u
    directed = np.einsum(
        "ir,jr->ij", u2 * u * lam[None, :], u
    ).astype(np.float64)
    paired = np.einsum(
        "ir,jr->ij", u2 * lam[None, :], u2
    ).astype(np.float64)
    return np.diag(directed).copy(), off_diagonal(directed), off_diagonal(paired)


def push_cubic(star: np.ndarray, weight: np.ndarray) -> np.ndarray:
    star = np.asarray(star, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    weight2 = weight * weight
    diagonal = np.diag(star)
    directed = off_diagonal(star)
    transported = directed @ weight
    pushed_diagonal = np.sum(
        diagonal[:, None] * weight2 * weight
        + np.float32(3.0) * weight2 * transported,
        axis=0,
    )
    pushed = (
        weight2.T @ (diagonal[:, None] * weight)
        + np.float32(2.0) * (weight * transported).T @ weight
        + weight2.T @ transported
    )
    np.fill_diagonal(pushed, pushed_diagonal)
    return pushed.astype(np.float64)


def push_quartic(
    star: np.ndarray,
    pairing_alpha: float,
    weight: np.ndarray,
    pre_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    star = np.asarray(star, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    weight2 = weight * weight
    weight3 = weight2 * weight
    diagonal = np.diag(star)
    directed = off_diagonal(star)
    transported = directed @ weight
    mixed = weight2.T @ (weight * transported)
    pushed_d = np.sum(
        diagonal[:, None] * weight2 * weight2
        + np.float32(4.0) * weight3 * transported,
        axis=0,
    ).astype(np.float64)
    pushed_a = (
        weight3.T @ (diagonal[:, None] * weight)
        + weight3.T @ transported
        + np.float32(3.0) * (weight2 * transported).T @ weight
    ).astype(np.float64)
    pushed_b = (
        weight2.T @ (diagonal[:, None] * weight2)
        + np.float32(2.0) * (mixed + mixed.T)
    ).astype(np.float64)
    covariance = np.asarray(pre_covariance, dtype=np.float64)
    variance = np.diag(covariance)
    alpha = np.float64(pairing_alpha)
    return (
        pushed_d + 3.0 * alpha * variance * variance,
        off_diagonal(
            pushed_a + 3.0 * alpha * variance[:, None] * covariance
        ),
        off_diagonal(
            pushed_b
            + alpha
            * (np.outer(variance, variance) + 2.0 * covariance * covariance)
        ),
    )


def signed_permutation(width: int) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(width)
    destination = (5 * source + 3) % width
    signs = np.where(source % 2 == 0, 1.0, -1.0).astype(np.float32)
    weight = np.zeros((width, width), dtype=np.float32)
    weight[source, destination] = signs
    return weight, destination


def mapped_invalid(invalid: np.ndarray, destination: np.ndarray) -> np.ndarray:
    result = np.zeros_like(invalid)
    result[destination] = invalid
    return result


def current_carrier_gate(record: dict[str, object]) -> dict[str, int | bool]:
    invalid = np.asarray(record["invalid"], dtype=bool)
    valid = ~invalid
    plane = valid[:, None] & valid[None, :]
    targets = record["targets"]
    k21, k4, k31, k22 = (np.asarray(value) for value in targets)
    cp_post = np.asarray(record["cp_post"])
    provider = tuple(np.asarray(value) for value in record["provider"])
    cubic = np.asarray(record["cubic"], dtype=np.float64)
    quartic = np.asarray(record["quartic"], dtype=np.float64)
    alpha = float(record["pairing_alpha"])
    covariance = np.asarray(record["covariance"], dtype=np.float64)
    variance = np.asarray(record["variance"], dtype=np.float64)
    lam3, p3, q3 = (np.asarray(value) for value in record["cp3_state"])
    lam4, u4 = (np.asarray(value) for value in record["cp4_state"])
    actual_cp3 = cp3_read(lam3, p3, q3)
    actual_cp4 = cp4_planes(lam4, u4)
    pair_plane = off_diagonal(
        np.outer(variance, variance) + 2.0 * covariance * covariance
    )
    reconstructed_k21 = actual_cp3 + cubic
    reconstructed_k4 = actual_cp4[0] + np.diag(quartic) + 3.0 * alpha * variance * variance
    reconstructed_k31 = actual_cp4[1] + off_diagonal(quartic) + (
        3.0 * alpha * variance[:, None] * covariance
    )
    reconstructed_k22 = actual_cp4[2] + alpha * pair_plane
    finite_values = (
        k21, k4, k31, k22, cp_post, *provider, cubic, quartic,
        actual_cp3, *actual_cp4, reconstructed_k21, reconstructed_k4,
        reconstructed_k31, reconstructed_k22,
    )
    return {
        "target_incident_nonzero": (
            int(np.count_nonzero(np.where(~plane, k21, 0.0)))
            + int(np.count_nonzero(k4[invalid]))
            + int(np.count_nonzero(np.where(~plane, k31, 0.0)))
            + int(np.count_nonzero(np.where(~plane, k22, 0.0)))
        ),
        "gate_or_birth_invalid_nonzero": int(record["gate_birth_violation"]),
        "pairing_alpha_nonzero": int(bool(invalid.any()) and alpha != 0.0),
        "analytic_cp3_incident_nonzero": incident_nonzero(cp_post, invalid),
        "actual_cp3_incident_nonzero": incident_nonzero(actual_cp3, invalid),
        "returned_cp4_incident_nonzero": (
            incident_nonzero(provider[0], invalid)
            + incident_nonzero(provider[1], invalid)
            + incident_nonzero(provider[2], invalid)
        ),
        "actual_cp4_incident_nonzero": (
            incident_nonzero(actual_cp4[0], invalid)
            + incident_nonzero(actual_cp4[1], invalid)
            + incident_nonzero(actual_cp4[2], invalid)
        ),
        "residual_incident_nonzero": (
            incident_nonzero(cubic, invalid)
            + incident_nonzero(quartic, invalid)
        ),
        "reconstructed_incident_nonzero": (
            incident_nonzero(reconstructed_k21, invalid)
            + incident_nonzero(reconstructed_k4, invalid)
            + incident_nonzero(reconstructed_k31, invalid)
            + incident_nonzero(reconstructed_k22, invalid)
        ),
        "all_fields_finite": all(np.isfinite(value).all() for value in finite_values),
    }


def one_push_gate(record: dict[str, object], weight: np.ndarray, pushed_invalid: np.ndarray) -> dict[str, int | bool]:
    lam3, p3, q3 = (np.asarray(value) for value in record["cp3_state"])
    lam4, u4 = (np.asarray(value) for value in record["cp4_state"])
    cubic = np.asarray(record["cubic"], dtype=np.float32)
    quartic = np.asarray(record["quartic"], dtype=np.float32)
    alpha = float(record["pairing_alpha"])
    covariance = np.asarray(record["covariance"], dtype=np.float64)
    weight32 = np.asarray(weight, dtype=np.float32)
    weight64 = weight32.astype(np.float64)
    pushed_p = weight32.T @ np.asarray(p3, dtype=np.float32)
    pushed_q = weight32.T @ np.asarray(q3, dtype=np.float32)
    pushed_cp3 = cp3_read(lam3, pushed_p, pushed_q)
    pushed_cubic = push_cubic(cubic, weight32)
    pushed_u = weight32.T @ np.asarray(u4, dtype=np.float32)
    pushed_cp4 = cp4_planes(lam4, pushed_u)
    pre_covariance = (weight64.T @ covariance) @ weight64
    pushed_star = push_quartic(quartic, alpha, weight32, pre_covariance)
    reconstructed = (
        pushed_cp3 + pushed_cubic,
        pushed_cp4[0] + pushed_star[0],
        pushed_cp4[1] + pushed_star[1],
        pushed_cp4[2] + pushed_star[2],
    )
    return {
        "cp3_incident_nonzero": incident_nonzero(pushed_cp3, pushed_invalid),
        "cubic_incident_nonzero": incident_nonzero(pushed_cubic, pushed_invalid),
        "cp4_incident_nonzero": sum(
            incident_nonzero(value, pushed_invalid) for value in pushed_cp4
        ),
        "quartic_incident_nonzero": sum(
            incident_nonzero(value, pushed_invalid) for value in pushed_star
        ),
        "reconstructed_incident_nonzero": sum(
            incident_nonzero(value, pushed_invalid) for value in reconstructed
        ),
        "all_fields_finite": all(np.isfinite(value).all() for value in reconstructed),
    }


def run_instrumented(label: str, weights: list[np.ndarray], expected: dict[str, object], helper) -> dict[str, object]:
    module = load(f"q24_cs2_independent_{label}", SOURCE)
    records: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    pending: dict[str, object] | None = None
    certificate_pending: dict[str, object] | None = None
    original_certificate = module._certified_invalid_marginals
    original_full_reset = module._full_relu_reset
    original_cp_post = module._cp_post_read
    original_cp4_reset = module._cp4_reset
    original_reset_stars = module._reset_stars
    original_cp_grow = module._cp_grow

    def observed_certificate(raw1, raw2, raw3, raw4):
        nonlocal certificate_pending
        invalid = original_certificate(raw1, raw2, raw3, raw4)
        raw_arrays = tuple(np.asarray(value).copy() for value in (raw1, raw2, raw3, raw4))
        invalid_array = np.asarray(invalid, dtype=bool).copy()
        exact_negative_a = 0
        exact_negative_b = 0
        false_certifications = 0
        for values, flag in zip(zip(*raw_arrays), invalid_array):
            margin_a, margin_b = exact_margins(values)
            exact_negative_a += int(margin_a < 0)
            exact_negative_b += int(margin_b < 0)
            false_certifications += int(bool(flag) and margin_a >= 0 and margin_b >= 0)
        certificate_pending = {
            "raw": raw_arrays,
            "invalid": invalid_array,
            "exact_negative_a": exact_negative_a,
            "exact_negative_b": exact_negative_b,
            "false_certifications": false_certifications,
            "dtype_all_float64": all(value.dtype == np.float64 for value in raw_arrays),
            "all_finite": all(np.isfinite(value).all() for value in raw_arrays),
        }
        certificate_rows.append(certificate_pending)
        return invalid

    def observed_full_reset(*args, **kwargs):
        nonlocal pending
        value = original_full_reset(*args, **kwargs)
        if certificate_pending is None:
            raise AssertionError("full reset did not consume a certificate")
        valid = np.asarray(value[-1], dtype=bool).copy()
        if not np.array_equal(valid, ~np.asarray(certificate_pending["invalid"])):
            raise AssertionError("returned validity mask differs from certificate")
        pending = {
            "invalid": ~valid,
            "targets": tuple(np.asarray(item).copy() for item in value[2:6]),
            "covariance": np.asarray(value[1]).copy(),
            "variance": np.asarray(value[9]).copy(),
            "gate_birth_violation": 0,
        }
        return value

    def observed_cp_post(*args, **kwargs):
        value = original_cp_post(*args, **kwargs)
        if pending is not None:
            pending["cp_post"] = np.asarray(value).copy()
            invalid = np.asarray(pending["invalid"])
            gate = np.asarray(args[1])
            birth = np.asarray(args[4])
            pending["gate_birth_violation"] = int(
                pending["gate_birth_violation"]
            ) + int(np.count_nonzero(gate[invalid])) + int(np.count_nonzero(birth[invalid]))
        return value

    def observed_cp4_reset(*args, **kwargs):
        value = original_cp4_reset(*args, **kwargs)
        if pending is not None:
            invalid = np.asarray(pending["invalid"])
            cp4_lambda = np.asarray(args[0], dtype=np.float32)
            pushed_u = np.asarray(args[1], dtype=np.float32)
            gate32 = np.asarray(args[2], dtype=np.float32)
            k4_target = np.asarray(args[4], dtype=np.float64)
            inherited_u = gate32[:, None] * pushed_u
            inherited_d = cp4_planes(cp4_lambda, inherited_u)[0]
            birth = np.asarray(k4_target - inherited_d, dtype=np.float32)
            pending["gate_birth_violation"] = int(
                pending["gate_birth_violation"]
            ) + int(np.count_nonzero(gate32[invalid])) + int(np.count_nonzero(birth[invalid]))
            pending["cp4_state"] = (
                np.asarray(value[0]).copy(), np.asarray(value[1]).copy()
            )
            pending["provider"] = tuple(np.asarray(item).copy() for item in value[2])
        return value

    def observed_reset_stars(*args, **kwargs):
        value = original_reset_stars(*args, **kwargs)
        if pending is not None:
            pending["cubic"] = np.asarray(value[0]).copy()
            pending["quartic"] = np.asarray(value[1]).copy()
            pending["pairing_alpha"] = float(np.asarray(value[2]))
            pending["pairing_enabled"] = bool(np.asarray(kwargs["pairing_enabled"]))
        return value

    def observed_cp_grow(*args, **kwargs):
        nonlocal pending
        value = original_cp_grow(*args, **kwargs)
        if pending is not None:
            pending["cp3_state"] = tuple(np.asarray(item).copy() for item in value)
            current = current_carrier_gate(pending)
            width = int(np.asarray(pending["invalid"]).shape[0])
            identity = np.eye(width, dtype=np.float32)
            permutation, destination = signed_permutation(width)
            invalid = np.asarray(pending["invalid"], dtype=bool)
            identity_push = one_push_gate(pending, identity, invalid)
            permutation_push = one_push_gate(
                pending, permutation, mapped_invalid(invalid, destination)
            )
            records.append(
                {
                    "invalid_count": int(np.count_nonzero(invalid)),
                    "pairing_enabled": bool(pending["pairing_enabled"]),
                    "current": current,
                    "identity_push": identity_push,
                    "signed_permutation_push": permutation_push,
                }
            )
            pending = None
        return value

    module._certified_invalid_marginals = observed_certificate
    module._full_relu_reset = observed_full_reset
    module._cp_post_read = observed_cp_post
    module._cp4_reset = observed_cp4_reset
    module._reset_stars = observed_reset_stars
    module._cp_grow = observed_cp_grow
    mlp = MLP(
        width=int(weights[0].shape[0]), depth=16, weights=weights,
        seed=0, name=label,
    )
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = np.asarray(output)
        outcome = "FINITE" if np.isfinite(array).all() else "NONFINITE"
        output_sha256 = helper.digest_array(array)
        flops_used = int(budget.flops_used)
        calls = operation_count(budget)
    except Exception as exc:
        outcome = "EXCEPTION"
        output_sha256 = None
        flops_used = None
        calls = None
        exception = (type(exc).__name__, str(exc))
    if outcome != expected["outcome"]:
        raise AssertionError(f"{label}: outcome mismatch")
    if outcome == "FINITE" and output_sha256 != expected["output_sha256"]:
        raise AssertionError(f"{label}: output hash mismatch")
    # The frozen probe's observation hooks deliberately use additional fnp
    # reductions, so its per-fixture bill includes instrumentation.  This
    # independent hook uses plain NumPy only; source-only price is established
    # separately below by uninstrumented identity runs.
    if outcome == "EXCEPTION" and exception != (
        expected["exception_type"], expected["exception_message"]
    ):
        raise AssertionError(f"{label}: exception mismatch")
    if len(records) != 14 or len(certificate_rows) != 14:
        raise AssertionError(f"{label}: expected fourteen recursive resets")
    activated = sum(int(row["invalid_count"]) for row in records)
    violation_total = 0
    finite_gate = True
    for row in records:
        invalid_count = int(row["invalid_count"])
        if bool(row["pairing_enabled"]) != (invalid_count == 0):
            violation_total += 1
        for section in ("current", "identity_push", "signed_permutation_push"):
            fields = row[section]
            finite_gate = finite_gate and bool(fields["all_fields_finite"])
            violation_total += sum(
                int(value)
                for key, value in fields.items()
                if key != "all_fields_finite"
            )
    return {
        "fixture": label,
        "outcome": outcome,
        "output_sha256": output_sha256,
        "flops": flops_used,
        "calls": calls,
        "recursive_reset_count": len(records),
        "activation_count": activated,
        "carrier_and_one_push_violation_count": violation_total,
        "all_reconstructed_and_pushed_fields_finite": finite_gate,
        "certificate_false_certification_count": sum(
            int(row["false_certifications"]) for row in certificate_rows
        ),
        "certificate_exact_negative_a_count": sum(
            int(row["exact_negative_a"]) for row in certificate_rows
        ),
        "certificate_exact_negative_b_count": sum(
            int(row["exact_negative_b"]) for row in certificate_rows
        ),
        "certificate_inputs_all_finite_float64": all(
            bool(row["all_finite"]) and bool(row["dtype_all_float64"])
            for row in certificate_rows
        ),
    }


def run_plain(path: Path, tag: str, width: int) -> tuple[int, int]:
    module = load(tag, path)
    eye = np.eye(width, dtype=np.float32)
    mlp = MLP(
        width=width, depth=16, weights=[eye.copy() for _ in range(16)],
        seed=0, name=f"identity_width{width}",
    )
    with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
        module._predict(mlp)
    return int(budget.flops_used), operation_count(budget)


def main() -> int:
    bindings = {
        "source_sha256": digest(SOURCE),
        "parent_sha256": digest(PARENT),
        "preregister_sha256": digest(PREREGISTER),
        "runner_sha256": digest(RUNNER),
        "result_sha256": digest(RESULT),
        "source_hold_sha256": digest(SOURCE_HOLD),
        "fixture_helper_sha256": digest(FIXTURE_HELPER),
        "design_preregister_sha256": digest(DESIGN_PREREGISTER),
    }
    assert bindings == {
        "source_sha256": SOURCE_SHA256,
        "parent_sha256": PARENT_SHA256,
        "preregister_sha256": PREREGISTER_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "result_sha256": RESULT_SHA256,
        "source_hold_sha256": SOURCE_HOLD_SHA256,
        "fixture_helper_sha256": FIXTURE_HELPER_SHA256,
        "design_preregister_sha256": DESIGN_PREREGISTER_SHA256,
    }
    assert SOURCE.stat().st_size == 72_277

    source_text = SOURCE.read_text(encoding="utf-8")
    parent_text = PARENT.read_text(encoding="utf-8")
    source_tree = ast.parse(source_text, filename=str(SOURCE))
    parent_tree = ast.parse(parent_text, filename=str(PARENT))
    expected_imports = sorted([
        "flopscope", "flopscope.numpy", "whestbench.BaseEstimator",
        "whestbench.domain.MLP",
    ])
    assert import_names(source_tree) == expected_imports
    assert import_names(parent_tree) == expected_imports
    forbidden_calls = {"open", "eval", "exec", "compile", "__import__"}
    assert not [
        node for node in ast.walk(source_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden_calls
    ]
    source_functions = function_map(source_tree)
    parent_functions = function_map(parent_tree)
    added_functions = sorted(set(source_functions) - set(parent_functions))
    changed_functions = sorted(
        name for name in set(source_functions) & set(parent_functions)
        if ast.dump(source_functions[name], include_attributes=False)
        != ast.dump(parent_functions[name], include_attributes=False)
    )
    assert added_functions == [
        "_add_sub_error", "_certified_invalid_marginals",
        "_multiply_error", "_shutdown_connected_state", "_up",
    ]
    assert changed_functions == ["_full_relu_reset", "_predict", "_reset_stars"]
    assert ast.dump(source_functions["_q24_context"], include_attributes=False) == ast.dump(
        parent_functions["_q24_context"], include_attributes=False
    )
    assert ast.dump(source_functions["_univariate_edgeworth"], include_attributes=False) == ast.dump(
        parent_functions["_univariate_edgeworth"], include_attributes=False
    )
    assert source_text.count("_TWO_U = 2.220446049250313e-16") == 1
    assert source_text.count("pairing_enabled=fnp.all(valid_high_order)") == 1
    assert source_text.count("or int(budget) < _PHASE2_FLOP_BUDGET") == 1
    classes = [
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Estimator"
    ]
    assert len(classes) == 1 and ast.unparse(classes[0].bases[0]) == "BaseEstimator"
    predict = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    ]
    assert len(predict) == 1
    assert [arg.arg for arg in predict[0].args.args] == ["self", "mlp", "budget"]

    frozen = json.loads(RESULT.read_text())
    rows = frozen["rows"]
    invariant_keys = (
        "pairing_alpha_violation_counts_by_reset",
        "star_incident_nonzero_counts_by_reset",
        "cp_carrier_incident_nonzero_counts_by_reset",
        "combined_carrier_incident_nonzero_counts_by_reset",
    )
    recomputed_gates = {
        "activation_count": sum(
            sum(row["shutdown"]["shutdown_counts_by_reset"]) for row in rows
        ),
        "no_parent_finite_regression": all(
            not (
                row["parent"]["outcome"] == "FINITE"
                and row["shutdown"]["outcome"] != "FINITE"
            )
            for row in rows
        ),
        "stored_carrier_invariants": all(
            all(
                value == 0 for key in invariant_keys
                for value in row["shutdown"][key]
            )
            for row in rows
        ),
        "no_activation_output_parity": all(
            sum(row["shutdown"]["shutdown_counts_by_reset"]) != 0
            or row["finite_pair_disagreement"] is not None
            and row["finite_pair_disagreement"]["all_layer_mse"] == 0.0
            for row in rows
        ),
        "all_finite_pairs_within_parent_scale": all(
            row["finite_pair_disagreement"] is None
            or row["finite_pair_disagreement"]["rms_scale_gate"]
            and row["finite_pair_disagreement"]["maximum_scale_gate"]
            for row in rows
        ),
        "small_width_resource_law": all(
            row["flop_delta"]
            == 350 * row["width"] * row["width"] + 4634 * row["width"] + 98
            and row["call_delta"] == 2296
            for row in frozen["resource_rows"]
        ),
        "parent_failure_recovered_count": sum(
            row["parent"]["outcome"] != "FINITE"
            and row["shutdown"]["outcome"] == "FINITE"
            for row in rows
        ),
    }
    assert recomputed_gates == {
        key: frozen["gates"][key] for key in recomputed_gates
    }
    assert frozen["gates"]["source_probe_pass"] is True
    expected = {row["fixture"]: row["shutdown"] for row in rows}
    helper = load("q24_cs2_independent_fixture_helper", FIXTURE_HELPER)
    dynamic_rows = []
    for label, weights in helper.fixture_rows():
        row = run_instrumented(label, weights, expected[label], helper)
        dynamic_rows.append(row)
        print(
            label,
            row["outcome"],
            row["activation_count"],
            row["carrier_and_one_push_violation_count"],
            flush=True,
        )
    assert sum(row["activation_count"] for row in dynamic_rows) == 2327
    assert all(row["carrier_and_one_push_violation_count"] == 0 for row in dynamic_rows)
    assert all(row["all_reconstructed_and_pushed_fields_finite"] for row in dynamic_rows)
    assert all(row["certificate_false_certification_count"] == 0 for row in dynamic_rows)
    assert all(row["certificate_inputs_all_finite_float64"] for row in dynamic_rows)

    resource_rows = []
    for width in (4, 8, 16, 32):
        parent_flops, parent_calls = run_plain(
            PARENT, f"q24_cs2_audit_parent_{width}", width
        )
        child_flops, child_calls = run_plain(
            SOURCE, f"q24_cs2_audit_child_{width}", width
        )
        delta = child_flops - parent_flops
        call_delta = child_calls - parent_calls
        assert delta == 350 * width * width + 4634 * width + 98
        assert call_delta == 2296
        resource_rows.append({
            "width": width,
            "flop_delta": delta,
            "call_delta": call_delta,
        })
    projected_increment = 350 * 1024 * 1024 + 4634 * 1024 + 98
    projected_total = PARENT_LITERAL_FLOPS + projected_increment
    assert projected_total < PHASE2_BUDGET
    assert PARENT_LITERAL_CALLS + 2296 == 9504
    assert not any(HERE.rglob("__pycache__"))
    assert not any(HERE.rglob("*.pyc"))
    summary = {
        "bindings": bindings,
        "static": {
            "imports": expected_imports,
            "added_functions": added_functions,
            "changed_functions": changed_functions,
            "source_bytes": SOURCE.stat().st_size,
            "production_guard": True,
        },
        "frozen_probe_recomputed_gates": recomputed_gates,
        "dynamic_rows": dynamic_rows,
        "resource_rows": resource_rows,
        "projection": {
            "increment_flops_width1024": projected_increment,
            "total_flops_width1024": projected_total,
            "headroom_flops_width1024": PHASE2_BUDGET - projected_total,
            "utilization_percent": 100.0 * projected_total / PHASE2_BUDGET,
            "projected_calls": 9504,
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
