#!/usr/bin/env python3
"""Target-free author gates for the additive A1_NO_CP4 source."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
PARENT = P2 / "fullbudget_k21_k4_closure_v4" / "estimator.py"
CANDIDATE = HERE / "estimator.py"
PARENT_RECEIPT = (
    P2
    / "meter_protocol"
    / "runs"
    / "fb_ec4_v4_sparse_resetarena_cp3compact_n1_s8_seed20260824"
    / "receipt.json"
)
CP4_AUDIT = P2 / "FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md"
PARENT_SHA256 = "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
PARENT_FLOPS = 2_179_742_954_469
CP4_PROJECTED_FLOPS = 234_507_794_432
CP4_DOMINANT_FLOPS = 233_997_074_432
PHASE2_BUDGET = 1 << 41
EXPECTED_CALL_DELETION = 925


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation_calls(budget) -> int:
    return sum(
        item["calls"] for item in budget.summary_dict()["operations"].values()
    )


def _function_ast_map(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _static_gate(parent_source: str, candidate_source: str) -> dict[str, object]:
    parent_tree = ast.parse(parent_source)
    candidate_tree = ast.parse(candidate_source)
    imports = []
    for node in candidate_tree.body:
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(
                f"{node.module}.{item.name}" for item in node.names
            )
    assert imports == [
        "flopscope",
        "flopscope.numpy",
        "whestbench.BaseEstimator",
        "whestbench.domain.MLP",
    ]

    forbidden_calls = {
        "eval", "exec", "open", "compile", "__import__",
    }
    observed_forbidden_calls = []
    cp4_names = []
    for node in ast.walk(candidate_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                observed_forbidden_calls.append(node.func.id)
        if isinstance(node, ast.Name) and "cp4" in node.id.lower():
            cp4_names.append(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "cp4" in node.name.lower():
                cp4_names.append(node.name)
    assert not observed_forbidden_calls
    assert not cp4_names

    parent_functions = _function_ast_map(parent_source)
    candidate_functions = _function_ast_map(candidate_source)
    removed_functions = sorted(set(parent_functions) - set(candidate_functions))
    expected_removed = sorted(
        [
            "_assemble_quartic_provider",
            "_cp4_push",
            "_cp4_read_diagonal",
            "_cp4_read_planes",
            "_cp4_reset",
            "_quartic_residual",
        ]
    )
    assert removed_functions == expected_removed

    unchanged_science = [
        "_fusion_context",
        "_q38_coefficients",
        "_univariate_edgeworth",
        "_q32_context",
        "_edgeworth_weights",
        "_four_pair_edgeworth",
        "_connected_planes",
        "_repair_covariance",
        "_cp_read",
        "_cp_k3",
        "_cp_push",
        "_cp_post_read",
        "_cp_grow",
        "_push_cubic",
        "_push_cubic_diagonal",
        "_push_quartic",
        "_push_quartic_diagonal",
        "_exact_first_relu",
        "_full_relu_reset",
    ]
    for name in unchanged_science:
        assert candidate_functions[name] == parent_functions[name], name

    reset = next(
        node
        for node in candidate_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_stars"
    )
    reset_calls = {
        node.func.id
        for node in ast.walk(reset)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_quartic_residual" not in reset_calls
    assert "_off_diagonal" in reset_calls
    assert "_replace_diagonal" in reset_calls

    return {
        "imports_exact": imports,
        "ast_nodes": sum(1 for _ in ast.walk(candidate_tree)),
        "removed_functions_exact": removed_functions,
        "unchanged_science_function_count": len(unchanged_science),
        "forbidden_calls": observed_forbidden_calls,
        "cp4_ast_names": cp4_names,
        "parent_ast_nodes": sum(1 for _ in ast.walk(parent_tree)),
    }


def _reference_without_cp4(parent):
    """Disable CP4 in the frozen parent without copying candidate code."""

    def reset(
        cp4_lambda,
        pushed_u,
        gate,
        pre_provider_planes,
        k4_target,
        identity,
        diagonal_context,
    ):
        del cp4_lambda, pre_provider_planes, k4_target, identity, diagonal_context
        width = int(pushed_u.shape[0])
        return (
            fnp.zeros(0, dtype=fnp.float32),
            fnp.zeros((width, 0), dtype=fnp.float32),
            None,
            fnp.asarray(gate, dtype=fnp.float32),
        )

    parent._cp4_reset = reset
    parent._cp4_read_planes = lambda *_args: None
    parent._cp4_read_diagonal = lambda _lam, u: fnp.zeros(
        int(u.shape[0]), dtype=fnp.float64
    )
    return parent


def _fixtures() -> list[tuple[str, MLP]]:
    width = 8
    depth = 16
    identity = [np.eye(width, dtype=np.float32) for _ in range(depth)]

    rng = np.random.default_rng(20260823)
    near_identity = [
        (np.eye(width) + 0.02 * rng.standard_normal((width, width))).astype(
            np.float32
        )
        for _ in range(depth)
    ]

    rng = np.random.default_rng(20260823)
    qr_weights = []
    for _ in range(depth):
        q, _r = np.linalg.qr(rng.standard_normal((width, width)))
        qr_weights.append((np.eye(width) + 0.35 * q).astype(np.float64))

    rng = np.random.default_rng(17)
    contractive = [
        (
            0.88 * np.eye(width)
            + 0.12 * rng.standard_normal((width, width)) / np.sqrt(width)
        ).astype(np.float32)
        for _ in range(depth)
    ]
    return [
        ("identity_width8_float32", MLP(width, depth, identity, 0)),
        (
            "near_identity_seed20260823_scale0p02_float32",
            MLP(width, depth, near_identity, 20260823),
        ),
        (
            "identity_plus_qr_seed20260823_scale0p35_float64",
            MLP(width, depth, qr_weights, 20260823),
        ),
        (
            "contractive_seed17_scale0p12_float32",
            MLP(width, depth, contractive, 17),
        ),
    ]


def _trajectory_gate(parent, candidate) -> dict[str, object]:
    reference = _reference_without_cp4(parent)
    fixtures = []
    for name, mlp in _fixtures():
        with flops.BudgetContext(
            flop_budget=100_000_000_000, quiet=True
        ) as reference_budget:
            reference_output = reference._predict(mlp)
        with flops.BudgetContext(
            flop_budget=100_000_000_000, quiet=True
        ) as candidate_budget:
            candidate_output = candidate._predict(mlp)
        reference_array = np.asarray(reference_output)
        candidate_array = np.asarray(candidate_output)
        assert np.isfinite(reference_array).all()
        assert np.isfinite(candidate_array).all()
        assert np.array_equal(reference_array, candidate_array), name
        fixtures.append(
            {
                "name": name,
                "shape": list(candidate_array.shape),
                "bitwise_equal": True,
                "output_sha256": _array_sha256(candidate_array),
                "reference_flops": int(reference_budget.flops_used),
                "candidate_flops": int(candidate_budget.flops_used),
                "reference_calls": _operation_calls(reference_budget),
                "candidate_calls": _operation_calls(candidate_budget),
            }
        )
    return {"fixture_count": len(fixtures), "fixtures": fixtures}


def _weight_context(module, weight):
    weight32 = fnp.asarray(weight, dtype=fnp.float32)
    weight2 = weight32 * weight32
    weight3 = weight2 * weight32
    return (
        weight32,
        fnp.transpose(weight32),
        weight2,
        fnp.transpose(weight2),
        weight3,
        fnp.transpose(weight3),
    )


def _cp3_case(module, *, width: int, rank: int, seed: int):
    rng = np.random.default_rng(seed)
    lam = (0.01 * rng.standard_normal(rank)).astype(np.float32)
    p = (0.05 * rng.standard_normal((width, rank))).astype(np.float32)
    q = (0.05 * rng.standard_normal((width, rank))).astype(np.float32)
    weight = (
        np.eye(width) + 0.03 * rng.standard_normal((width, width))
    ).astype(np.float32)
    destination_p = fnp.asarray(
        np.full((width, module._CP3_CAP), np.nan, dtype=np.float32)
    )
    destination_q = fnp.asarray(
        np.full((width, module._CP3_CAP), np.nan, dtype=np.float32)
    )
    product_a = fnp.asarray(
        np.full(width * module._CP3_CAP, np.nan, dtype=np.float32)
    )
    product_b = fnp.asarray(
        np.full(width * module._CP3_CAP, np.nan, dtype=np.float32)
    )
    context = _weight_context(module, weight)
    cp_lambda, pushed_p, pushed_q = module._cp_push(
        fnp.asarray(lam),
        fnp.asarray(p),
        fnp.asarray(q),
        context,
        destination_p,
        destination_q,
    )
    read = module._cp_read(cp_lambda, pushed_p, pushed_q, product_a, product_b)
    diagonal = module._cp_k3(cp_lambda, pushed_p, pushed_q, product_a)
    return {
        "pushed_p": np.asarray(pushed_p).copy(),
        "pushed_q": np.asarray(pushed_q).copy(),
        "read": np.asarray(read).copy(),
        "diagonal": np.asarray(diagonal).copy(),
        "destination_p_inactive_nan": bool(
            np.isnan(np.asarray(destination_p)[:, rank:]).all()
        ),
        "destination_q_inactive_nan": bool(
            np.isnan(np.asarray(destination_q)[:, rank:]).all()
        ),
        "p_alias": bool(np.shares_memory(np.asarray(pushed_p), np.asarray(destination_p))),
        "q_alias": bool(np.shares_memory(np.asarray(pushed_q), np.asarray(destination_q))),
    }


def _cp3_grow_case(module, *, width: int, rank: int, seed: int):
    rng = np.random.default_rng(seed)
    lam = (0.01 * rng.standard_normal(rank)).astype(np.float32)
    pushed_p = (0.05 * rng.standard_normal((width, rank))).astype(np.float32)
    pushed_q = (0.05 * rng.standard_normal((width, rank))).astype(np.float32)
    gate = np.linspace(0.2, 0.8, width, dtype=np.float32)
    raw = rng.standard_normal((width, width))
    covariance = raw @ raw.T / width + np.eye(width)
    birth_lambda = np.linspace(0.1, 0.3, width, dtype=np.float64)
    identity = np.eye(width, dtype=np.float32)
    p_bank_np = np.full((width, module._CP3_CAP), np.nan, dtype=np.float32)
    q_bank_np = np.full((width, module._CP3_CAP), np.nan, dtype=np.float32)
    lambda_bank_np = np.full(module._CP3_CAP, np.nan, dtype=np.float32)
    lambda_bank_np[:rank] = lam
    p_bank = fnp.asarray(p_bank_np)
    q_bank = fnp.asarray(q_bank_np)
    lambda_bank = fnp.asarray(lambda_bank_np)
    out_lambda, out_p, out_q = module._cp_grow(
        fnp.asarray(lam),
        fnp.asarray(pushed_p),
        fnp.asarray(pushed_q),
        fnp.asarray(gate),
        fnp.asarray(covariance),
        fnp.asarray(birth_lambda),
        fnp.asarray(identity),
        p_bank,
        q_bank,
        lambda_bank,
    )
    new_rank = min(rank + width, module._CP3_CAP)
    assert int(out_lambda.shape[0]) == new_rank
    assert np.isfinite(np.asarray(out_lambda)).all()
    assert np.isfinite(np.asarray(out_p)).all()
    assert np.isfinite(np.asarray(out_q)).all()
    return {
        "new_rank": new_rank,
        "lambda": np.asarray(out_lambda).copy(),
        "p": np.asarray(out_p).copy(),
        "q": np.asarray(out_q).copy(),
        "lambda_alias": bool(
            np.shares_memory(np.asarray(out_lambda), np.asarray(lambda_bank))
        ),
        "p_alias": bool(np.shares_memory(np.asarray(out_p), np.asarray(p_bank))),
        "q_alias": bool(np.shares_memory(np.asarray(out_q), np.asarray(q_bank))),
    }


def _cp3_gate(parent, candidate) -> dict[str, object]:
    cases = []
    for width, rank, seed in [(32, 513, 31), (4, 16_382, 47), (4, 16_384, 53)]:
        with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True):
            parent_result = _cp3_case(parent, width=width, rank=rank, seed=seed)
        with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True):
            candidate_result = _cp3_case(
                candidate, width=width, rank=rank, seed=seed
            )
        for field in ["pushed_p", "pushed_q", "read", "diagonal"]:
            assert np.array_equal(parent_result[field], candidate_result[field])
            assert np.isfinite(candidate_result[field]).all()
        for field in [
            "destination_p_inactive_nan",
            "destination_q_inactive_nan",
            "p_alias",
            "q_alias",
        ]:
            assert candidate_result[field] is True
            assert candidate_result[field] == parent_result[field]
        cases.append(
            {
                "width": width,
                "rank": rank,
                "read_sha256": _array_sha256(candidate_result["read"]),
                "diagonal_sha256": _array_sha256(candidate_result["diagonal"]),
                "poison_and_alias_pass": True,
            }
        )

    grow_cases = []
    for width, rank, seed in [(32, 513, 61), (4, 16_382, 67), (4, 16_384, 71)]:
        with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True):
            parent_result = _cp3_grow_case(
                parent, width=width, rank=rank, seed=seed
            )
        with flops.BudgetContext(flop_budget=10_000_000_000, quiet=True):
            candidate_result = _cp3_grow_case(
                candidate, width=width, rank=rank, seed=seed
            )
        for field in ["lambda", "p", "q"]:
            assert np.array_equal(parent_result[field], candidate_result[field])
        for field in ["lambda_alias", "p_alias", "q_alias"]:
            assert candidate_result[field] is True
            assert candidate_result[field] == parent_result[field]
        grow_cases.append(
            {
                "width": width,
                "old_rank": rank,
                "new_rank": candidate_result["new_rank"],
                "lambda_sha256": _array_sha256(candidate_result["lambda"]),
                "poison_and_alias_pass": True,
            }
        )
    return {"push_read_cases": cases, "grow_cases": grow_cases}


def _h4_reset_gate(candidate) -> dict[str, object]:
    width = 7
    rng = np.random.default_rng(20260824)
    raw = rng.standard_normal((width, width))
    covariance = raw @ raw.T / width + 0.7 * np.eye(width)
    variance = np.diag(covariance).copy()
    k4 = rng.standard_normal(width) * 0.04
    k31 = rng.standard_normal((width, width)) * 0.03
    np.fill_diagonal(k31, 0.0)
    k22 = rng.standard_normal((width, width)) * 0.02
    k22 = 0.5 * (k22 + k22.T)
    np.fill_diagonal(k22, 0.0)
    k21 = rng.standard_normal((width, width)) * 0.03
    cp_post = rng.standard_normal((width, width)) * 0.01
    diagonal = np.eye(width, dtype=bool)
    covariance2 = covariance * covariance
    variance_outer = np.outer(variance, variance)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        cubic, quartic, alpha = candidate._reset_stars(
            fnp.asarray(k21),
            fnp.asarray(cp_post),
            fnp.asarray(k4),
            fnp.asarray(k31),
            fnp.asarray(k22),
            fnp.asarray(covariance),
            fnp.asarray(variance),
            fnp.asarray(covariance2),
            fnp.asarray(variance_outer),
            fnp.asarray(diagonal),
        )
    alpha = float(np.asarray(alpha))
    pairing = variance[:, None] * variance[None, :] + 2.0 * covariance * covariance
    np.fill_diagonal(pairing, 0.0)
    expected_alpha = float(np.sum(k22 * pairing) / np.sum(pairing * pairing))
    assert alpha == expected_alpha
    expected_h = k31 - 3.0 * alpha * variance[:, None] * covariance
    expected_diagonal = k4 - 3.0 * alpha * variance * variance
    np.fill_diagonal(expected_h, expected_diagonal)
    expected_h32 = expected_h.astype(np.float32)
    actual_h32 = np.asarray(quartic)
    assert np.array_equal(actual_h32, expected_h32)
    assert np.array_equal(np.asarray(cubic), (k21 - cp_post).astype(np.float32))

    h64 = actual_h32.astype(np.float64)
    replay_k4 = np.diag(h64) + 3.0 * alpha * variance * variance
    replay_k31 = h64 + 3.0 * alpha * variance[:, None] * covariance
    np.fill_diagonal(replay_k31, 0.0)
    k4_error = float(np.max(np.abs(replay_k4 - k4)))
    k31_error = float(np.max(np.abs(replay_k31 - k31)))
    cast_error = float(np.max(np.abs(expected_h32.astype(np.float64) - expected_h)))
    residual_k22 = k22 - alpha * pairing
    orthogonality = float(abs(np.sum(residual_k22 * pairing)))
    orthogonality_scale = float(np.sum(np.abs(k22 * pairing)) + 1.0)
    assert k4_error <= cast_error + 1.0e-15
    assert k31_error <= cast_error + 1.0e-15
    assert orthogonality <= 5.0e-15 * orthogonality_scale
    return {
        "width": width,
        "alpha_bitwise_numpy_formula": True,
        "h4_storage_bitwise_expected_float32": True,
        "cubic_storage_bitwise_expected_float32": True,
        "max_h4_float32_cast_error": cast_error,
        "max_k4_replay_error": k4_error,
        "max_k31_replay_error": k31_error,
        "k22_pairing_residual_inner_product_abs": orthogonality,
    }


def _guard_gate(candidate) -> dict[str, object]:
    class SmokeMLP:
        def __init__(self, width: int, depth: int):
            self.width = width
            self.depth = depth

    estimator = candidate.Estimator()
    with flops.BudgetContext(flop_budget=100, quiet=True) as off_shape_budget:
        off_shape = estimator.predict(
            SmokeMLP(4, 2), candidate._PHASE2_FLOP_BUDGET
        )
    with flops.BudgetContext(flop_budget=100, quiet=True) as low_budget:
        low = estimator.predict(
            SmokeMLP(1024, 16), candidate._PHASE2_FLOP_BUDGET - 1
        )
    assert off_shape.shape == (2, 4)
    assert low.shape == (16, 1024)
    assert not np.any(np.asarray(off_shape))
    assert not np.any(np.asarray(low))
    assert off_shape_budget.flops_used == 0
    assert low_budget.flops_used == 0
    assert _operation_calls(off_shape_budget) == 1
    assert _operation_calls(low_budget) == 1
    return {
        "off_shape_zero_shape": [2, 4],
        "low_budget_zero_shape": [16, 1024],
        "each_guard_calls": 1,
        "each_guard_flops": 0,
    }


def _call_and_resource_gate(candidate) -> dict[str, object]:
    parent = _load("fb_ec4_v4_call_parent", PARENT)
    parent._CP4_CAP = 16
    name, mlp = _fixtures()[1]
    with flops.BudgetContext(
        flop_budget=100_000_000_000, quiet=True
    ) as parent_budget:
        parent._predict(mlp)
    with flops.BudgetContext(
        flop_budget=100_000_000_000, quiet=True
    ) as candidate_budget:
        candidate._predict(mlp)
    parent_calls = _operation_calls(parent_budget)
    candidate_calls = _operation_calls(candidate_budget)
    call_deletion = parent_calls - candidate_calls
    assert call_deletion == EXPECTED_CALL_DELETION
    operations = parent_budget.summary_dict()["operations"]
    assert operations["argsort"]["calls"] == 13

    receipt = json.loads(PARENT_RECEIPT.read_text())
    parent_operation_calls = sum(
        item["calls"]
        for item in receipt["measured"]["operation_totals"].values()
    )
    assert parent_operation_calls == 6_878
    projected_calls = parent_operation_calls - call_deletion
    projected_flops = PARENT_FLOPS - CP4_PROJECTED_FLOPS
    reduction_fraction = CP4_PROJECTED_FLOPS / PARENT_FLOPS
    assert reduction_fraction >= 0.107
    assert projected_flops < PHASE2_BUDGET
    return {
        "branch_schedule_fixture": name,
        "branch_schedule": {
            "width": 8,
            "cp4_cap": 16,
            "depth": 16,
            "capped_resets": 13,
        },
        "small_parent_calls": parent_calls,
        "small_candidate_calls": candidate_calls,
        "exact_call_deletion_same_branch_schedule": call_deletion,
        "parent_literal_calls": parent_operation_calls,
        "projected_candidate_calls": projected_calls,
        "parent_literal_flops": PARENT_FLOPS,
        "cp4_dominant_kernel_flops": CP4_DOMINANT_FLOPS,
        "cp4_declared_source_projection_flops": CP4_PROJECTED_FLOPS,
        "projected_candidate_flops": projected_flops,
        "projected_candidate_fraction_of_2pow41": projected_flops / PHASE2_BUDGET,
        "projected_flop_reduction_fraction_of_parent": reduction_fraction,
        "projected_headroom_flops": PHASE2_BUDGET - projected_flops,
        "literal_meter_still_required": True,
    }


def verify() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flops.configure(symmetry_warnings=False)
    assert _sha256(PARENT) == PARENT_SHA256
    assert _sha256(CP4_AUDIT) == (
        "9f044c2665c2143413e5b7da398487c38545994b20ff177f3494c004b0811588"
    )
    parent_source = PARENT.read_text()
    candidate_source = CANDIDATE.read_text()
    static = _static_gate(parent_source, candidate_source)
    parent = _load("fb_ec4_v4_reference_parent", PARENT)
    candidate = _load("fb_ec4_a1_candidate", CANDIDATE)
    trajectory = _trajectory_gate(parent, candidate)
    cp3_parent = _load("fb_ec4_v4_cp3_parent", PARENT)
    cp3_candidate = _load("fb_ec4_a1_cp3_candidate", CANDIDATE)
    cp3 = _cp3_gate(cp3_parent, cp3_candidate)
    h4 = _h4_reset_gate(candidate)
    guards = _guard_gate(candidate)
    resources = _call_and_resource_gate(candidate)
    return {
        "schema": "fb-ec4-32-a1-no-cp4-author-verification-v1",
        "status": "PASS_TARGET_FREE_A1_SOURCE_ALGEBRA_PARITY_AND_PROJECTION",
        "candidate": {
            "path": "estimator.py",
            "bytes": CANDIDATE.stat().st_size,
            "sha256": _sha256(CANDIDATE),
        },
        "parent": {
            "path": "../fullbudget_k21_k4_closure_v4/estimator.py",
            "bytes": PARENT.stat().st_size,
            "sha256": _sha256(PARENT),
        },
        "static_phase2": static,
        "intended_science_reference": trajectory,
        "cp3_preservation": cp3,
        "coherent_h4_reset": h4,
        "production_guards": guards,
        "resource_projection": resources,
        "access_counters": {
            "dataset_members_opened": 0,
            "targets_opened": 0,
            "scores_computed": 0,
            "fits": 0,
            "samples": 0,
            "submissions": 0,
            "graph_edits": 0,
            "full_shape_runs": 0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
