#!/usr/bin/env python3
"""Read-only static verifier for the P2R0C1D7R1 recovery design."""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D7 = P2 / "p2r0c1d7_cp3_staged_factorized_einsum_design_v1"
SOURCE = HERE / "staged_winograd_kernel.py"
MANIFEST = HERE / "ACCOUNTING_AND_RESOURCE_MANIFEST.json"
DESIGN = HERE / "DESIGN.md"
RUNNER = HERE / "run_tiny_runtime_discriminator.py"
D5_KERNEL_SMOKE = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/KERNEL_SMOKE_RESULT.json"
D5_KERNEL_RUNNER = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/run_kernel_smoke.py"
DUAL_LAUNCH_AUDIT = D7 / "INDEPENDENT_D7_DUAL_LAUNCH_FAILURE_AND_R1_ELIGIBILITY_AUDIT.md"
ATTEMPT_001_SEAL = D7 / "TINY_RUNTIME_LAUNCH_SEAL_ATTEMPT_001.json"
ATTEMPT_001_TRANSCRIPT = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_001_TRANSCRIPT.txt"
ATTEMPT_001_FAILURE = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_001_FAILURE.json"
ATTEMPT_002_SEAL = D7 / "TINY_RUNTIME_LAUNCH_SEAL.json"
ATTEMPT_002_TRANSCRIPT = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_002_TRANSCRIPT.txt"
ATTEMPT_002_FAILURE = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_002_FAILURE.json"
RECOVERY_AUTHORITIES = {
    DUAL_LAUNCH_AUDIT: "98db28ab6c80c5fd012fc569ed123cfbcf48d33fea7ec16a380b04be53560998",
    ATTEMPT_001_SEAL: "8aeb6adc651a8292dea39be53f6d3c01f66c0a4a9e82be257e05bc9253fb31fa",
    ATTEMPT_001_TRANSCRIPT: "0e244c00291c6a915f2129082df5bed2510f7ffae4df00c49a96086207281417",
    ATTEMPT_001_FAILURE: "5bdcc2530ff208843de0dc169bbc8d8b17746e75f12a1829b626531d24d0bbff",
    ATTEMPT_002_SEAL: "72c7bc9e264f63de2f6bcbe5129b927ed006bfe41b4cbbb98e7282c35c379602",
    ATTEMPT_002_TRANSCRIPT: "4ac804fd6b724871362fb274216ca640d0e19ea1806b01c512d3707540b949f2",
    ATTEMPT_002_FAILURE: "6084d8f0e7f36be2c82ac5c67cf2ccb0b0dc3de719ee171adee6083587f33bcf",
}

EXPECTED = {
    SOURCE: "81d14f2ca983c2820ab078d15a3a8a861427ac5a8b1b5e82828300256648bbe9",
    MANIFEST: "94ad1b2e89a58417bf91cb4fd844ac1d02674b58a2738b0218bf7cc63eb145ec",
    DESIGN: "3ccdefd80d69817f1ad8519124f2f0def2f03e95a5c96385d921edfdc97b1b2e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment_literal(tree: ast.Module, name: str):
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def fnp_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "fnp"
    ]


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d7(m: int, k: int, n: int) -> int:
    a, b, c = m * k, k * n, m * n
    leaf = 7**4 * mm(m // 16, k // 16, n // 16)
    transforms = sum(
        49 * 7**level * (a + b) // 4 ** (level + 1)
        for level in range(4)
    )
    reconstruction = sum(
        52 * 7**level * c // 4 ** (level + 1)
        for level in range(4)
    )
    return leaf + transforms + reconstruction + a + b + 2 * c


def q_elements(elements: int, level: int) -> int:
    value = Fraction(7, 4) ** level * elements
    assert value.denominator == 1
    return value.numerator


def conservative_peak(m: int, k: int, n: int) -> dict[str, int]:
    a, b, c = m * k, k * n, m * n
    a4 = q_elements(a, 4)
    b3, b4 = q_elements(b, 3), q_elements(b, 4)
    p3, p4 = q_elements(c, 3), q_elements(c, 4)
    return {
        "logical_b_stage": 4 * (a4 + b3 + b4),
        "expanding_transform": 4 * (a4 + b3 + 2 * b4),
        "leaf": 4 * (a4 + b4 + p4),
        "leaf_full_input_pack": 4 * (2 * a4 + 2 * b4 + p4),
        "reconstruction_first": 4 * (p4 + p3),
        "reconstruction_full_input_copy": 4 * (2 * p4 + p3),
    }


def verify_one_level_bilinear_identity(u, v, w) -> None:
    # Every pair of matrix-unit inputs must produce the corresponding 2x2
    # matrix-product unit.  Coefficients are restricted to exact integers.
    for values in (u, v, w):
        for first in values:
            for second in first:
                if isinstance(second, tuple):
                    assert all(int(x) == x for x in second)
                else:
                    assert int(second) == second
    for a_row in range(2):
        for a_col in range(2):
            for b_row in range(2):
                for b_col in range(2):
                    branches = [
                        int(u[r][a_row][a_col]) * int(v[r][b_row][b_col])
                        for r in range(7)
                    ]
                    for out_row in range(2):
                        for out_col in range(2):
                            observed = sum(
                                int(w[out_row][out_col][r]) * branches[r]
                                for r in range(7)
                            )
                            expected = int(
                                a_col == b_row
                                and out_row == a_row
                                and out_col == b_col
                            )
                            assert observed == expected, (
                                a_row, a_col, b_row, b_col,
                                out_row, out_col, observed, expected,
                            )


def main() -> None:
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)
    assert sha256(D5_KERNEL_SMOKE) == (
        "a50ca69aae9e4b39fca8d3de9014bcc7688af63a1f5ced6aab1f94de957235c9"
    )
    assert sha256(D5_KERNEL_RUNNER) == (
        "44940602fc2f6fdb336911821de024bddb94da021a01ddf0ba9570e251bfb996"
    )
    assert sha256(D7 / "staged_winograd_kernel.py") == EXPECTED[SOURCE]
    for authority, expected in RECOVERY_AUTHORITIES.items():
        assert authority.is_file(), authority
        assert sha256(authority) == expected, (authority, sha256(authority), expected)

    runner_text = RUNNER.read_text()
    ast.parse(runner_text)
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in runner_text
    assert "D7R1_TINY_ATTEMPT_MARKER.json" in runner_text
    assert "D7R1_BIND_INVENTORY_PREASSERT.json" in runner_text
    assert "D7R1_RUNTIME_EXCEPTION_RECEIPT.json" in runner_text
    assert "D7R1_TINY_RUNTIME_RESULT.json" in runner_text
    assert runner_text.index("marker = claim_attempt(sha256(SEAL))") < (
        runner_text.index("kernel = load_kernel()")
    )
    assert runner_text.index("write_once(BIND_PREASSERT") < runner_text.index(
        "assert bind_inventory == expected_bind_inventory"
    )
    assert '"flops": 168' in runner_text
    assert '"costs": {"asarray": 168}' in runner_text
    for forbidden in (
        "ATTEMPT_MARKER.unlink",
        "os.unlink(ATTEMPT_MARKER",
        "os.remove(ATTEMPT_MARKER",
    ):
        assert forbidden not in runner_text
    for relative in (
        "D7R1_TINY_ATTEMPT_MARKER.json",
        "D7R1_BIND_INVENTORY_PREASSERT.json",
        "D7R1_RUNTIME_EXCEPTION_RECEIPT.json",
        "D7R1_TINY_RUNTIME_RESULT.json",
    ):
        assert not (HERE / relative).exists(), relative

    tree = ast.parse(SOURCE.read_text())
    kernel = function(tree, "staged_winograd")
    bind = function(tree, "bind_coefficients")
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(kernel))
    delete_targets = [
        ast.unparse(target)
        for node in kernel.body
        if isinstance(node, ast.Delete)
        for target in node.targets
    ]
    assert delete_targets == [
        "a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3",
        "a4", "b4", "p4", "p3", "p2", "p1", "out_blocks",
    ]
    kernel_calls = fnp_calls(kernel)
    inventory = Counter(call.func.attr for call in kernel_calls)
    assert inventory == {"reshape": 3, "einsum": 12, "matmul": 1}
    assert Counter(call.func.attr for call in fnp_calls(bind)) == {"asarray": 3}
    einsums = [call for call in kernel_calls if call.func.attr == "einsum"]
    out_einsums = [
        call for call in einsums
        if any(keyword.arg == "out" for keyword in call.keywords)
    ]
    assert len(out_einsums) == 1
    assert ast.unparse(out_einsums[0]).startswith(
        "fnp.einsum('pir,rvwxMqzyN->pvwxMiqzyN'"
    )
    for call in einsums:
        optimize = [kw.value for kw in call.keywords if kw.arg == "optimize"]
        assert len(optimize) == 1 and ast.unparse(optimize[0]) == "PAIRWISE_PATH"
    source_text = SOURCE.read_text()
    for forbidden in ("fnp.stack", "fnp.block", "fnp.concatenate", "_np.einsum"):
        assert forbidden not in source_text

    u = assignment_literal(tree, "U_COEFFICIENTS")
    v = assignment_literal(tree, "V_COEFFICIENTS")
    w = assignment_literal(tree, "W_COEFFICIENTS")
    assert (len(u), len(u[0]), len(u[0][0])) == (7, 2, 2)
    assert (len(v), len(v[0]), len(v[0][0])) == (7, 2, 2)
    assert (len(w), len(w[0]), len(w[0][0])) == (2, 2, 7)
    assert tuple(w[1][1]) == (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    verify_one_level_bilinear_identity(u, v, w)

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == (
        "FROZEN_D7R1_RECOVERY_PRE_TINY_REVIEW_NO_RUNTIME_NO_EXACT_SHAPE"
    )
    assert manifest["node"] == "P2R0C1D7R1"
    assert manifest["source"]["sha256"] == EXPECTED[SOURCE]
    roster = (
        [(1024, 1024, 1024 * layer) for layer in range(1, 16) for _ in range(2)]
        + [(1024, 1024 * layer, 1024) for layer in range(1, 15) for _ in range(2)]
    )
    assert len(roster) == 58
    direct = sum(mm(*shape) for shape in roster)
    candidate = sum(d7(*shape) for shape in roster)
    assert direct == manifest["accounting"]["direct_cp3_roster_flops"] == 966_086_623_232
    assert candidate == manifest["accounting"]["roster_flops"] == 704_947_724_288
    assert direct - candidate == 261_138_898_944
    assert manifest["accounting"]["roster_meter_inventory"] == {
        "reshape": 174, "einsum": 696, "matmul": 58, "copyto": 58,
    }
    assert sum(manifest["accounting"]["roster_meter_inventory"].values()) == 986
    assert manifest["accounting"]["roster_python_counted_wrapper_calls"] == 928
    accounting = manifest["accounting"]
    assert accounting["coefficient_binding_cold_flops"] == 168
    assert accounting["coefficient_binding_cold_inventory"] == {"asarray": 3}
    assert accounting["coefficient_binding_cold_costs"] == {"asarray": 168}
    assert accounting["coefficient_binding_materialized_elements"] == 84
    assert accounting["coefficient_binding_source_dtype_rate"] == 2
    assert accounting["projected_d7r1_full_calls_35_279_35"] == [
        1_431_110_075_935, 1_469_753_252_360, 1_431_106_923_008,
    ]
    assert accounting["projected_incumbent_minus_d7"] == [
        261_374_828_376, 261_374_828_544, 261_374_828_544,
    ]

    recovery = manifest["recovery"]
    assert recovery["ancestor_node"] == "P2R0C1D7"
    assert recovery["launch_owner"] == "/root"
    assert recovery["dual_launch_audit"]["sha256"] == sha256(DUAL_LAUNCH_AUDIT)
    attempt_1 = recovery["failed_attempt_chains"]["attempt_001"]
    attempt_2 = recovery["failed_attempt_chains"]["attempt_002"]
    assert attempt_1["seal_sha256"] == sha256(ATTEMPT_001_SEAL)
    assert attempt_1["transcript_sha256"] == sha256(ATTEMPT_001_TRANSCRIPT)
    assert attempt_1["failure_sha256"] == sha256(ATTEMPT_001_FAILURE)
    assert attempt_2["seal_sha256"] == sha256(ATTEMPT_002_SEAL)
    assert attempt_2["transcript_sha256"] == sha256(ATTEMPT_002_TRANSCRIPT)
    assert attempt_2["failure_sha256"] == sha256(ATTEMPT_002_FAILURE)

    push = conservative_peak(1024, 1024, 15360)
    read = conservative_peak(1024, 14336, 1024)
    assert push == {
        "logical_b_stage": 966_590_464,
        "expanding_transform": 1_556_660_224,
        "leaf": 1_219_477_504,
        "leaf_full_input_pack": 1_848_885_248,
        "reconstruction_first": 927_252_480,
        "reconstruction_full_input_copy": 1_517_322_240,
    }
    assert read == {
        "logical_b_stage": 1_416_167_424,
        "expanding_transform": 1_966_899_200,
        "leaf": 1_140_801_536,
        "leaf_full_input_pack": 2_242_265_088,
        "reconstruction_first": 61_816_832,
        "reconstruction_full_input_copy": 101_154_816,
    }
    memory = manifest["memory"]
    assert memory["worst_tracked_peak_bytes_including_coefficients"] == (
        read["leaf_full_input_pack"] + 336
    ) == 2_242_265_424
    assert memory["conservative_increment_cap_including_512mib_contingency"] == (
        2_242_265_424 + 512 * 1024**2
    ) == 2_779_136_336
    assert memory["incumbent_anchor_plus_conservative_increment_cap"] == (
        3_421_503_488 + 2_779_136_336
    ) == 6_200_639_824 < 7 * 1024**3
    assert memory["literal_absolute_rss_strictly_below"] == 7 * 1024**3

    d5_smoke = json.loads(D5_KERNEL_SMOKE.read_text())
    discriminator = manifest["discriminator"]
    assert d5_smoke["status"] == "PASS_TARGET_FREE_KERNEL_SMOKE"
    assert d5_smoke["depth"] == 4
    assert d5_smoke["max_abs_delta_vs_float64"] == 1.5845729923968932e-5
    assert d5_smoke["rms_delta_vs_float64"] == 2.6939190170640823e-6
    assert discriminator[
        "kernel_maximum_absolute_delta_normalized_by_reference_rms_max"
    ] == 1e-4 / 0.9888505191597005 == 1.0112751933930056e-4
    authority = discriminator["d5_synthetic_kernel_authority"]
    assert authority["result_sha256"] == sha256(D5_KERNEL_SMOKE)
    assert authority["runner_sha256"] == sha256(D5_KERNEL_RUNNER)
    assert authority["sealed_max_absolute_gate_vs_float64"] == 1e-4
    assert authority["reference_rms_for_sealed_case"] == 0.9888505191597005
    assert "1e-4/0.9888505191597005" in discriminator[
        "kernel_numerical_gate_derivation"
    ]

    access = manifest["access_declarations"]
    assert access["estimator_imported"] is False
    assert access["estimator_calls"] == access["exact_shape_calls"] == 0
    assert access["target_columns_opened"] == access["scores_opened"] == []
    assert access["submission"] is access["network"] is False
    print("PASS_P2R0C1D7R1_RECOVERY_DESIGN_STATIC")


if __name__ == "__main__":
    main()
