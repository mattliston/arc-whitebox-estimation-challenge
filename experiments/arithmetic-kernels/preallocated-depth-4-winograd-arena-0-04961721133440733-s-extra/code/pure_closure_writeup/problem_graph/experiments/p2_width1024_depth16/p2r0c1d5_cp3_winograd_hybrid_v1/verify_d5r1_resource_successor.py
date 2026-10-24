#!/usr/bin/env python3
"""Static verifier for the D5R1 causal audit and design-only D6 successor.

This script reads receipts and source text only.  It does not import or call an
estimator and contains no target, score, board, production, or submission path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


bindings = {
    ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py":
        "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    HERE / "estimator.py":
        "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    HERE / "RECOVERED_REFERENCE_EVIDENCE.json":
        "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    HERE / "RECOVERY_DESCENDANT_RESULT.json":
        "a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549",
    HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json":
        "8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0",
    HERE / "INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md":
        "c17c38cffe77dd3f48de6bd0cf2554d92d5659c0a0c9bc4f9e94225b7b056de9",
    HERE / "verify_independent_recovery_descendant_terminal.py":
        "ede98b6732fe0e7b5eaeec3c6471da997ab2c4c14383a9fb8785c702f73e985c",
    ROOT / "autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py":
        "c3c6981b4687724e8bbabf55c8103586b589bfbf0b2755bf3e71a584fef64ec9",
    ROOT / "autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py":
        "4dee5200cb0c1c72eb5e9c9d8c0cecc989423363fdcd08b232e0e6bf4ead7820",
    ROOT / "autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py":
        "f6dc80663b9e35743e48141796e21087861bd1ea2e83015c3ebab38b9616b6d6",
    ROOT / "autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py":
        "87d790b1a24e08cfd6f18b38c8ffd7a31d21209ded4ba221cbeda18a7d9ec531",
    ROOT / "autoresearch/flopscope_swarm/K_probe_strassen.py":
        "e68e478cdddc50f7d3a6cdd883a4bcbcb39faabbaaba02c323dd03dda1f5f65b",
}
for path, expected in bindings.items():
    assert path.is_file(), path
    assert sha256(path) == expected, path

reference = json.loads((HERE / "RECOVERED_REFERENCE_EVIDENCE.json").read_text())
candidate = json.loads((HERE / "RECOVERY_DESCENDANT_RESULT.json").read_text())
terminal = json.loads((HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json").read_text())

assert reference["call_order"] == [35, 279, 35]
assert candidate["call_order"] == [35, 279, 35]
assert len(reference["trials"]) == 3
assert len(candidate["trials"]) == 1
r = reference["trials"][0]
c = candidate["trials"][0]
assert (r["ordinal"], r["role_rank"]) == (0, 35)
assert (c["ordinal"], c["role_rank"]) == (0, 35)

assert terminal["node"] == "P2R0C1D5R1"
assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
assert terminal["process_exit_code"] == 1
assert terminal["candidate_calls_completed"] == 1
assert terminal["reference_estimator_calls_in_descendant"] == 0
assert terminal["failed_ordinal"] == 0
assert terminal["uncalled_ordinals"] == [1, 2]
assert terminal["uncalled_role_ranks"] == [279, 35]
assert terminal["access_declarations"] == {
    "board_opened": False,
    "names_opened": [],
    "network": False,
    "scores_opened": [],
    "submission": False,
    "target_columns_opened": [],
    "training": False,
}

rm = r["literal_meter"]
cm = c["literal_meter"]
assert rm["flops_used"] == 1_692_484_904_311
assert cm["flops_used"] == 1_320_876_655_991
assert rm["flops_used"] - cm["flops_used"] == 371_608_248_320
assert c["literal_operation_sum"] == cm["flops_used"]
assert c["measured_reference_minus_candidate_flops"] == 371_608_248_320

exact = {
    "reference_outer": 9.890440291725099,
    "candidate_outer": 24.356781082693487,
    "reference_backend": 8.859719954431057,
    "candidate_backend": 21.66671060444787,
    "reference_overhead": 0.648201156873256,
    "candidate_overhead": 0.8143845787271857,
    "reference_residual": 0.3825201806612313,
    "candidate_residual": 1.875688566826284,
    "reference_rss": 2_950_840_320,
    "candidate_rss": 11_953_569_792,
}
assert r["exact_outer_wall_seconds"] == exact["reference_outer"]
assert c["exact_outer_wall_seconds"] == exact["candidate_outer"]
assert rm["flopscope_backend_time_s"] == exact["reference_backend"]
assert cm["flopscope_backend_time_s"] == exact["candidate_backend"]
assert rm["flopscope_overhead_time_s"] == exact["reference_overhead"]
assert cm["flopscope_overhead_time_s"] == exact["candidate_overhead"]
assert rm["residual_wall_time_s"] == exact["reference_residual"]
assert cm["residual_wall_time_s"] == exact["candidate_residual"]
assert r["maximum_process_rss_bytes"] == exact["reference_rss"]
assert c["maximum_process_rss_bytes"] == exact["candidate_rss"]
assert c["drift_vs_exact_incumbent"]["final_layer_rms"] == 4.109649235540117e-8
assert c["drift_vs_exact_incumbent"]["final_layer_maximum_absolute"] == 2.384185791015625e-7
assert c["gates"]["residual_strictly_below_0_4"] is False
assert c["gates"]["deployment_residual_at_most_0_385"] is False
assert c["gates"]["rss_strictly_below_8_gib"] is False
assert c["gates"]["conjunctive_call_pass"] is False

ops = ("subtract", "stack", "add", "block", "matmul", "concatenate", "einsum")
expected_ops = {
    "subtract": (495, 1679, 639_848_448, 8_804_896_768),
    "stack": (150, 456, 270_661_632, 14_436_964_352),
    "add": (1024, 2060, 1_692_827_872, 7_104_954_592),
    "block": (0, 128, 0, 2_887_843_840),
    "matmul": (345, 352, 1_016_019_814_400, 1_064_706_721_792),
    "concatenate": (92, 132, 48_433_152, 300_091_392),
    "einsum": (174, 146, 663_005_442_048, 212_063_236_096),
}
for name in ops:
    rop = rm["operations"].get(name, {"calls": 0, "flop_cost": 0})
    cop = cm["operations"].get(name, {"calls": 0, "flop_cost": 0})
    assert (rop["calls"], cop["calls"], rop["flop_cost"], cop["flop_cost"]) == expected_ops[name]

glue = ("subtract", "stack", "add", "block")
assert sum(cm["operations"][name]["flop_cost"] for name in glue) == 33_234_659_552
assert sum(rm["operations"].get(name, {"flop_cost": 0})["flop_cost"] for name in glue) == 2_603_337_952
assert sum(cm["operations"][name]["flop_cost"] for name in glue) - sum(
    rm["operations"].get(name, {"flop_cost": 0})["flop_cost"] for name in glue
) == 30_631_321_600

# Frozen D5 source mechanics: fully nested stack/block CP3 and paired-channel contexts.
d5_source = (HERE / "estimator.py").read_text()
assert "def _cp3_winograd(a, b, levels, out=None):" in d5_source
assert "aa = fnp.stack((a11, a12, s4, a22, s1, s2, s3), axis=0)" in d5_source
assert "bb = fnp.stack((b11, b21, b22, t4, t1, t2, t3), axis=0)" in d5_source
assert "return fnp.block([[u1, u5], [u6, u7]])" in d5_source
assert "paired_left = fnp.stack((weight32_t, weight32_t), axis=0)" in d5_source
assert "fnp.empty((2, 2, width, _CP3_CAP), dtype=fnp.float32)" in d5_source
assert d5_source.count("fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32)") == 2

# Dedupe mechanics: Fable and the earlier Winograd spellings are whole-product
# recursive stack/block depth-3 forms.  K_probe is an assembly probe, not D6.
for rel in (
    "autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py",
    "autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py",
    "autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py",
):
    text = (ROOT / rel).read_text()
    assert "def _sample_strassen_batched(a, b, levels):" in text
    assert "STRASSEN_LEVELS = 3" in text
    assert text.count("aa = fnp.stack((") >= 1
    assert "return fnp.block([[c11, c12], [c21, c22]])" in text
winograd_s3 = (ROOT / "autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py").read_text()
assert "# CX STRASSEN-WINOGRAD-S3 BATCHED" in winograd_s3
assert "def _sample_strassen_batched(a, b, levels):" in winograd_s3
assert "aa = fnp.stack((a11, a12, s4, a22, s1, s2, s3), axis=0)" in winograd_s3
assert "return fnp.block([[c11, c12], [c21, c22]])" in winograd_s3
probe = (ROOT / "autoresearch/flopscope_swarm/K_probe_strassen.py").read_text()
assert "stack alternatives" in probe
assert "block alternatives" in probe
assert "def viacat():" in probe and "def viaset():" in probe

# D6 exact conservative recursive-capacity arena and strict pre-run gates.
incumbent_anchor = 3_421_503_488
eight_gib = 8 * 1024**3
from fractions import Fraction

def arena_elements(m: int, k: int, n: int, depth: int = 4) -> int:
    q = Fraction(7, 4)
    value = sum(q**j * (m * k + k * n) for j in range(1, depth + 1))
    value += q ** (depth - 1) * max(m * k, k * n)
    value += q**depth * m * n
    value += 3 * q ** (depth - 1) * m * n
    value += 2 * m * n
    assert value.denominator == 1
    return value.numerator

push_elements = arena_elements(1024, 1024, 15360)
read_elements = arena_elements(1024, 14336, 1024)
assert push_elements == 844_165_120
assert read_elements == 681_480_192
push_arena = 4 * push_elements
read_arena = 4 * read_elements
assert push_arena == 3_376_660_480
assert read_arena == 2_725_920_768
assert incumbent_anchor + push_arena == 6_798_163_968
assert incumbent_anchor + read_arena == 6_147_424_256
assert eight_gib - (incumbent_anchor + push_arena) == 1_791_770_624
smoke_increment_cap = push_arena + 512 * 1024**2
assert smoke_increment_cap == 3_913_531_392
assert incumbent_anchor + smoke_increment_cap < 7 * 1024**3
assert 7**4 == 2401

receipt = RECEIPT.read_text()
required_receipt_markers = (
    "D5R1 CLOSED_THIS_IMPLEMENTATION_FORM_NO_RETRY",
    "P2R0C1D6",
    "DESIGN_ONLY, UNIMPLEMENTED, UNMEASURED, AND NOT AUTHORIZED TO RUN",
    "sequential-channel, fully preallocated depth-4 Winograd arena",
    "58 leaf-batch matmul invocations",
    "844,165,120",
    "3,376,660,480",
    "6,798,163,968",
    "1,791,770,624",
    "3,913,531,392",
    "literal absolute peak process RSS must\nalso be `< 7 GiB`",
    "`<= 0.010` seconds",
    "[1,320,876,655,991, 1,359,519,832,584, 1,320,873,503,232]",
    "371,608,248,320",
    "residual wall strictly `< 0.4` seconds",
    "maximum process RSS strictly `< 8 GiB`",
    "active_work`: empty",
    "P2R0C1D`: remains `open`",
)
for marker in required_receipt_markers:
    assert marker in receipt, marker

print("PASS_D5R1_RESOURCE_CAUSAL_AND_D6_DESIGN")
