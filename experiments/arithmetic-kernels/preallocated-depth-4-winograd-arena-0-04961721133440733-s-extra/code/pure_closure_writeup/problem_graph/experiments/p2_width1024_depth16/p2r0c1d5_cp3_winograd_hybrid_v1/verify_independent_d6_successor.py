#!/usr/bin/env python3
"""Read-only verifier for the independent P2R0C1D6 successor review.

No estimator is imported or called. No target, score, board, production, or
submission path exists in this verifier.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
AUDIT = HERE / "INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


bindings = {
    HERE / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md":
        "808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7",
    HERE / "verify_d5r1_resource_successor.py":
        "6214a1e493a9c543815bbd9f3020023e11021f369449ba815715346cc1bcbe5a",
    HERE / "estimator.py":
        "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
    ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py":
        "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    HERE / "RECOVERED_REFERENCE_EVIDENCE.json":
        "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    HERE / "RECOVERY_DESCENDANT_RESULT.json":
        "a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549",
    HERE / "RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json":
        "8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0",
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
    ROOT / "autoresearch/flopscope_swarm/J_STRASSEN.md":
        "decba5d3ca9d948733401310fa18e574d613f76a56e334fcb38d032d1d7b7b0f",
    ROOT / "autoresearch/experiments/cands/fable_clean_td52best_20260810/estimator.py":
        "6a2a25a153a43e9e46ddf637ad0bb07b325180a08979f9c4878b3d469547a612",
    ROOT / "autoresearch/experiments/cands/util_probe_strassen/estimator.py":
        "3e3602868bcd8e0daf26f3cef13948bc552da60414434d4c9c1a0f4a98250b3a",
    ROOT / "autoresearch/experiments/probes/fable_billhunt_meter_20260805/FINDINGS.md":
        "e38c051ea10e95d77825939dfcd5f56b2fdb56631c605d789eae485d920c846f",
    ROOT / "autoresearch/experiments/cands/native_vrmc_winograd_native_probe/WINOGRAD_GATE_RECEIPT.md":
        "06615f9da62898f613521deca285bfb1091f80a088d00914338024cdf3ebb121",
    HERE / "PREREGISTER.md":
        "5b7d8d778c684082cc07b04040f7e9e4b9fc8d60c1ab6298aaa7703e36306740",
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
assert terminal["status"] == "FAIL_TERMINAL_NO_RETRY"
assert terminal["candidate_calls_completed"] == 1
assert terminal["uncalled_ordinals"] == [1, 2]
assert terminal["reference_estimator_calls_in_descendant"] == 0

residuals = [
    trial["literal_meter"]["residual_wall_time_s"]
    for trial in reference["trials"]
]
assert residuals == [
    0.3825201806612313,
    0.3858333397656679,
    0.37958328472450376,
]
strict_margin = 0.4 - max(residuals)
assert abs(strict_margin - 0.0141666602343321) < 1e-15

r0 = reference["trials"][0]
c0 = candidate["trials"][0]
assert c0["literal_meter"]["flops_used"] == 1_320_876_655_991
assert (
    r0["literal_meter"]["flops_used"]
    - c0["literal_meter"]["flops_used"]
) == 371_608_248_320
assert c0["literal_meter"]["residual_wall_time_s"] == 1.875688566826284
assert c0["maximum_process_rss_bytes"] == 11_953_569_792
assert c0["gates"]["residual_strictly_below_0_4"] is False
assert c0["gates"]["rss_strictly_below_8_gib"] is False
assert c0["drift_vs_exact_incumbent"]["final_layer_rms"] == 4.109649235540117e-8
assert c0["drift_vs_exact_incumbent"]["final_layer_maximum_absolute"] == 2.384185791015625e-7

d5_increment = c0["literal_meter"]["residual_wall_time_s"] - residuals[0]
assert abs(d5_increment - 1.4931683861650527) < 1e-15
required_elimination = 1.0 - strict_margin / d5_increment
assert abs(required_elimination - 0.9905123491994652) < 1e-15

# Exact CP3 roster: two channels for push l=1..15 and read l=1..14.
w = 1024
roster = (
    [(w, w, w * level) for level in range(1, 16) for _ in range(2)]
    + [(w, w * level, w) for level in range(1, 15) for _ in range(2)]
)
assert len(roster) == 58


def winograd_materialized_bill(m: int, k: int, n: int, depth: int) -> int:
    """D5 law: leaf matmul, 15 adds/level, stack inputs, block/concat output."""
    scale = 2**depth
    leaf = 7**depth * (
        2 * (m // scale) * (k // scale) * (n // scale)
        - (m // scale) * (n // scale)
    )
    bill = leaf
    for level in range(1, depth + 1):
        quadrants = 4**level
        parents = 7 ** (level - 1)
        bill += parents * (
            4 * (m * k // quadrants)
            + 4 * (k * n // quadrants)
            + 7 * (m * n // quadrants)
        )
        bill += 7**level * ((m * k + k * n) // quadrants)
        bill += parents * (m * n // (4 ** (level - 1)))
    return bill


dense_bill = sum(2 * m * k * n - m * n for m, k, n in roster)
assert dense_bill == 966_086_623_232
kernel_bills = {
    depth: sum(winograd_materialized_bill(m, k, n, depth) for m, k, n in roster)
    for depth in range(1, 5)
}
assert kernel_bills == {
    1: 847_842_377_728,
    2: 746_611_408_896,
    3: 661_941_616_640,
    4: 594_693_332_992,
}
common_left_stack = 20_971_520
removed_incumbent_reshapes = 235_929_600
savings = {
    depth: dense_bill - bill - common_left_stack + removed_incumbent_reshapes
    for depth, bill in kernel_bills.items()
}
assert savings == {
    1: 118_459_203_584,
    2: 219_690_172_416,
    3: 304_359_964_672,
    4: 371_608_248_320,
}

# Allocation-volume screen for depth-only versions of D5's dynamic tree.
volume4 = sum(Fraction(7, 4) ** level for level in range(1, 5))
predicted_depth_only = {}
for depth in (1, 2, 3):
    ratio = sum(Fraction(7, 4) ** level for level in range(1, depth + 1)) / volume4
    predicted_depth_only[depth] = residuals[0] + float(ratio) * d5_increment
assert abs(predicted_depth_only[1] - 0.5161744138284527) < 1e-15
assert abs(predicted_depth_only[2] - 0.7500693218710904) < 1e-15
assert abs(predicted_depth_only[3] - 1.1593854109457062) < 1e-15

# Dispatch comparison among schedules retaining depth-4 arithmetic.
assert 58 * 1 == 58
assert 58 * 7 == 406
assert 58 * 7**2 == 2_842
assert 58 * 7**4 == 139_258
assert 58 - 37 == 21


def arena_elements(m: int, k: int, n: int) -> int:
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    cap = sum(q**level * (a + b) for level in range(1, 5))
    cap += q**3 * max(a, b)
    cap += q**4 * c
    cap += 3 * q**3 * c
    cap += 2 * c
    assert cap.denominator == 1
    return cap.numerator


push_elements = arena_elements(1024, 1024, 15360)
read_elements = arena_elements(1024, 14336, 1024)
assert push_elements == 844_165_120
assert read_elements == 681_480_192
push_bytes = 4 * push_elements
read_bytes = 4 * read_elements
assert push_bytes == 3_376_660_480
assert read_bytes == 2_725_920_768

anchor = 3_421_503_488
seven_gib = 7 * 1024**3
eight_gib = 8 * 1024**3
assert anchor + push_bytes == 6_798_163_968
assert anchor + read_bytes == 6_147_424_256
assert eight_gib - (anchor + push_bytes) == 1_791_770_624
assert seven_gib - (anchor + push_bytes) == 718_028_800
increment_cap = push_bytes + 512 * 1024**2
assert increment_cap == 3_913_531_392
assert anchor + increment_cap == 7_335_034_880
assert seven_gib - (anchor + increment_cap) == 181_157_888

# Dedupe mechanics and the Phase-2 prohibition.
for rel in (
    "autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py",
    "autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py",
    "autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py",
):
    source = (ROOT / rel).read_text()
    assert "STRASSEN_LEVELS = 3" in source
    assert "def _sample_strassen_batched(a, b, levels):" in source
    assert "fnp.stack" in source and "fnp.block" in source
prior_winograd = (
    ROOT / "autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py"
).read_text()
assert "def _sample_strassen_batched(a, b, levels):" in prior_winograd
assert "fnp.stack" in prior_winograd and "fnp.block" in prior_winograd
preregister = (HERE / "PREREGISTER.md").read_text()
assert "Forbidden: `tensordot`, TD52 or high-rank singleton padding" in preregister

text = AUDIT.read_text()
for marker in (
    "AUTHORIZE_PREPARATION_ONLY for P2R0C1D6",
    "does not authorize an estimator call",
    "There is not yet an external\nconstraint",
    "58 leaf-batch `matmul` dispatches",
    "844,165,120",
    "6,798,163,968",
    "3,913,531,392",
    "only `181,157,888` bytes\nbelow 7 GiB",
    "does **not** certify the later deployment gate",
    "No raw MSE was measured",
    "`active_work`: empty",
):
    assert marker in text, marker

print("PASS_INDEPENDENT_REDTEAM_D6_SUCCESSOR")
