# Precision bundle plus core carry: slightly worse than the precision parent

2026-09-28 · experiment · rejected · measured

The core-carry addition preserved early layers but was slightly worse than the precision parent on the fixed two-case screen. Its gain against the older baseline still missed the1% gate.

Recorded measurement:

    TRAIN2 adjusted-.961509948488%,raw+.111934480963%,bill-1.079988987103%,2/2wins versus #332738;precision parent-.965657060929%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_precision_core_bundle_v1/assess.py](code/hill_climb_precision_core_bundle_v1/assess.py)
- [code/hill_climb_precision_core_bundle_v1/build.py](code/hill_climb_precision_core_bundle_v1/build.py)
- [code/hill_climb_precision_core_bundle_v1/candidate.py](code/hill_climb_precision_core_bundle_v1/candidate.py)
- [code/hill_climb_core_carry_consistency_v1/fragment.py](code/hill_climb_core_carry_consistency_v1/fragment.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess.py](code/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/hill_climb_precision_symmetry_bundle_v1/build.py](code/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/hill_climb_precision_symmetry_bundle_v1/precision.py](code/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/hill_climb_star32_q2_v1/assess.py](code/hill_climb_star32_q2_v1/assess.py)
- [code/hill_climb_star32_q2_v1/build.py](code/hill_climb_star32_q2_v1/build.py)
- [code/hill_climb_star32_q2_v1/candidate.py](code/hill_climb_star32_q2_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)

</details>

## Evidence

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L23](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L23)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-precision-bundle-plus-core-carry-slightly-worse-than-precision) · [Copy provenance](../../../PROVENANCE.json)

