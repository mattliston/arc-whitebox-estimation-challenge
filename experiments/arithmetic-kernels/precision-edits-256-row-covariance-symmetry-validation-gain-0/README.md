# Precision edits with 256-row covariance symmetry: validation gain 0.7988805% stays below 1%

2026-09-28 · experiment · inconclusive · measured

Eight reversible precision edits and a fixed256-row symmetry fragment were combined on the exact-cost parent. Explicit unchanged-source extensions preserved the failed initial1% gate and confirmed a positive subthreshold component.

Recorded measurement:

    Native2 adjusted-.9656570609%,raw+.1079910891%,bill-1.0802605597%,2/2wins;TRAIN8-.819181720280%,8/8wins,CI[-1.114302450755%,-.528279532456%];VALID40-.7988805%,38wins,negativeCI.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/build.py](code/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_precision_broad_v1/assess_broad.py](code/hill_climb_precision_broad_v1/assess_broad.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess.py](code/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/hill_climb_precision_symmetry_bundle_v1/precision.py](code/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/hill_climb_star32_q2_v1/assess.py](code/hill_climb_star32_q2_v1/assess.py)
- [code/hill_climb_star32_q2_v1/build.py](code/hill_climb_star32_q2_v1/build.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/hill_climb_star32_q2_v1/candidate.py](code/hill_climb_star32_q2_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/native_eval.py](code/native_eval.py)

</details>

## Evidence

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L15](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L15)
- [handoff_archive_20260928T1402Z.md#L136](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L136)

<details>
<summary>All result files and records (8)</summary>

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md) · record
- [RESULT_train8.md](RESULT_train8.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [RESULT.md](RESULT.md) · record
- [broad_assessment.json](broad_assessment.json) · result
- [handoff_archive_20260928T1402Z.md](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-precision-edits-256-row-covariance-symmetry-validation-gain-0) · [Copy provenance](../../../PROVENANCE.json)

