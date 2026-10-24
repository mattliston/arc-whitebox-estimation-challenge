# Four-term Gaussian evaluation with refined covariance diagonals: residual-time qualification fails

2026-09-28 · experiment · rejected · measured

Four-term Gaussian evaluation and diagonal blocks[128,64,64] improved the combined score but added many client operations. Both original and one fresh paired boundary test failed the fixed residual projection.

Recorded measurement:

    Combined TRAIN8 adjusted-1.077494%,8wins,negativeCI;original residual projection.492650751>.35,requests+15701/+16659;fresh paired projection.467450751>.35;wall97.90756<=105.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_precision_g4_symdiag_v1/candidate.py](code/hill_climb_precision_g4_symdiag_v1/candidate.py)
- [code/hill_climb_precision_g4_symdiag_v1/build.py](code/hill_climb_precision_g4_symdiag_v1/build.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_precision_g4_symdiag_v1/assess.py](code/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/hill_climb_precision_gaussian4_v1/candidate.py](code/hill_climb_precision_gaussian4_v1/candidate.py)
- [code/hill_climb_precision_symdiag_v1/candidate.py](code/hill_climb_precision_symdiag_v1/candidate.py)
- [code/hill_climb_precision_symdiag_v1/symmetry.py](code/hill_climb_precision_symdiag_v1/symmetry.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess.py](code/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/hill_climb_precision_symmetry_bundle_v1/build.py](code/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [code/hill_climb_precision_symmetry_bundle_v1/precision.py](code/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/hill_climb_star32_q2_v1/assess.py](code/hill_climb_star32_q2_v1/assess.py)
- [code/hill_climb_star32_q2_v1/build.py](code/hill_climb_star32_q2_v1/build.py)
- [code/hill_climb_star32_q2_v1/candidate.py](code/hill_climb_star32_q2_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)

</details>

## Evidence

- [handoff_archive_20260928T1402Z.md#L122](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L122)
- [handoff_archive_20260928T1402Z.md#L112](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L112)

<details>
<summary>All result files and records (6)</summary>

- [handoff_archive_20260928T1402Z.md](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-four-term-gaussian-evaluation-refined-covariance-diagonals-residual-time) · [Copy provenance](../../../PROVENANCE.json)

