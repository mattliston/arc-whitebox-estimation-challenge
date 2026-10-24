# Four-term Gaussian evaluation alone: incremental gain has confidence crossing zero

2026-09-28 · experiment · rejected · measured

The previously frozen Gaussian-only source was measured separately after the combined residual failure. Its fixed eight-network incremental gate failed.

Recorded measurement:

    Native2 component-.409294376%,raw-.290168560%,bill-.119440694%,2wins;TRAIN8 component-.155078089%,5wins,CI[-.368537128,+.061785909]%;whole-baseline-.972989438%,8wins,raw+.044888525%,bill-1.023872782%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_precision_gaussian4_v1/candidate.py](code/hill_climb_precision_gaussian4_v1/candidate.py)
- [code/hill_climb_precision_gaussian4_v1/build.py](code/hill_climb_precision_gaussian4_v1/build.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_precision_gaussian4_v1/assess.py](code/hill_climb_precision_gaussian4_v1/assess.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess.py](code/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/hill_climb_precision_symmetry_bundle_v1/build.py](code/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/hill_climb_precision_symmetry_bundle_v1/precision.py](code/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/hill_climb_star32_q2_v1/assess.py](code/hill_climb_star32_q2_v1/assess.py)
- [code/hill_climb_star32_q2_v1/build.py](code/hill_climb_star32_q2_v1/build.py)
- [code/hill_climb_star32_q2_v1/candidate.py](code/hill_climb_star32_q2_v1/candidate.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)

</details>

## Evidence

- [handoff_archive_20260928T1402Z.md#L117](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L117)

<details>
<summary>All result files and records (5)</summary>

- [handoff_archive_20260928T1402Z.md](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-four-term-gaussian-evaluation-alone-incremental-gain-has-confidence) · [Copy provenance](../../../PROVENANCE.json)

