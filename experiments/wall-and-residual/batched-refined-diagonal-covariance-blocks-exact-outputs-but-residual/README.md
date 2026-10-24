# Batched refined diagonal covariance blocks: exact outputs but residual projection still fails

2026-09-28 · experiment · rejected · measured

Batching preserved the previous combined arithmetic exactly while reducing operations. The eight-network score passed1%, but early boundary residual remained above the unchanged limit.

Recorded measurement:

    Primitive537 versus1287 operations/helper;TRAIN8 adjusted-1.066788770%,8wins,CI[-1.311948933,-.830116284]%,raw+.093159022%,bill-1.166672059%;residual projection.381850751>.35;setup3.368s,wall89.907664s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_precision_g4_symdiag_batched_v2/candidate.py](code/hill_climb_precision_g4_symdiag_batched_v2/candidate.py)
- [code/hill_climb_precision_g4_symdiag_batched_v2/build.py](code/hill_climb_precision_g4_symdiag_batched_v2/build.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_hk15_blocked_v2/kernel_blocked.py](code/hill_climb_hk15_blocked_v2/kernel_blocked.py)
- [code/hill_climb_precision_g4_symdiag_batched_v2/assess.py](code/hill_climb_precision_g4_symdiag_batched_v2/assess.py)
- [code/hill_climb_precision_g4_symdiag_batched_v2/fragment.py](code/hill_climb_precision_g4_symdiag_batched_v2/fragment.py)
- [code/hill_climb_precision_g4_symdiag_batched_v2/probe.py](code/hill_climb_precision_g4_symdiag_batched_v2/probe.py)
- [code/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py](code/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py)
- [code/hill_climb_precision_g4_symdiag_v1/assess.py](code/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/hill_climb_precision_g4_symdiag_v1/build.py](code/hill_climb_precision_g4_symdiag_v1/build.py)
- [code/hill_climb_precision_g4_symdiag_v1/candidate.py](code/hill_climb_precision_g4_symdiag_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/assess.py](code/hill_climb_precision_symmetry_bundle_v1/assess.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/hill_climb_precision_symmetry_bundle_v1/build.py](code/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
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

- [handoff_archive_20260928T1402Z.md#L109](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L109)

<details>
<summary>All result files and records (6)</summary>

- [handoff_archive_20260928T1402Z.md](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-batched-refined-diagonal-covariance-blocks-exact-outputs-but-residual) · [Copy provenance](../../../PROVENANCE.json)

