# Inherited-young negative-row trace with exact newborn transport: adjusted mean is flat

2026-09-28 · experiment · rejected · measured

The newest newborn transport was kept exact while only inherited young columns were cropped. The old local diagnostic did not predict current endpoint behavior.

Recorded measurement:

    TRAIN2 adjusted+.0005764391%,raw+1.2220190435%,bill-1.2061584068%,1/2wins;rank35+.9660563203%,rank100-.9077847205%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/build.py](code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/build.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v2/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/candidate_dead_tensor_conditional_mean_v2.py](code/codex_resume_20260926_v2/candidate_dead_tensor_conditional_mean_v2.py)
- [code/codex_resume_20260926_v2/candidate_dead_tensor_trace_v1.py](code/codex_resume_20260926_v2/candidate_dead_tensor_trace_v1.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/dead_tensor_conditional_mean_fragment_v2.py](code/codex_resume_20260926_v2/dead_tensor_conditional_mean_fragment_v2.py)
- [code/codex_resume_20260926_v2/dead_tensor_trace_fragment_v1.py](code/codex_resume_20260926_v2/dead_tensor_trace_fragment_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_inherited_young_negative_trace_v1/probe.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [POST_S33_SOURCE_SEARCH_20260928_v1.md#L42](../../../instruments/boundary-feasibility-assessor-separates-receipt-schema-errors-unchanged-runtime/POST_S33_SOURCE_SEARCH_20260928_v1.md#L42)

<details>
<summary>All result files and records (5)</summary>

- [POST_S33_SOURCE_SEARCH_20260928_v1.md](../../../instruments/boundary-feasibility-assessor-separates-receipt-schema-errors-unchanged-runtime/POST_S33_SOURCE_SEARCH_20260928_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-inherited-young-negative-row-trace-exact-newborn-transport-adjusted) · [Copy provenance](../../../PROVENANCE.json)

