# Depth-three covariance diagonal products: adjusted gain 0.355945% misses the fixed 0.5% gate

2026-09-29 · experiment · rejected · measured

Five active diagonal product depths changed from5 to3 after source pricing selected one fixed implementation. The native endpoint failed its unchanged mean-gain gate despite two wins.

Recorded measurement:

    TRAIN2 adjusted-.355945%,raw+1.169%,bill-1.50787,2wins;fixed gain gate.5%;primitive738303136F/378calls.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/build.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/build.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d3_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
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
- [code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [handoff_root_archive_20260930T024527Z.md#L106](handoff_root_archive_20260930T024527Z.md#L106)

<details>
<summary>All result files and records (4)</summary>

- [handoff_root_archive_20260930T024527Z.md](handoff_root_archive_20260930T024527Z.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-29-depth-three-covariance-diagonal-products-adjusted-gain-0-355945) · [Copy provenance](../../../PROVENANCE.json)

