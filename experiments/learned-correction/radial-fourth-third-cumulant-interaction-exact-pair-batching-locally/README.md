# Radial fourth-to-third cumulant interaction with exact pair batching: locally validated candidate

2026-09-05T02:57Z · experiment · informational · measured

These are LOCAL scores, not official leaderboard results. Official best remains329790, adjusted1.43699341254e-8.

Recorded measurement:

    | Group | Mean raw MSE | Mean utilization | Mean adjusted | | --- | ---: | ---: | ---: | | Public8 | 2.45023949397e-8 | .21895 | 5.37165714446e-9 | | Virgin6 | 2.00427699535e-8 | .217366667 | 4.37040032467e-9 | | All14 | 2.25911270885e-8 | .218271429 | 4.94254707884e-9 | These are LOCAL scores, not official leaderboard results. Official best remains329790, adjusted1.43699341254e-8. The preceding source/full-pair learned model, before scalar-final savings, scored public5.530e-9 /virgin4.532e-9 on identical panels. Thus this step saves about2.9% /3.6% adjusted. The earlier compact scalar-final model was public5.8500e-9 /virgin5.0310e-9. The fixed source/full-pair training had heldout one-step5.703532e-9, and full-loop development raw2.04444475e-8 / 2.00406279e-8 against stored organizer1e9-sample means onr35/r100. The matched final-only fit uses its own frozen prefix,200nets/fixed160/40, 40epochs. Heldout raw2.24756231e-8 versus the original final head2.23118555e-8 (+0.73%). Trace-aware scalar reference includes k3 = CPdiag + stardiag + 3*diag(Bpre)*bpre, and separates active covariance from cold diagonal variance. Full-loop development raw2.07256816e-8 / 2.05830925e-8. Frozen full-matrix host means match within9.91e-8 and final inputs within8.35e-6. No current-layer target feedback enters any input. Native official-version35/100/35 INCLUDING both heads passes; repeated outputs byte-identical; max mean1.632e-6, max final-delta6.0e-8, peak7.324GB. Actual numpy-free remote gate: utilization.218223239, wall66.27s, residual.23575527s, peak6.654GB. Whest.16 tiny plumbing validation passes. Local archivefullpair_scalar_final_batched_d6_v1_whest016.tar.gz contains onlyestimator.py+manifest.json. Its estimator is byte-identical to tested source. Bound evidence:verified_local_fullpair_scalar_final_v1.json.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_batched_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_batched_d6_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_cpcubes_release_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_cpcubes_release_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_fullpair_scalar_release_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_fullpair_scalar_release_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_gaussfirst_release_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_gaussfirst_release_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_squarecache_release_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/audit_squarecache_release_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_learned_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_learned_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_secondorder_unlearned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_secondorder_unlearned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_cpcubes_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_cpcubes_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_gaussfirst_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_gaussfirst_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_squarecache_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_squarecache_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_squareopt_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_fullpair_scalar_final_squareopt_v1.py)

<details>
<summary>Additional programs and dependencies (31)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_secondorder_training_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_all_compact_tensorfeat_rows_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_secondorder_rows_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_tensorfeat_rows_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_radial_rows.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/packed_arena_generic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/packed_arena_generic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/price_direct_kernel_schedules_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/price_direct_kernel_schedules_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_birthonly_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_birthonly_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_secondorder_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_tensorfeat_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_cubic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_features.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_features.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_mehler.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_secondorder_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_secondorder_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_compact_secondorder_closedloop_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_compact_secondorder_closedloop_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_radial_closedloop.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_radial_closedloop.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_compact_tensorfeat_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_compact_tensorfeat_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/export_gnn_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/train_gnn_oracle.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L8](../../flop-accounting/exact-pair-batching-local-resource-validation-passed-combined-radial/RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L8)
- [RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L19](../../flop-accounting/exact-pair-batching-local-resource-validation-passed-combined-radial/RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L19)
- [RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L1](../../flop-accounting/exact-pair-batching-local-resource-validation-passed-combined-radial/RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L1)
- [RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L29](../../flop-accounting/exact-pair-batching-local-resource-validation-passed-combined-radial/RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md#L29)

<details>
<summary>All result files and records (2)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md](../../flop-accounting/exact-pair-batching-local-resource-validation-passed-combined-radial/RESEARCH_RECEIPTS_2026_09_04_1955_PDT.md) · record
- [verified_local_fullpair_scalar_final_v1.json](verified_local_fullpair_scalar_final_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-radial-fourth-third-cumulant-interaction-exact-pair-batching-locally) · [Copy provenance](../../../PROVENANCE.json)

