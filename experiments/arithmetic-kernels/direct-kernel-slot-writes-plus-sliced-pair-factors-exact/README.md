# Direct kernel-slot writes plus sliced pair factors: exact warm FLOPs -0.0971% with lazy tree views

2026-10-16T07:20:57Z · experiment · inconclusive · measured

Two direct data-movement edits eliminated a scaled-operand copy and a factor stack. Combined with lazy tree boxing they preserved all boundary predictions and passed release gates.

Recorded measurement:

    directedit-95.6MFLOPs,+5ops;combinedwarm-238.0M/-.0971%;first-253.9M/-.1038%;slowfirst-478.6M/-.195%;requests-2/-5/-21;risk13.36%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_exact_v1/repeat_census_v1.py](code/opus_exact_v1/repeat_census_v1.py)
- [code/opus_exact_v1/repeat_census_v2.py](code/opus_exact_v1/repeat_census_v2.py)
- [code/opus_exact_v1/site_census_v1.py](code/opus_exact_v1/site_census_v1.py)
- [code/opus_stack_v1/build_s2_v1.py](code/opus_stack_v1/build_s2_v1.py)
- [code/opus_stack_v1/build_stack_v1.py](code/opus_stack_v1/build_stack_v1.py)
- [code/opus_stack_v1/candidate_S1_gk4_ref_v1.py](code/opus_stack_v1/candidate_S1_gk4_ref_v1.py)
- [code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py](code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py)
- [code/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](code/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](code/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [code/opus_exact_v1/dm2/build_dm2_v1.py](code/opus_exact_v1/dm2/build_dm2_v1.py)
- [code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/opus_governor_v1/candidate_GOV_v4f.py](code/opus_governor_v1/candidate_GOV_v4f.py)
- [code/opus_governor_v1/test_parity_v4f.py](code/opus_governor_v1/test_parity_v4f.py)
- [code/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py](code/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py)
- [code/opus_smtable_v1/cand_GOV3_smtable_v1.py](code/opus_smtable_v1/cand_GOV3_smtable_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3999](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3999)

<details>
<summary>All result files and records (16)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EXACT_DM_CENSUS_SUMMARY_v1.json](../../../records/research/2026-10-16-data-movement-census-1-58-bill-real-repeated-buffer/EXACT_DM_CENSUS_SUMMARY_v1.json) · result
- [guard_census_gov3.json](../../../records/research/2026-10-16-data-movement-census-1-58-bill-real-repeated-buffer/guard_census_gov3.json) · result
- [guard_repeat2_gov3.json](../../../records/research/2026-10-16-data-movement-census-1-58-bill-real-repeated-buffer/guard_repeat2_gov3.json) · result
- [guard_repeat_gov3.json](../../../records/research/2026-10-16-data-movement-census-1-58-bill-real-repeated-buffer/guard_repeat_gov3.json) · result
- [MORNING_DECISION_v1.md](MORNING_DECISION_v1.md) · result
- [MORNING_DECISION_v2.md](MORNING_DECISION_v2.md) · result
- [MORNING_DECISION_v3.md](MORNING_DECISION_v3.md) · result
- [S2_RESULT_v1.json](S2_RESULT_v1.json) · result
- [fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt](fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt) · result
- [fragment_safe_pb_five_full_key_v1.py.txt](fragment_safe_pb_five_full_key_v1.py.txt) · result
- [fragment_safe_pb_five_full_key_v2.py.txt](fragment_safe_pb_five_full_key_v2.py.txt) · result
- [S2_BUILD_PROOF_v1.json](S2_BUILD_PROOF_v1.json) · result
- [fragment_lazybind_v2.py.txt](../unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · code
- [rqx_patch_v1.py.txt](rqx_patch_v1.py.txt) · code
- [fragment_tree_r1_v1.py.txt](../tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_r1_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-direct-kernel-slot-writes-plus-sliced-pair-factors-exact) · [Copy provenance](../../../PROVENANCE.json)

