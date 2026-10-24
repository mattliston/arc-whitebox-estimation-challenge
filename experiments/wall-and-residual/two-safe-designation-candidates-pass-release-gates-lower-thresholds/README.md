# Two safe designation candidates pass release gates; lower thresholds change only runtime routing

2026-10-16T09:03:20Z · experiment · inconclusive · measured

The canonical safe geometry and its lower-threshold variant completed full release without firing. A no-push alternative also passed, and record-based risk projections favored a robust secondary for best-of-two scoring.

Recorded measurement:

    safeP3.05%,warm42080<=42700,setup4.442s,slow4.585s,worst108.5s;lowerresidual.37/.39/.43s,wall108/115/121s;no-pushP13.36%,worst109.8s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_stack_v1/build_s2_v1.py](code/opus_stack_v1/build_s2_v1.py)
- [code/opus_stack_v1/build_stack_v1.py](code/opus_stack_v1/build_stack_v1.py)
- [code/opus_stack_v1/candidate_S1_gk4_ref_v1.py](code/opus_stack_v1/candidate_S1_gk4_ref_v1.py)
- [code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py](code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py)
- [code/opus_governor_v1/grade_v1.py](code/opus_governor_v1/grade_v1.py)
- [code/opus_governor_v1/test_census1.py](code/opus_governor_v1/test_census1.py)
- [code/opus_governor_v1/build_v1.py](code/opus_governor_v1/build_v1.py)
- [code/opus_governor_v1/build_v2.py](code/opus_governor_v1/build_v2.py)
- [code/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](code/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](code/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [code/opus_exact_v1/dm2/build_dm2_v1.py](code/opus_exact_v1/dm2/build_dm2_v1.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)
- [code/opus_governor_v1/candidate_GOV_v1.py](code/opus_governor_v1/candidate_GOV_v1.py)
- [code/opus_governor_v1/candidate_GOV_v2.py](code/opus_governor_v1/candidate_GOV_v2.py)
- [code/opus_governor_v1/candidate_GOV_v4f.py](code/opus_governor_v1/candidate_GOV_v4f.py)
- [code/opus_governor_v1/test_parity_v4f.py](code/opus_governor_v1/test_parity_v4f.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/opus_smtable_v1/cand_GOV3_smtable_v1.py](code/opus_smtable_v1/cand_GOV3_smtable_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4064](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4064)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4043](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4043)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4056](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4056)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4059](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4059)

<details>
<summary>All result files and records (24)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MORNING_DECISION_v1.md](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v1.md) · result
- [MORNING_DECISION_v2.md](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v2.md) · result
- [MORNING_DECISION_v3.md](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v3.md) · result
- [S2_RESULT_v1.json](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_RESULT_v1.json) · result
- [fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt) · result
- [fragment_safe_pb_five_full_key_v1.py.txt](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v1.py.txt) · result
- [fragment_safe_pb_five_full_key_v2.py.txt](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v2.py.txt) · result
- [S2_BUILD_PROOF_v1.json](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_BUILD_PROOF_v1.json) · result
- [fragment_census_v1.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_census_v1.py.txt) · result
- [GRADE_nt6v3_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt6v3_v1.json) · result
- [GRADE_nt_fe3v4f_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fe3v4f_v1.json) · result
- [GRADE_nt_fl6v4c_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fl6v4c_v1.json) · result
- [GRADE_nt_l6v4c_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_l6v4c_v1.json) · result
- [GRADE_nt_shed8v4d_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_shed8v4d_v1.json) · result
- [GRADE_psm_v2_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psm_v2_v1.json) · result
- [GRADE_psmf_v2_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psmf_v2_v1.json) · result
- [fragment_governor_v1.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v1.py.txt) · code
- [fragment_governor_v2.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v2.py.txt) · code
- [fragment_trace_v2.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v2.py.txt) · code
- [fragment_trace_v3.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v3.py.txt) · code
- [fragment_lazybind_v2.py.txt](../../arithmetic-kernels/unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · code
- [rqx_patch_v1.py.txt](../../arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/rqx_patch_v1.py.txt) · code
- [fragment_tree_r1_v1.py.txt](../../arithmetic-kernels/tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_r1_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-two-safe-designation-candidates-pass-release-gates-lower-thresholds) · [Copy provenance](../../../PROVENANCE.json)

