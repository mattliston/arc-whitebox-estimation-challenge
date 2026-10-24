# Exact-view and grouped-push runtime-protected stack qualifies but is held below the competitive target

2026-10-16T08:32:48Z · decision · inconclusive · measured

The complete candidate passed release gates, exact captured-setup parity and prospective fresh-pair evaluation. It was not fired because the declared score projection failed the owner competitive threshold.

Recorded measurement:

    warmbill-.228%,SE.020;first-.037%;xMSE-.276%,SE.116treatedrounding;projected-.433%vs336342=1.4738e-9;target1.463531e-9;gate13.13%,worst111.8s,setup4.446s

## Programs

- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py)
- [../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py)
- [../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4046](../../TEAM_LOG_2026-10-01_to_10-22.md#L4046)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4042](../../TEAM_LOG_2026-10-01_to_10-22.md#L4042)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4048](../../TEAM_LOG_2026-10-01_to_10-22.md#L4048)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MORNING_DECISION_v1.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v1.md) · result
- [MORNING_DECISION_v2.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v2.md) · result
- [MORNING_DECISION_v3.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v3.md) · result
- [S2_RESULT_v1.json](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_RESULT_v1.json) · result
- [fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt) · result
- [fragment_safe_pb_five_full_key_v1.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v1.py.txt) · result
- [fragment_safe_pb_five_full_key_v2.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v2.py.txt) · result
- [S2_BUILD_PROOF_v1.json](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_BUILD_PROOF_v1.json) · result
- [fragment_lazybind_v2.py.txt](../../../experiments/arithmetic-kernels/unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · code
- [rqx_patch_v1.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/rqx_patch_v1.py.txt) · code
- [fragment_tree_r1_v1.py.txt](../../../experiments/arithmetic-kernels/tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_r1_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-exact-view-grouped-push-runtime-protected-stack-qualifies-but) · [Copy provenance](../../../PROVENANCE.json)

