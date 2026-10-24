# Safe covariance geometry requires restoring a missing prebind key and keeping the list at 652

2026-10-16T08:50:43Z · incident · inconclusive · measured

The inherited safe fragment first failed an import assertion because the census-refitted list omitted a required key. Appending that key then failed the 90% binding gate; replacing a list entry restored compatibility.

Recorded measurement:

    missing(0,6,1024,896);653keys587bound89.9%FAIL;652keys587boundPASS;first43765/warm42080;risk3.05%;worst108.5s;bill+1.11%/+1.15%

## Programs

- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py)
- [build_safe_s2t_v1.py](build_safe_s2t_v1.py)
- [candidate_S2t_safe_v1.py](candidate_S2t_safe_v1.py)
- [judge_safe_v1.py](judge_safe_v1.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [fragment_covariance_two_strip_HELD_v1.py](fragment_covariance_two_strip_HELD_v1.py)
- [candidate_rtt2_safe_HELD_v2.py](candidate_rtt2_safe_HELD_v2.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

- [fragment_safe_five_full_HELD_v2.py](fragment_safe_five_full_HELD_v2.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py](../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py)
- [../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/stage_c_official120_v1.py](../../../instruments/release-pipeline-combines-boundary-gates-captured-setup-parity-prefire/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/stage_c_official120_v1.py)
- [../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py)
- [../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py)
- [../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_release_v1/s2t/candidate_S2t_cblazy_v1.py](../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_release_v1/s2t/candidate_S2t_cblazy_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_client.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/boundary_run.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/fastgate_pin_v1.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py](../../../experiments/arithmetic-kernels/large-tree-unfusing-eager-bind-time-fusion-rejected-after/code/opus_reexam_20261001_v1/release/server_light.py)
- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4059](../../TEAM_LOG_2026-10-01_to_10-22.md#L4059)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4053](../../TEAM_LOG_2026-10-01_to_10-22.md#L4053)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4058](../../TEAM_LOG_2026-10-01_to_10-22.md#L4058)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4064](../../TEAM_LOG_2026-10-01_to_10-22.md#L4064)

<details>
<summary>All result files and records (17)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MORNING_DECISION_v1.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v1.md) · result
- [MORNING_DECISION_v2.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v2.md) · result
- [MORNING_DECISION_v3.md](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/MORNING_DECISION_v3.md) · result
- [S2_RESULT_v1.json](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_RESULT_v1.json) · result
- [fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt) · result
- [fragment_safe_pb_five_full_key_v1.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v1.py.txt) · result
- [fragment_safe_pb_five_full_key_v2.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/fragment_safe_pb_five_full_key_v2.py.txt) · result
- [S2_BUILD_PROOF_v1.json](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/S2_BUILD_PROOF_v1.json) · result
- [SAFE_JUDGE_RESULT_safes2t_20261016_opus_v1.json](SAFE_JUDGE_RESULT_safes2t_20261016_opus_v1.json) · result
- [SAFE_JUDGE_RESULT_secsafe2_20261016_opus_v1.json](SAFE_JUDGE_RESULT_secsafe2_20261016_opus_v1.json) · result
- [SAFE_SOURCE_MANIFEST_v1.json](SAFE_SOURCE_MANIFEST_v1.json) · result
- [JUDGE_SAFE_safes2t_20261016_opus_v1.txt](JUDGE_SAFE_safes2t_20261016_opus_v1.txt) · result
- [JUDGE_SAFE_secsafe2_20261016_opus_v1.txt](JUDGE_SAFE_secsafe2_20261016_opus_v1.txt) · result
- [fragment_lazybind_v2.py.txt](../../../experiments/arithmetic-kernels/unbound-depth-six-tree-views-served-maximal-boxes-exact/fragment_lazybind_v2.py.txt) · code
- [rqx_patch_v1.py.txt](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/rqx_patch_v1.py.txt) · code
- [fragment_tree_r1_v1.py.txt](../../../experiments/arithmetic-kernels/tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_r1_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-safe-covariance-geometry-requires-restoring-missing-prebind-key-keeping) · [Copy provenance](../../../PROVENANCE.json)

