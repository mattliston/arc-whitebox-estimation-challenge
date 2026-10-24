# Public-fire runtime thresholds optimized for expected leaderboard improvement per slot

2026-10-16T11:21:20Z · decision · adopted · measured

The objective counted public failure or approximate emergency routing as loss of a small board gain. Threshold-only variants were fully released, and the expanded-grid candidate with revised exact thresholds became the later submission #336506 source.

Recorded measurement:

    FULL.40s,EXACT.42s,EXACTwall116s;swapgain/fire.262/.240%vs.217/.178%;expandedgridgain/fire.375/.339%;cleanprojection-.42%vs336412=1.4697e-9;L6Ppublic8.0%,Pany15.4%

## Programs

- [build_pubmaxa_pg2_v1.py](build_pubmaxa_pg2_v1.py)
- [build_pubmaxa_pubgov_v1.py](build_pubmaxa_pubgov_v1.py)
- [build_swappub_v1.py](build_swappub_v1.py)
- [board_watch_v1.py](board_watch_v1.py)
- [cand_pubmaxa_pg2_v1_release.py](cand_pubmaxa_pg2_v1_release.py)
- [cand_pubmaxa_pubgov_v1_release.py](cand_pubmaxa_pubgov_v1_release.py)
- [cand_swappub_v1_release.py](cand_swappub_v1_release.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_s2_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/build_stack_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S1_gk4_ref_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_stack_v1/candidate_S2_f384_cb_gk4_t167r7_rqx_r1_v1.py)
- [../2026-10-13-leaderboard-slice-end-local-score-1-52127e-9-0/a11_rescan_astra_v2.py](../2026-10-13-leaderboard-slice-end-local-score-1-52127e-9-0/a11_rescan_astra_v2.py)

<details>
<summary>Additional programs and dependencies (16)</summary>

- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/a5_scan_driver.py)
- [../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py](../../../instruments/public-leaderboard-scanner-merges-results-alerts-changes-exceeding-2/code/scan_submissions.py)
- [../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py](../../../experiments/join-sketch/layer-10-join-rank-384-initial-single-network-adjusted/code/lead/runs/astra_full384_smirnov_v1/candidate_full384_smirnov_HELD_v2.py)
- [../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](../2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py](../../../experiments/arithmetic-kernels/kernel-dispatch-expanded-167-shapes-layer-one-read-reallocation/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_unseen_shapes_v1/candidate_smirnov_unseen_table_R7_HELD_v2.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_exact_v1/dm2/build_dm2_v1.py)
- [../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](../../../experiments/wall-and-residual/removing-low-value-grouped-read-plans-worst-wall-109/code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/candidate_GOV_v4f.py)
- [../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py](../../../experiments/arithmetic-kernels/direct-kernel-slot-writes-plus-sliced-pair-factors-exact/code/opus_governor_v1/test_parity_v4f.py)
- [../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_pubmax_v1/cand_pubmax_A_parity.py](../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_pubmax_v1/cand_pubmax_A_parity.py)
- [../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_pubmax_v1/cand_pubmax_A_release.py](../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_pubmax_v1/cand_pubmax_A_release.py)
- [../../../experiments/arithmetic-kernels/wall-capped-read-grid-swap-12-zero-saving-plans/code/opus_releaseidea_v1/cand_swap_v1_parity.py](../../../experiments/arithmetic-kernels/wall-capped-read-grid-swap-12-zero-saving-plans/code/opus_releaseidea_v1/cand_swap_v1_parity.py)
- [../../../experiments/arithmetic-kernels/wall-capped-read-grid-swap-12-zero-saving-plans/code/opus_releaseidea_v1/cand_swap_v1_release.py](../../../experiments/arithmetic-kernels/wall-capped-read-grid-swap-12-zero-saving-plans/code/opus_releaseidea_v1/cand_swap_v1_release.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py)
- [../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_stack_v1/assemble_final_v2.py](../../../experiments/wall-and-residual/extra-read-grid-join-kernels-lose-their-tariff-gains/code/opus_stack_v1/assemble_final_v2.py)
- [../../../experiments/wall-and-residual/later-approximate-wall-triggers-recover-modeled-0-116-public/code/cand_pubmax_A_v4h_release.py](../../../experiments/wall-and-residual/later-approximate-wall-triggers-recover-modeled-0-116-public/code/cand_pubmax_A_v4h_release.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4158](../../TEAM_LOG_2026-10-01_to_10-22.md#L4158)

<details>
<summary>All result files and records (17)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [board_watch_state_v1.json](board_watch_state_v1.json) · result
- [EVIDENCE_pubmaxa_v1.txt](EVIDENCE_pubmaxa_v1.txt) · result
- [EVIDENCE_pubmaxapg2_v1.txt](EVIDENCE_pubmaxapg2_v1.txt) · result
- [EVIDENCE_pubmaxapg_v1.txt](EVIDENCE_pubmaxapg_v1.txt) · result
- [EVIDENCE_swappub_v1.txt](EVIDENCE_swappub_v1.txt) · result
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

## Related work

- shipped in: [Submission #336506: uncapped strided reads with residual routing: 11 networks failed](../../../submissions/336506/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-public-fire-runtime-thresholds-optimized-expected-leaderboard-improvement-per) · [Copy provenance](../../../PROVENANCE.json)

