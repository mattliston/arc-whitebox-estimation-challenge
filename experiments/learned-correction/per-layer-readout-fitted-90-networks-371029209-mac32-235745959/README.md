# Per-layer readout fitted on 90 networks: -.371029209% Mac32 and -.235745959% Linux32

2026-10-21T03:46:31Z · experiment · adopted · measured

A fixed local-error fit changed only existing readout coefficients, adding no inference operations. Both native full-32 and held-out results were negative, while a separate shared-64 fit remained too weak.

Recorded measurement:

    Current best #337422 and J2W gap -0.4664333625% remain. No submission fired; ten UTC21 submissions unused. Root local90 per-layer960 is a NEW MATERIAL CANDIDATE, not a qualified bank yet: original Mac32 completed03:34:53Z at -0.371029209% adjusted / held25 -0.512993796%, bill0, SE.144836,20/32 lower. First7 was +0.135987%, so do not infer full result from pilot. SCI56549dd5bb95ad7bdc8a6614e9c5013bdc7009c14392d7239b80982f0fcbf88c. VM full32 PID107162 pending (24/32 at03:39); standalone fresh watcher111569 will dispatch only on clean Linux32 <=-.10%/heldnegative. Original edge8 bank remains -0.1506657% Mac32/-0.2140704% Linux32/-0.2338808% fresh4; never sum it with head gain.
    TRAIN90 capture completed 90/90 at 03:17:27Z. Root LOCAL-innovation90 fits BOTH passed frozen whole-network LOO: shared64 proxy -0.5909170%, 80/90 lower, five negative blocks, physical correction/head RMS .095374 overall; per-layer960 proxy -1.7109169%, 90/90 lower, five negative blocks, physical .158826 overall/.820526 max layer. These are normalized local-error proxies, NOT terminal MSE or adjusted gains. Fit d0ad9967ea0927ea85bfbd218da18eb020ec7768b5c634369ae38e6cb6a1f412, original guard PASS/no swap, both existing heads only/no added operations. Qualification completed 03:20:54Z with all30 official tiny head cases per arm and NumPy-free imports, source/literal/full AST checks. SCI shared5337c31bebbfcfe5e2333f06a05bf782ab09bc0ab49131519ba093f757f7bbc3, REL4b42734fe602c0ba7b52211bbb92e1478d16ec3650665c38d1c3a2e979ba5195; SCI perlayer56549dd5bb95ad7bdc8a6614e9c5013bdc7009c14392d7239b80982f0fcbf88c, RELb86a26e85c0f1e168a253290228af23acea92609ae316f204a588c96a5bf8c25. Both candidates, fit, and receipts copied back to D/local_innovation90_v1; independent copied-source review PASS03:26:51Z. VM32 shared PID107128 and perlayer PID107162 launched03:23:42/44, original jobs12/shared36/BLAS4; strict422 watchers attached. Mac perlayer original32 started03:28:08Z, root session19412, D/local_innovation90_v1/science32_local_innovation90_per_layer960_parallel5_v1. Shared64 Mac next. BEST422 PINNED while these run. Fresh/edge8 stack preparation authorized; native gain must pass full32 and held before promotion.
    Root local90 shared64 original Mac32 completed03:43:08Z at -0.031089069% adjusted / held -0.013420012%, SE.109093,17/32 lower, bill effectively0. Weak/no bank; fresh HOLD written03:44:35 before any fresh dispatch, fixed stack prepared only and not authorized for native absent contrary full evidence. VM107128 finishes normally. Root fit90 proxy screens are not score claims.
    Standalone perlayer90 confirmed on Linux full32: adjusted -.235745959%, held -.286641251%, SE .104517,21/32lower; raw -.233784018%,bill -.001978468%. Strict receipt3ee5da6d. Its fresh92–95 launched03:44:35 PID117885,2/4 at03:50:21; no bank claim until confirmed. Shared64 Linux-.087462119%/held-.110730852%,Mac-.031089069%/held-.013420012%, so weak/no bank, fresh HOLD beforedispatch. L/local90_results_v1 contains exactreports/launchreceipt.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/local_innovation90_v1/run_qualification_pipeline_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/run_qualification_pipeline_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/run_pipeline_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/run_pipeline_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/qualify_passed_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/qualify_passed_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/fit_local90_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/fit_local90_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/qualify_edge_stacks_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/qualify_edge_stacks_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/prepare_edge_stacks_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/prepare_edge_stacks_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/check_edge_stack_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/check_edge_stack_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/edge8_current_rank16_stack_v1/tail_exact337422_from_v4a_v1.py](code/astra_oct20_bill_v1/edge8_current_rank16_stack_v1/tail_exact337422_from_v4a_v1.py)
- [code/astra_oct20_bill_v1/shared_innovation_head_v1/check_exploratory_import_v1.py](code/astra_oct20_bill_v1/shared_innovation_head_v1/check_exploratory_import_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6518](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6518)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6498](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6498)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6520](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6520)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6536](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6536)

<details>
<summary>All result files and records (20)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fresh4_92_95_local_innovation90_per_layer960_conditional_markline_blas4_v1_LAUNCH.json](fresh4_92_95_local_innovation90_per_layer960_conditional_markline_blas4_v1_LAUNCH.json) · result
- [scores_local_innovation90_shared64_v1__local_innovation90_shared64_FULL32_VS_337422.json](scores_local_innovation90_shared64_v1__local_innovation90_shared64_FULL32_VS_337422.json) · result
- [scores_old_vpair75_v1__msg8_old_vpair75_FULL32_VS_337422.json](scores_old_vpair75_v1__msg8_old_vpair75_FULL32_VS_337422.json) · result
- [EDGE8_PERLAYER_VM32_EXACT_v1.json](EDGE8_PERLAYER_VM32_EXACT_v1.json) · result
- [EDGE8_PERLAYER_FRESH4_LAUNCH_v1.json](EDGE8_PERLAYER_FRESH4_LAUNCH_v1.json) · result
- [scores_local_innovation90_per_layer960_v1__local_innovation90_per_layer960_FULL32_VS_337422.json](scores_local_innovation90_per_layer960_v1__local_innovation90_per_layer960_FULL32_VS_337422.json) · result
- [local_innovation90_shared64_QUALIFIED_v1.json](local_innovation90_shared64_QUALIFIED_v1.json) · result
- [PERLAYER_FRESH4_EXACT_v1.json](PERLAYER_FRESH4_EXACT_v1.json) · result
- [msg8_old_vpair75_QUALIFIED_v1.json](msg8_old_vpair75_QUALIFIED_v1.json) · result
- [STATE_v1.json](STATE_v1.json) · result
- [local_innovation90_per_layer960_QUALIFIED_v1.json](local_innovation90_per_layer960_QUALIFIED_v1.json) · result
- [CURRENT_SOURCE7_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/PLAN_v1.json) · result
- [RESULT32_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/RESULT32_v1.json) · result
- [EDGE_STACK_PAYLOAD_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/EDGE_STACK_PAYLOAD_v1.json) · result
- [LOCAL_SOURCE_REVIEW_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/LOCAL_SOURCE_REVIEW_v1.json) · result
- [PREREG_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/PREREG_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337466: Edge width 8 with per-layer correction fitted on 90 networks](../../../submissions/337466/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-per-layer-readout-fitted-90-networks-371029209-mac32-235745959) · [Copy provenance](../../../PROVENANCE.json)

