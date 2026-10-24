# Width-eight edge plus per-layer 90-network readout: -.533265988% Mac32 and -.441676584% fresh

2026-10-21T03:53:27Z · experiment · adopted · measured

The actual combined stack passed Mac32, Linux32 and its own four-network fresh confirmation. It preserved runtime functions and changed only qualified edge and readout parameters before submission #337466.

Recorded measurement:

    Actual edge8+perlayer90 original Mac32 COMPLETE03:50:32Z: adjusted -0.533265988%, held25 -0.620917355%, SE .140579,22/32lower; raw -.378570826%, bill -.155287781%. This exceeds the last verified J2W gap -.4664333625% only on this local result, with a thin margin; NOT a public pass or fire-ready bank. VM stack113227 was21/32 at03:50:21; strictwatcher114631 and conditionalfresh116769 remain active. SCI 434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4, REL 660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200. OriginalMac output D/local_innovation90_v1/science32_edge8_local90_perlayer_parallel5_v1. Root session21976 COMPLETE/resumed77740.
    Combined stack actual Mac32 -.533265988% adjusted /held-.620917355%; Linux32 -.454913209% /held-.484789529%,26/32lower. Both versus currentbest422, not additive gains. Linux slightly short of J2W; no confirmed public clearance. Standalone head fresh4 -.258160639% is separate and cannot substitute stack fresh. Stack fresh PID123158, all4 cold samehost422 controls,1/4 complete at04:02:27Z, remaining3 active/no failures; strict joint watcher owns result. Public scan03:54:08Z through337461 unchanged.
    Evidence versus337422: original Mac32 -.533265988% adjusted /held-.620917355%; Linux32 -.454913209% /held-.484789529%; own fresh4 -.441676584% (3/4lower, SE.379310834,raw-.288620746%,bill-.153487607%). J2W requires-.4664333625% by latestverifiedboard, so near-tie remains an estimate, not publicclearance. Exactfreshresulta273c86f and COPY40files in L/edge8_local90_fresh_confirmation_v1; actualVM32 all82files/12groups/sourcepredictions in L/edge8_local90_vm32_receipts_v1, copyreceipt944e7f43. Root independently rehashedall122files and auditedall20originalVMguards clean/retired/zeroSwap/OOM/unchangedcaps.
    Quietsourcepair original35,150,200 completed04:10:09Z, exactownSCI bytes/FLOPs, bothoriginal125s/free50/7.5GB guardsPASS,free>=80%,zeroswapgrowth. Mean wall-.0409319%,backend-.111256%,requests0%, residualinstructions+.837302% (smallincrease acknowledged, not runtimegain). No newbindingkeys/missing orskippedcompiles/all32rescue0. Rootruntime review1154b0ac acceptedunchangedgates. Originalrelease_min completed04:11:51Z: fiveofficialsetupstartsPASS max3.541504s, StageAPASS exactsource/archive, prefirezero neweinsums andonlypreviouslyapprovedgcimportexception. Rootprefiredecision A/edge8_local90_release_v1/ROOT_PREFIRE_DECISION_v1.json pinsallchecks+authorization; originalsource/gates unchanged. Bothisolationrunsfinishedandresumed77740sameidentity; sessions26922/34981closed. Fire session4413closedSUCCESS.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/local_innovation90_v1/run_qualification_pipeline_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/run_qualification_pipeline_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/run_pipeline_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/run_pipeline_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/qualify_passed_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/qualify_passed_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/fit_local90_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/fit_local90_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/qualify_edge_stacks_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/qualify_edge_stacks_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/prepare_edge_stacks_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/prepare_edge_stacks_v1.py)
- [code/astra_oct20_bill_v1/local_innovation90_v1/check_edge_stack_v1.py](code/astra_oct20_bill_v1/local_innovation90_v1/check_edge_stack_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/run_isolated_root_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/run_isolated_root_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/analyze_quiet_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/analyze_quiet_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/prepare_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/edge8_local90_release_v1/prepare_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/astra_oct20_bill_v1/edge8_current_rank16_stack_v1/tail_exact337422_from_v4a_v1.py](code/astra_oct20_bill_v1/edge8_current_rank16_stack_v1/tail_exact337422_from_v4a_v1.py)
- [code/astra_oct20_bill_v1/shared_innovation_head_v1/check_exploratory_import_v1.py](code/astra_oct20_bill_v1/shared_innovation_head_v1/check_exploratory_import_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/qualify_runtime_companion_isolated_v3.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/qualify_runtime_companion_isolated_v3.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)
- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py](code/fable_oct19_kc_v1/candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py)
- [code/fable_oct19_kc_v1/compare_best_v1.py](code/fable_oct19_kc_v1/compare_best_v1.py)
- [code/fable_oct19_kc_v1/compare_relative_v1.py](code/fable_oct19_kc_v1/compare_relative_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6555](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6555)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6565](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6565)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6567](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6567)

<details>
<summary>All result files and records (34)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EDGE_STACK_PAYLOAD_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/EDGE_STACK_PAYLOAD_v1.json) · result
- [LOCAL_SOURCE_REVIEW_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/LOCAL_SOURCE_REVIEW_v1.json) · result
- [PREREG_v1.json](../../../instruments/ninety-network-training-capture-excludes-judge-held-out-control/PREREG_v1.json) · result
- [ISOLATION_quiet_v1.json](../current-message-edge-context-refit-current-consumer-projector-metric/ISOLATION_quiet_v1.json) · result
- [ISOLATION_release_v1.json](../current-message-edge-context-refit-current-consumer-projector-metric/ISOLATION_release_v1.json) · result
- [ROOT_RUNTIME_REVIEW_v1.json](../current-message-edge-context-refit-current-consumer-projector-metric/ROOT_RUNTIME_REVIEW_v1.json) · result
- [MANIFEST_v1.json](../current-message-edge-context-refit-current-consumer-projector-metric/MANIFEST_v1.json) · result
- [ROOT_PREFIRE_DECISION_v1.json](../current-message-edge-context-refit-current-consumer-projector-metric/ROOT_PREFIRE_DECISION_v1.json) · result
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result
- [CANDIDATE_PLAN_v1.json](CANDIDATE_PLAN_v1.json) · result
- [BANK_PLAN_v1.json](BANK_PLAN_v1.json) · result
- [CONFIG_v1.json](CONFIG_v1.json) · result
- [RESULT_v1.json](RESULT_v1.json) · result
- [COPY_RECEIPT_v1.json](COPY_RECEIPT_v1.json) · result
- [part04_GUARD.json](part04_GUARD.json) · result
- [part09_GUARD.json](part09_GUARD.json) · result
- [part10_GUARD.json](part10_GUARD.json) · result
- [part01_GUARD.json](part01_GUARD.json) · result
- [part06_GUARD.json](part06_GUARD.json) · result
- [part12_GUARD.json](part12_GUARD.json) · result
- [part03_GUARD.json](part03_GUARD.json) · result
- [part11_GUARD.json](part11_GUARD.json) · result
- [COMPLETE.json](COMPLETE.json) · result
- [part05_GUARD.json](part05_GUARD.json) · result
- [part08_GUARD.json](part08_GUARD.json) · result
- [part02_GUARD.json](part02_GUARD.json) · result
- [COPY_RECEIPT_v1-944e7f43.json](COPY_RECEIPT_v1-944e7f43.json) · result
- [part07_GUARD.json](part07_GUARD.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result

</details>

## Related work

- shipped in: [Submission #337466: Edge width 8 with per-layer correction fitted on 90 networks](../../../submissions/337466/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-width-eight-edge-plus-per-layer-90-network-readout) · [Copy provenance](../../../PROVENANCE.json)

