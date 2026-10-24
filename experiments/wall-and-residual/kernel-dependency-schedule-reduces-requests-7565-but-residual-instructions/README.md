# Kernel dependency schedule reduces requests .7565% but residual instructions only .0923764%

2026-10-21T06:29:02Z · experiment · rejected · measured

The exact three-network dependency schedule preserved predictions and FLOPs but did not create sufficient measured runtime margin. It was closed without a repeat timing run.

Recorded measurement:

    DAG quiet completed06:15:35Z: all3exactbytes/FLOPs,requests-.7565%,instructions-.0923764%mean(onecase+.1322%),wall-.3465%. CLOSED as insufficientrepair; no rerun. Squared+SMoffSCI95e04ad5/REL5df51227 independentlysource-reviewed3acca71a, fullimports+36tinyreuse+AST proofPASS. ActualVM32controller188478/start3087977 launchedonce06:25:58Z via canonicalmarkline,strict188764; localincrementcallback23729 measuresactualpaidbillvsown155c. RootMac originalquietfresh466control→5df51227 session47026 started06:25:04Z,controlfinished06:27:38Z, candidateactive. Original77740queue pausedbyidentity and willrestore. NootherMacnative. Combinedsquared+SMoff+NPoffSCI b090d3c3/REL5c0d38a9 qualifiedsourceonly; no measuredruntimeclaim/additivity.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/tail_sm_direct_v1.py](code/astra_oct20_bill_v1/tail_sm_direct_v1.py)
- [code/astra_oct20_bill_v1/tail_smdirect_compact_ap_complete_v1.py](code/astra_oct20_bill_v1/tail_smdirect_compact_ap_complete_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/check_sm_v1.py](code/astra_oct20_vm_frontier_v1/runtime_current337466_v1/check_sm_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/build_v1.py](code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/build_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/candidate_smdag_science_v1.py](code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/candidate_smdag_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/check_smdag_v1.py](code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/check_smdag_v1.py)
- [code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/price_metadata_v2.py](code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/price_metadata_v2.py)
- [code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/tail_smdag_schedule_v1.py](code/astra_oct20_vm_frontier_v1/runtime_smdag_schedule_v1/tail_smdag_schedule_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_disable_fast_rescue_astra_v1.py)
- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6748](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6748)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-kernel-dependency-schedule-reduces-requests-7565-but-residual-instructions) · [Copy provenance](../../../PROVENANCE.json)

