# Feature and pair grouping plus kernel scratch compaction reduce requests with exact arithmetic

2026-10-20T21:36:20Z · experiment · adopted · measured

Feature grouping reduced 42 requests per network for +401408 FLOPs; pair grouping reduced 70 requests for +114688 warm FLOPs. Kernel scratch and output-coverage changes passed tiny parity, but native source integration initially failed import and required a repaired runtime companion.

Recorded measurement:

    Feature grouping already in new32 sources: tiny official checks byte exact, -42 requests/net for +401408 FLOPs (~+.00017% bank bill). Pair-group tail b095e57d prepared, tiny official exact, -70 warm requests/net for +114688 FLOPs (~+.00005% bill); whole-network parity still required before release attachment. SMDIRECT private internal scratch compaction tail590dd3b0 tiny36 cases/precompile cache pass, unchanged FLOPs, -44 calls/rank35; separate fixed-output run-cover optimization under audit. None of these runtime changes has demonstrated a platform residual margin yet.
    Exact runtime companions BUILT, NOT NATIVE-QUALIFIED: K/RUNTIME_GROUPED_CANDIDATES_PREPARED_v1.json. FG+pair+SM/AP7de468b9, no-governor last. Base SCI08fd3570/REL10d0ca41; EH14 SCI3233908a/REL94f3007a; GK4 SCI310c1f2a/RELa3e79962; EH14+GK4 SCI7c49fbb2/REL0d664d4f. Independent A/SMDIRECT_COMPACT_INDEPENDENT_AUDIT_v1.json PASS. SM/AP tiny36cases, precompile/cache/stridedpadding, unchangedFLOPs, projected-72calls onrank35. Pair-70requests/+114688warm FLOPs; FG-42/+401408. Whole-network byte/tariff parity and original release gates STILL REQUIRED; don't claim platform residual margin. V1 two-rank helper held; agent preparing v2 THREE ranks35,150,200 matching science sequence to avoid known rank200 warmorder offset.
    Choice: EH14runtime3sciencee77b617d/releasef8ec4d68/fullarchiveb5a29755 alsoqualifies, transferred32-.2787798%/held-.2862537% withwhole3byte/tariffparity; essentiallytiebutmeasuredquietwall+3.502%,threadresidualinstructions+1.234%vsbank. Choseexistingmicroedgefallbackwithbetterruntimeevidence(+1.600%wall/+.360%threadinstr), notnegligible.002ppscientificincrement. RT3pairfixpreservesrawnonlinearand_tr_clearing wrapper; oldRT323/94failedimportassert0predictionsandisinvalid. All8RT3fullimportsPASSofficialNumPyfreeclient,119.309swhole3guardPASS,originalreleasegatesPASS. NoRT3fire. Bankwasgatedbeforedeadline; futurefireONLYverifiedprojectionbelowJ2W, notanotherincrementalbank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/check_materialize_v1.py](code/astra_oct20_bill_v1/check_materialize_v1.py)
- [code/astra_oct20_bill_v1/check_smcompact_ap_v1.py](code/astra_oct20_bill_v1/check_smcompact_ap_v1.py)
- [code/astra_oct20_bill_v1/check_smcompact_v1.py](code/astra_oct20_bill_v1/check_smcompact_v1.py)
- [code/astra_oct20_bill_v1/prepare_smcompact_ap_v1.py](code/astra_oct20_bill_v1/prepare_smcompact_ap_v1.py)
- [code/astra_oct20_bill_v1/prepare_smcompact_v1.py](code/astra_oct20_bill_v1/prepare_smcompact_v1.py)
- [code/astra_oct20_bill_v1/tail_copy_all_exact_complete_v1.py](code/astra_oct20_bill_v1/tail_copy_all_exact_complete_v1.py)
- [code/astra_oct20_bill_v1/tail_ecut_radical_v1.py](code/astra_oct20_bill_v1/tail_ecut_radical_v1.py)
- [code/astra_oct20_bill_v1/tail_edge_hidden14_v1.py](code/astra_oct20_bill_v1/tail_edge_hidden14_v1.py)
- [code/astra_oct20_bill_v1/tail_flatload_v1.py](code/astra_oct20_bill_v1/tail_flatload_v1.py)
- [code/astra_oct20_bill_v1/tail_micro_v1.py](code/astra_oct20_bill_v1/tail_micro_v1.py)
- [code/astra_oct20_bill_v1/tail_sm_direct_v1.py](code/astra_oct20_bill_v1/tail_sm_direct_v1.py)
- [code/astra_oct20_bill_v1/tail_smdirect_compact_ap_complete_v1.py](code/astra_oct20_bill_v1/tail_smdirect_compact_ap_complete_v1.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/astra_oct20_bill_v1/tail_smdirect_compact_internal_v1.py](code/astra_oct20_bill_v1/tail_smdirect_compact_internal_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_features_grouped_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_features_grouped_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_gk4_114_microedge_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_gk4_114_microedge_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_pair_grouped_current_v1.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/tail_pair_grouped_current_v1.py)
- [code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_astra_sci_v1.py](code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_astra_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_gk4p_rt_astra_sci_v1.py](code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_gk4p_rt_astra_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_rt_astra_sci_v1.py](code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_eh14_rt_astra_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_gk4p_rt_astra_sci_v1.py](code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_gk4p_rt_astra_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_rt_astra_sci_v1.py](code/fable_oct19_kc_v1/candidate_micro_copyall_ecut_rt_astra_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_eh14_gk4p_rt_astra.py](code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_eh14_gk4p_rt_astra.py)
- [code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_eh14_rt_astra.py](code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_eh14_rt_astra.py)
- [code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_gk4p_rt_astra.py](code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_gk4p_rt_astra.py)
- [code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_rt_astra.py](code/fable_oct19_kc_v1/candidate_rel_micro_copyall_ecut_rt_astra.py)
- [code/fable_oct19_kc_v1/candidate_rel_v4a_sci_v1.py](code/fable_oct19_kc_v1/candidate_rel_v4a_sci_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6327](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6327)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6334](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6334)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6348](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6348)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RUNTIME_GROUPED_CANDIDATES_PREPARED_v1.json](RUNTIME_GROUPED_CANDIDATES_PREPARED_v1.json) · result
- [SMDIRECT_COMPACT_INDEPENDENT_AUDIT_v1.json](SMDIRECT_COMPACT_INDEPENDENT_AUDIT_v1.json) · result

</details>

## Related work

- shipped in: [Submission #337421: Layer-14 edge width reduction with selective flat-upload restoration](../../../submissions/337421/README.md)

[Research graph](../../../graph/index.html#node=2026-10-20-feature-pair-grouping-plus-kernel-scratch-compaction-reduce-requests) · [Copy provenance](../../../PROVENANCE.json)

