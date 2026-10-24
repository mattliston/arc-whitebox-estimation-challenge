# Width-eight edge plus hidden complement: -.191409749% Mac32 incremental over #337466

2026-10-21T04:46:50Z · experiment · adopted · measured

The combined edge and hidden-complement readout was tested directly rather than inferred by adding gains. Linux32 and its own fresh confirmation also improved before submission #337473.

Recorded measurement:

    Exact b54f81ff/4685e145 edge8+hidden-complement stack actual Mac32 is -0.724003464% adjusted versus #337422, held25 -0.716179178%, 24/32 lower, SE .194989; raw -.571260799%, bill -.153627589%. Root independently verified all12 original guards complete/retired, no errors or swap growth, unchanged7.9GB caps. Actual paired incremental versus last-fired #337466 SCI434b is -.191409749% all32 /held-.095849159%, 19/32 lower. Receipt C/MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json; root audit C/ROOT_EDGE8_COMPLEMENT_MAC32_REVIEW_v1.json. Local projection1.41234946e-9 is not a public score; target1.41 still needs more gain. No new fire or additive extrapolation.
    Actual b54 gain versus NEW337466: Mac32 -.191409749% /held-.095849159%; Linux32 -.126288014% /held-.199195740%,18/32lower,SE.153338. Mac legacy frozenv4a bar remains-1.731738654% overall/held-1.711745658% (well below-.35%). Old422-relative numbers are historical only: Mac-.724003464%,VM-.581045887%. Ownfresh4 finished all4lower versus422 at-.819607936%; actual337466 increment/custody copy pending, do not subtract means. Root rehashed82 VM32 custodyfiles and audited12 unchangedoriginal guards in A/edge8_complement_release_v1/ROOT_NATIVE32_VM_CUSTODY_v1.json; Macall12 reviewedearlier. Exact VM337466comparison dc301bb3 and VM422a3c2ead0 local L/edge8_complement_results_v1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/agent_astra_joint_v1/edge8_complement_release_v1/run_isolated_root_v1.py](code/agent_astra_joint_v1/edge8_complement_release_v1/run_isolated_root_v1.py)
- [code/agent_astra_joint_v1/edge8_complement_release_v1/analyze_quiet_v1.py](code/agent_astra_joint_v1/edge8_complement_release_v1/analyze_quiet_v1.py)
- [code/agent_astra_joint_v1/edge8_complement_release_v1/prepare_v1.py](code/agent_astra_joint_v1/edge8_complement_release_v1/prepare_v1.py)
- [code/agent_astra_joint_v1/qualify_runtime_companion_isolated_v3.py](code/agent_astra_joint_v1/qualify_runtime_companion_isolated_v3.py)
- [code/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py](code/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py)
- [code/candidate_rel_astra_o21_edge8_hidden_complement90_v1.py](code/candidate_rel_astra_o21_edge8_hidden_complement90_v1.py)
- [code/compare_best_v1.py](code/compare_best_v1.py)
- [code/compare_relative_v1.py](code/compare_relative_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6611](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6611)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6635](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6635)

<details>
<summary>All result files and records (22)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json](../hidden-complement-readout-adds-001660192-flops-improves-both-host/MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json) · result
- [ROOT_EDGE8_COMPLEMENT_MAC32_REVIEW_v1.json](../hidden-complement-readout-adds-001660192-flops-improves-both-host/ROOT_EDGE8_COMPLEMENT_MAC32_REVIEW_v1.json) · result
- [ROOT_NATIVE32_VM_CUSTODY_v1.json](ROOT_NATIVE32_VM_CUSTODY_v1.json) · result
- [SECOND_FINAL10_AUTHORIZATION_v1.json](SECOND_FINAL10_AUTHORIZATION_v1.json) · result
- [EXACT_AUTHORIZATION_QUOTE_CORRECTION_v1.json](EXACT_AUTHORIZATION_QUOTE_CORRECTION_v1.json) · result
- [ISOLATION_quiet_v1.json](ISOLATION_quiet_v1.json) · result
- [ISOLATION_release_v1.json](ISOLATION_release_v1.json) · result
- [ROOT_RUNTIME_REVIEW_v1.json](ROOT_RUNTIME_REVIEW_v1.json) · result
- [MANIFEST_v1.json](MANIFEST_v1.json) · result
- [ROOT_FRESH_VM_CUSTODY_v1.json](ROOT_FRESH_VM_CUSTODY_v1.json) · result
- [ROOT_PREFIRE_DECISION_v1.json](ROOT_PREFIRE_DECISION_v1.json) · result
- [PREPARED_v1.json](PREPARED_v1.json) · result
- [scores_edge8_hidden_complement90_v1__edge8_hidden_complement90_FULL32_VS_337422.json](scores_edge8_hidden_complement90_v1__edge8_hidden_complement90_FULL32_VS_337422.json) · result
- [edge8_hidden_complement90_FULL32.json](edge8_hidden_complement90_FULL32.json) · result
- [EDGE8_COMPLEMENT_FULL32_VS337466_v1.json](EDGE8_COMPLEMENT_FULL32_VS337466_v1.json) · result
- [edge8_hidden_complement90_FULL32_VS_PARENT.json](edge8_hidden_complement90_FULL32_VS_PARENT.json) · result
- [edge8_hidden_complement90_FIRST7_VS_PARENT.json](edge8_hidden_complement90_FIRST7_VS_PARENT.json) · result
- [COMPLETE.json](COMPLETE.json) · result
- [WATCH_MANIFEST_v1.json](WATCH_MANIFEST_v1.json) · result
- [STATE_v1.json](STATE_v1.json) · result
- [edge8_hidden_complement90_FIRST7.json](edge8_hidden_complement90_FIRST7.json) · result

</details>

## Related work

- shipped in: [Submission #337473: Hidden-complement readout; one hosted residual failure](../../../submissions/337473/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-width-eight-edge-plus-hidden-complement-191409749-mac32-incremental) · [Copy provenance](../../../PROVENANCE.json)

