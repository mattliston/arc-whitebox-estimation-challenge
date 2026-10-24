# Hidden complement readout adds .001660192% FLOPs and improves both host judges

2026-10-21T04:38:25Z · experiment · adopted · measured

The previously computed 128-dimensional hidden state supplied a complement beyond the existing 64-state readout. A fixed 90-network fit improved the old per-layer readout by -.189375277% Mac32 and -.237792381% Linux32.

Recorded measurement:

    Hidden-complement90 actual Mac32 completed04:23:10Z: -.559425049% adjusted versus422 /held-.653557698%,21/32lower, SE.208385,bill+.001660192%. Exact paired incremental versus old perlayer90 is-.189375277% all32 /held-.142060361%,18/32lower,SE.145618; receipt C/MAC32_INCREMENTAL_VS_PERLAYER90_v1.json (C=D/hidden_complement90_v1). VM131953 still running, strict watcher133287 and incremental-vs-56549 watcher136083 attached. First7 VMincrement-.0162239% is incomplete and not a verdict. Standalone fresh held; prefer exact stack fresh if stack wins.
    Parent-approved distinct per-layerhidden-complement90 screen frozen03:47:59Z (PREREGeaf6dbc3): originalstate/messagesunchanged, usealreadycomputedhidden128 nullspace64, residualizeandrefitbaselineinsideeach90LOO TRAINfold; fixedridge.1/damping.5 andincrementalgatesvsfold-fittedperlayer64. Officialcomplete15layerprice03:54:28Z PASS: +3,932,160FLOPs /+.0016601922% billvs337422, statebyteexact andmeanbindinglive. Fitterguard122514 submitted03:54:43Z, stilladmission-waiting04:01:25Z; donotclaimfitCPUrunning. Pass-onlysourcequalifierwaiter124680 iswired; no nativeautolaunch. Allnormalguard/source/protectedhashproofs remain.
    Fixed hidden-complement90 passed TRAIN90 incremental local-proxy LOO (-0.3417566242%, 90/90 lower); full GNN price +0.00166019% projected bill, state byte-exact. Source/import/tiny qualifications passed. Joint launched exact SCI cac374b1 original VM32 at 9:13:28 p.m. Pacific (PID131953; original shared36/BLAS4), parent owns concurrent Mac32. No adjusted/native gain claimed before those results. All candidate and fit receipts copied locally under hidden_complement90_v1.
    Hidden complement native piece now independently measured: Mac32 -0.559425049% adjusted versus422, paired incremental versus perlayer90 -0.1894%/held-0.1421%; Linux32 -0.472995853%/held-0.585113619%, paired -0.237792381%/held-0.299831547% (21/32 lower,SE.147374). Exact edge8+complement b54f81ff/4685e145 built via K/make_stack, full AST/parameter/live-binding plus official tiny qualification PASS. Parent owns Mac stack32 (9:35:53), joint owns Linux stack32 (9:39:00 PID143026), no duplicate work or additive inference.
    Standalone complement VM32 completed04:38:40Z: -.472995853% adjusted versus422 /held-.585113619%; exact incremental versus perlayer90 -.237792381% /held-.299831547%,21/32 lower. Local receipts A/cloud_linux_markline_v1/complement90_results_v1 SHA3e906526/8787cda7, independently rehashed by root. Conditional exact edge8+complement VM32 launched04:39:00Z PID143026, strictwatcher143045; all12 groups admitted by9:43PM. Its own conditional fresh4 continuation is being armed, no standalone redundant fresh.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/hidden_complement90_v1/check_hidden_complement_tiny_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/check_hidden_complement_tiny_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/qualify_passed_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/qualify_passed_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/run_qualify_pipeline_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/run_qualify_pipeline_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/run_fit_pipeline_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/run_fit_pipeline_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/price_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/price_hidden_complement_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/check_hidden_complement_import_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/check_hidden_complement_import_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py](code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6589](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6589)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6549](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6549)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6581](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6581)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6603](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6603)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6613](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6613)

<details>
<summary>All result files and records (19)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [MAC32_INCREMENTAL_VS_PERLAYER90_v1.json](MAC32_INCREMENTAL_VS_PERLAYER90_v1.json) · result
- [COMPLEMENT_FIRST7_VS_PERLAYER90_v1.json](COMPLEMENT_FIRST7_VS_PERLAYER90_v1.json) · result
- [COPY_20261021T043204Z.json](COPY_20261021T043204Z.json) · result
- [COMPLEMENT_FIRST7_VS422_v1.json](COMPLEMENT_FIRST7_VS422_v1.json) · result
- [W3_FULL32_EXACT_v1.json](W3_FULL32_EXACT_v1.json) · result
- [scores_hidden_complement90_per_layer_v1__hidden_complement90_per_layer_FULL32_VS_337422.json](scores_hidden_complement90_per_layer_v1__hidden_complement90_per_layer_FULL32_VS_337422.json) · result
- [EDGE8_COMPLEMENT_LAUNCH_v1.json](EDGE8_COMPLEMENT_LAUNCH_v1.json) · result
- [EDGE8_COMPLEMENT_INCREMENT_GATE_v1.json](EDGE8_COMPLEMENT_INCREMENT_GATE_v1.json) · result
- [W3_SCORE_GUARD_v1.json](W3_SCORE_GUARD_v1.json) · result
- [EDGE8_COMPLEMENT_SCORER_LAUNCH_v1.json](EDGE8_COMPLEMENT_SCORER_LAUNCH_v1.json) · result
- [FULL32_VS_PERLAYER90_v1.json](FULL32_VS_PERLAYER90_v1.json) · result
- [FIT_RECIPE_v1.json](FIT_RECIPE_v1.json) · result
- [ROOT_EDGE8_COMPLEMENT_MAC32_REVIEW_v1.json](ROOT_EDGE8_COMPLEMENT_MAC32_REVIEW_v1.json) · result
- [MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json](MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json) · result
- [QUALIFY_RECIPE_v1.json](QUALIFY_RECIPE_v1.json) · result
- [ROOT_SOURCE_REVIEW_v1.json](ROOT_SOURCE_REVIEW_v1.json) · result
- [PREREG_v1.json](PREREG_v1.json) · result
- [STATUS_CURRENT.md](STATUS_CURRENT.md) · record

</details>

## Related work

- shipped in: [Submission #337473: Hidden-complement readout; one hosted residual failure](../../../submissions/337473/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-hidden-complement-readout-adds-001660192-flops-improves-both-host) · [Copy provenance](../../../PROVENANCE.json)

