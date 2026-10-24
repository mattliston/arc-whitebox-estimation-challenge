# First-128 hidden readout: -.367205943% Mac32 and -.4015029588562216% Linux32 versus #337466

2026-10-21T06:29:02Z · experiment · adopted · measured

The added first-hidden readout passed full-32 judges on both hosts including actual FLOP debits. The exact parent increments were -.174040677% Mac32 and -.27235166970202945% Linux32.

Recorded measurement:

    First128Mac32 finished06:24:10Z; source4da68821,REL e491c4ae. Rootaudit9fb293c7 all32/12originalguards/zerorescue/compilesPASS. Actualsamehostvs466 adjusted-.367205943%,held25-.347306137%,23/32lower,SE.150838%,bill+.003241137%. Exactvsb54-.174040677%,held-.249929441%; originalv4a-1.90438197%. Observedsystemswapgrowthmax2.674GiB,minfree33%; originalconditional2GiB+free<25 and6GiBhardrulespassed. Do not claimzeroMacswap. Auditv1 mistakenlyinheritedstricterunconditional2GiBexpectation; failurepreserved, v2matchesunchangedoriginalrule; no guard/harness/result changed. VMfirst128175873/strict176113 stillfinishing, last23/32rows; no bankprojectionclaim.
    First128 SCI4da68821 completed original VM32 at 06:27:33Z. Actual paired adjusted versus latest verified BEST337466 is -0.4015029588562216% all32 (22 lower, SE0.1378525966%) and -0.5061495063269867% held25; exact b54 increment -0.27235166970202945% all32 / -0.30407431579163546% held25, using measured per-rank U and FLOPs, including every bill debit. H1/native32_vm_custody_v1/INDEPENDENT_CUSTODY_AUDIT_v1.json SHA f5de32b792aa06abea939d225cbc8f59c2a2c6ca327fdc54ea2d4a415d245c29 independently rehashed95 files, source, all32 prediction/FLOP rows and12 original guards; zero rescue/missing/skipped/owner miss/swap/OOM; max group RSS5.466GB. Current-best report SHA16bb9f13; exact b54 increment SHA dcc756f3. This is accuracy custody, not release/runtime qualification.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/FIRST_READOUT_PRICE_TAIL_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/FIRST_READOUT_PRICE_TAIL_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/check_first128_import_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/check_first128_import_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/check_first128_tiny_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/check_first128_tiny_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/fit_first128_local90_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/fit_first128_local90_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/qualify_passed_first128_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/qualify_passed_first128_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/READOUT_TAIL_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py](code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1/common_linux_v7.py)
- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6750](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6750)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6763](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6763)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337506: First-128 readout with fully charged residual-time repair](../../../submissions/337506/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-first-128-hidden-readout-367205943-mac32-4015029588562216-linux32-versus) · [Copy provenance](../../../PROVENANCE.json)

