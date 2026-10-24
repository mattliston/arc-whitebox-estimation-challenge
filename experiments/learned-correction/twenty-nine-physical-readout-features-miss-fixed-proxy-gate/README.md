# Twenty-nine physical readout features miss the fixed proxy gate; first-128 reconstruction passes

2026-10-21T06:09:02Z · experiment · rejected · measured

The physical-feature extension gave -.138076517% leave-one-out improvement, below the -.15% threshold. Certified first-128 reconstruction instead passed -.311401086% with 90/90 networks lower and all five blocks negative.

Recorded measurement:

    [accuracy_research] 2026-10-21T06:09:02Z Fresh takeover harvest: physical29 fixed LOO -0.138076517% missed -0.15% floor; closed/no retune/no native (fit5203d267/guard48eae9eb). First128 reconstruction90 + frozen LOO passed -0.311401086%,90/90 lower, all5 blocks negative, physical8.74294%/max22.22653% (fitd1bca887/guard320ec436). Unchanged pass-only source/import/tiny qualifier now originalguard172899 at06:08:33Z; no native until root review. On-policy sixparents build complete d6eaeb23; no captures yet. CurrentBEST337466; no adjusted gains claimed.
    First128 fixed90proxy passed-.311401086%,90/90lower/all5blocksnegative, physical.0874294/maxlayer.2222653. All90inversegates passed. Directb54SCI4da68821/RELe491c4ae qualified all15cold/warm/NumPy-free, added3,932,160FLOPs. Root independently reproduced coefficients and AST/prefix/pins; review5736dfff authorizes ONE originalVM32 viaops and rootMac32afterDAGquiet. Noadjustedgainclaimyet. Physical29 CLOSED-.138076517% missesfrozen-.15%floor; no retune. Onpolicy sixparentsbuilt; tiny/import/capture-planpreflight andsecond-stagefitterpreparing, NOcapturesauthorizedyet.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/fit_first128_local90_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/fit_first128_local90_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/freeze_first128_protocol_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/freeze_first128_protocol_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/reconstruct_first128_train90_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/reconstruct_first128_train90_v1.py)
- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/run_first128_pipeline_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/run_first128_pipeline_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py](code/astra_oct20_bill_v1/hidden_complement90_v1/fit_hidden_complement_v1.py)
- [code/astra_oct20_bill_v1/physical_input29_v1/fit_physical29_v1.py](code/astra_oct20_bill_v1/physical_input29_v1/fit_physical29_v1.py)
- [code/astra_oct20_bill_v1/physical_input29_v1/wait_and_fit_physical29_v1.py](code/astra_oct20_bill_v1/physical_input29_v1/wait_and_fit_physical29_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6706](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6706)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6728](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6728)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-twenty-nine-physical-readout-features-miss-fixed-proxy-gate) · [Copy provenance](../../../PROVENANCE.json)

