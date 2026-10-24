# First-hidden width 128 to 64 compression fails every teacher fold with 15.7-30.3% RMS errors

2026-10-21T07:56:45Z · experiment · rejected · measured

The current first-skip-compatible compression passed pricing and synthetic proofs but failed all five fixed teacher folds. No full-data refit, export, native evaluation or retry followed.

Recorded measurement:

    First64 fixed teacher fit scientifically CLOSED, all5folds failed: aggregate secondhidden15.7298%, state17.0842%, physical completehead30.2994%, message15.8138%, firstskip18.0333% RMS error. K/CLOSURE_v1.json dfdf3bae8f7256813ac25fd2ec1a460b36f8a616f541144320468434a2d0363e; FITTEDc819807f; original guard562550a9 retired clean07:51:21Z, all child events0/swap0, RSS324.6MB/cgrouppeak2.675GB. No full90 refit/model/export/native/retry; price savings cannot be claimed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py](code/astra_oct20_bill_v1/breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py)
- [code/astra_oct20_bill_v1/first_hidden64_current_v1/fit_first64_current_v1.py](code/astra_oct20_bill_v1/first_hidden64_current_v1/fit_first64_current_v1.py)
- [code/astra_oct20_bill_v1/first_hidden64_current_v1/price_first64_current_v1.py](code/astra_oct20_bill_v1/first_hidden64_current_v1/price_first64_current_v1.py)
- [code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_science_v1.py](code/astra_oct20_bill_v1/runtime_smoff_npxoff_first128_v1/candidate_first128_smoff_npxoff_science_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6934](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6934)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-first-hidden-width-128-64-compression-fails-every-teacher) · [Copy provenance](../../../PROVENANCE.json)

