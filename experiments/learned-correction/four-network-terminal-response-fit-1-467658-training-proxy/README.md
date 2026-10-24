# Four-network terminal-response fit: -1.467658% training proxy but +0.873410% leave-one-out

2026-10-21T02:37:30Z · experiment · rejected · measured

A state-response fit targeting terminal error overfit four training networks. It was closed before export despite source-exact full-state captures and a corrected synthetic derivative test.

Recorded measurement:

    Root final-error TRAIN4 study is CLOSED before native: D/propagated_state_response_v1. Full1024 state64/hidden128/gate/sigma captures TRAIN35/150/200/250, original777a source, all4 entire predictions BYTE-EQUAL original777a, both original guards PASS, completed02:20:26Z/PID74504/766s/peak5.466GB. Fixed shared64 mean-propagation response targets TERMINAL error (not local innovation), ridge0.1/damping0.5. Fitterv1 synthetic finite difference crossed a clipping boundary; immutablev2 changes ONLY synthetic biases to stay differentiable, same tolerances/algorithm; diagnosis preserved. Fitterv2 SHA2a0091f351f1bcb04de3cf6263a89cc43595e279a201aefe2bc4bd99e6635978, guard79002 PASS02:21:03Z/10.4s/275MB. FITTEDc16482a03dc91ef721846252939b478d4a4e7ba7834c58e7d97cacc22c530af1: trainproxy-1.467658%, LOO+0.873410%, only1/4lower. No candidate export, no tuning or native launch. Full-state captures remain useful for independent consumer/physics tests.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/propagated_state_response_v1/client_capture_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/client_capture_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/full_state_hook_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/full_state_hook_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/run_group_capture_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/run_group_capture_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/fit_terminal_response_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/fit_terminal_response_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/run_capture_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/run_capture_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/run_fit_pipeline_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/run_fit_pipeline_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/prepare_capture_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/prepare_capture_v1.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/fit_terminal_response_v2.py](code/astra_oct20_bill_v1/propagated_state_response_v1/fit_terminal_response_v2.py)
- [code/astra_oct20_bill_v1/propagated_state_response_v1/fragment_mark_complete_line_v1.py](code/astra_oct20_bill_v1/propagated_state_response_v1/fragment_mark_complete_line_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py)
- [code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py](code/astra_oct20_bill_v1/cloud_linux_v1/guard_entry_linux_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/client_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/common_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/guard_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_group_linux_v7.py)
- [code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py](code/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/run_science_linux_v7.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6465](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6465)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SELFTEST_V1_DIAGNOSIS_v1.json](SELFTEST_V1_DIAGNOSIS_v1.json) · result
- [PREREG_v1.json](PREREG_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-four-network-terminal-response-fit-1-467658-training-proxy) · [Copy provenance](../../../PROVENANCE.json)

