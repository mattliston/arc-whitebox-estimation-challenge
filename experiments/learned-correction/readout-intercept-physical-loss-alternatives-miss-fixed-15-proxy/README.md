# Readout intercept and physical-loss alternatives miss the fixed -.15% proxy improvement threshold

2026-10-21T04:23:14Z · experiment · rejected · measured

Both zero-operation readout alternatives improved the local proxy by less than the preregistered floor. They were closed without export, native execution or retuning.

Recorded measurement:

    Both fixed zero-op head continuations finished 9:18 p.m.: bias-intercept975 LOO incremental local proxy -0.0236221% (70/90), physical-loss960 -0.00391234% (57/90), both miss frozen -0.15% floor. CLOSED without export/native or retuning. FITTED05dbc43d/guardf981b929 clean. Actual four-direction queue reached152/240 endpoint predictions at9:21 p.m.; unchanged guards and fixed downstream fitter/actual30 continuation remain active.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/head_objectives90_v1/check_head_objectives_import_v1.py](code/astra_oct20_bill_v1/head_objectives90_v1/check_head_objectives_import_v1.py)
- [code/astra_oct20_bill_v1/head_objectives90_v1/fit_head_objectives_v1.py](code/astra_oct20_bill_v1/head_objectives90_v1/fit_head_objectives_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6583](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6583)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-readout-intercept-physical-loss-alternatives-miss-fixed-15-proxy) · [Copy provenance](../../../PROVENANCE.json)

