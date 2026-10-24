# Per-layer edge hidden width four: +5.393798% adjusted; width two already +10.315041% on seven networks

2026-10-21T01:59:48Z · experiment · rejected · measured

Aggressive layer-specific edge reconstruction lost much more raw accuracy than it saved in FLOPs. A separate one-hidden-128 student and head-only teacher fit were rejected by fidelity gates before native execution.

Recorded measurement:

    Per-layer edge hidden4 full32+5.393798%/held+5.633106% adjusted, raw+5.667289%,bill-0.258812%; all12guards good. Edge hidden2 first7+10.315041%,0/7lower; final group still finishing in latest receipt. Close this layer-specific tiny-edge reconstruction direction; no gain. One-hidden128 student rejected BEFORE native: validation state/head/message error27.34x/71.69x/56.07x priorrank16reference. Head-only teacher ridge also failed fidelity gate before native; no further generic teacher compression repetition.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/edge_hidden6_nonlinear_v1/fit_edge_hidden6_v1.py](code/astra_oct20_bill_v1/edge_hidden6_nonlinear_v1/fit_edge_hidden6_v1.py)
- [code/astra_oct20_bill_v1/edge_layer_hidden_v1/check_edge_v1.py](code/astra_oct20_bill_v1/edge_layer_hidden_v1/check_edge_v1.py)
- [code/astra_oct20_bill_v1/edge_layer_hidden_v1/export_v1.py](code/astra_oct20_bill_v1/edge_layer_hidden_v1/export_v1.py)
- [code/astra_oct20_bill_v1/edge_layer_hidden_v1/fit_v1.py](code/astra_oct20_bill_v1/edge_layer_hidden_v1/fit_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_current_msg8_messages_v2.py](code/astra_oct20_vm_frontier_v1/fit_current_msg8_messages_v2.py)
- [code/astra_oct20_vm_frontier_v1/fit_current_onehidden128_v2.py](code/astra_oct20_vm_frontier_v1/fit_current_onehidden128_v2.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6451](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6451)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-per-layer-edge-hidden-width-four-5-393798-adjusted) · [Copy provenance](../../../PROVENANCE.json)

