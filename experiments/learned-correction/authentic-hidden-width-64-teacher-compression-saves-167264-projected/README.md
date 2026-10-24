# Authentic hidden-width-64 teacher compression saves .167264% projected FLOPs but fails all five folds

2026-10-21T03:43:19Z · experiment · rejected · measured

Complete-path pricing was favorable, but teacher cross-validation produced state/head/message RMS errors 8.5866% / 15.3605% / 6.4339%. No width or ridge sweep followed.

Recorded measurement:

    Bounded authentic hidden2-neuron64 screen (siblinghidden2_prune90_v1) pricedcompletepath−396,165,120FLOPs/network /−0.167264% projectedmeanbill vs337422, then fixed5foldteacherCV failedallfolds: stateRMS8.5866%, head15.3605%, message6.4339%. GuardPASS; no truthlabels/export/native, no width/ridge sweep. Closedcostroute, notadjustedbank. Parent's independent local90 native results are separate. See lane STATUS_CURRENT.md and copiedreceiptjsons.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/hidden2_prune90_v1/fit_hidden2_v1.py](code/astra_oct20_bill_v1/hidden2_prune90_v1/fit_hidden2_v1.py)
- [code/astra_oct20_bill_v1/hidden2_prune90_v1/price_hidden2_v1.py](code/astra_oct20_bill_v1/hidden2_prune90_v1/price_hidden2_v1.py)
- [code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py](code/astra_oct20_vm_frontier_v1/check_frontier_tiny_v6.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py](code/astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6515](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6515)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-authentic-hidden-width-64-teacher-compression-saves-167264-projected) · [Copy provenance](../../../PROVENANCE.json)

