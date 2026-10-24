# Higher projector ranks, shared first affine and energy-balanced join selection fail full-32 tests

2026-10-21T01:59:48Z · experiment · rejected · measured

Increasing two message-projector ranks worsened adjusted score +0.035955% and +0.252044%. A shared first-affine fit and balanced join selector also produced no usable full-32 gain.

Recorded measurement:

    New completed closures versus exact422: W3rank6+0.035955%adjusted/held+0.069264%; GOP2rank3+0.252044%/held+0.274097%; shared actual777a first-affine+0.061546%/held+0.017855%. All32/all12 native guards good; none bankable. BalancedJSK originalMac32+0.0093759766%/held-0.0035547376%, SEall0.0546074,16/32lower; +0.000757112%bill. Wrapper finished01:47:59Z/restored oldqueue+fivejudges; BEST422 reference unpinned. Close these mechanisms, no duplicate VM rerun.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct20_bill_v1/balanced_jsk_current_v1/build_v1.py](code/astra_oct20_bill_v1/balanced_jsk_current_v1/build_v1.py)
- [code/astra_oct20_bill_v1/balanced_jsk_current_v1/candidate_msg8_balanced_jsk_sci_v1.py](code/astra_oct20_bill_v1/balanced_jsk_current_v1/candidate_msg8_balanced_jsk_sci_v1.py)
- [code/astra_oct20_bill_v1/balanced_jsk_current_v1/tail_msg8_balanced_jsk_v1.py](code/astra_oct20_bill_v1/balanced_jsk_current_v1/tail_msg8_balanced_jsk_v1.py)
- [code/astra_oct20_bill_v1/message_rank_frontier_v1/build_v1.py](code/astra_oct20_bill_v1/message_rank_frontier_v1/build_v1.py)
- [code/astra_oct20_bill_v1/message_rank_frontier_v1/candidate_msg8_gop2rank3_sci_v1.py](code/astra_oct20_bill_v1/message_rank_frontier_v1/candidate_msg8_gop2rank3_sci_v1.py)
- [code/astra_oct20_bill_v1/message_rank_frontier_v1/candidate_msg8_w3rank6_sci_v1.py](code/astra_oct20_bill_v1/message_rank_frontier_v1/candidate_msg8_w3rank6_sci_v1.py)
- [code/astra_oct20_bill_v1/message_rank_frontier_v1/tail_msg8_gop2rank3_v1.py](code/astra_oct20_bill_v1/message_rank_frontier_v1/tail_msg8_gop2rank3_v1.py)
- [code/astra_oct20_bill_v1/message_rank_frontier_v1/tail_msg8_w3rank6_v1.py](code/astra_oct20_bill_v1/message_rank_frontier_v1/tail_msg8_w3rank6_v1.py)
- [code/astra_oct20_vm_frontier_v1/export_bank8_shared_firstaffine_v1.py](code/astra_oct20_vm_frontier_v1/export_bank8_shared_firstaffine_v1.py)
- [code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py](code/astra_oct20_vm_frontier_v1/fit_frontier_v1.py)
- [code/astra_oct20_vm_frontier_v1/shared_firstaffine_math_v1.py](code/astra_oct20_vm_frontier_v1/shared_firstaffine_math_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6450](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6450)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-higher-projector-ranks-shared-first-affine-energy-balanced-join) · [Copy provenance](../../../PROVENANCE.json)

