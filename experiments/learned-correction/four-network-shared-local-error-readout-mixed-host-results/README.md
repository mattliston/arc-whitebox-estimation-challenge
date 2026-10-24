# Four-network shared local-error readout: mixed host results and no useful gain in edge stack

2026-10-21T02:37:30Z · experiment · rejected · measured

A fixed shared-state local-error readout had a small Mac gain but positive Linux held-out error. Combining it with the edge-width-eight parent failed to improve the parent consistently.

Recorded measurement:

    Root shared64 innovation SCI fb4bbca4 is MIXED, not banked: Linux32-0.04747415%/held+0.05561235%,SE0.113889,17/32lower; originalMac32-0.108148778%/held-0.050016109%,SE0.144409,19/32lower. Mac wrapper completed02:19:23Z/resumedqueue77740; no original source error. Initial five-percent coefficient-norm rejection remains preserved; explicit separate exploratory design used unchanged fitted coefficients after TRAIN-only physical audit. No standalone promotion or positive hosted claim.
    Root edge8+innovation SCI97491a17 completed originalMac32 at02:42:36Z: adjusted -0.1609447478%, held -0.0981452191%, SE0.132536,19/32lower; raw -0.0036784%,bill -0.15726625%. Almost no added gain versus edge8, worse held; no bank upgrade. Wrapper resumed77740. VM86584 full32 pending, first7 -0.629600%/SE0.221070/6of7lower is not the full result. PLAN/READY/IMPORT receipts copied back. Original Mac queue later had no active judges, so root launched qualified BDpadding SCI74335f92 original32 with unchanged isolatedparallel5 wrapper at02:54:27Z, session41400, output D/bd_padding_energy_v1/science32_bd_padding_energy_parallel5_v1. BEST422 PINNED for this new judge.
    Edge8+innovation SCI97491a17 is CLOSED as a stack improvement: originalMac32 -0.1609447%/held -0.0981452%, but VM full32 strict -0.0732775%/held +0.0824928%, SE0.126037,19/32lower,raw +0.0821422%,bill -0.155288%. VM receipt L/stack_innovation_scores_v1/scores_edge8_innovation_stack_v1__edge8_innovation_FULL32_VS_337422.json SHA62877865. Do not pursue fresh qualification or add standalone gains.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_oct19_kc_v1/candidate_astra_o21_edge8_innovation_v1_sci_v1.py](code/fable_oct19_kc_v1/candidate_astra_o21_edge8_innovation_v1_sci_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/price_current32_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/price_current32_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/check_tiny_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/check_tiny_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/tail_msg8_bd_padding_energy_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/tail_msg8_bd_padding_energy_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/candidate_msg8_bd_padding_energy_sci_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/candidate_msg8_bd_padding_energy_sci_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/candidate_msg8_bd_padding_energy_rel_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/candidate_msg8_bd_padding_energy_rel_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/selector_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/selector_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/build_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/build_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/prepare_linux_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/prepare_linux_v1.py)
- [code/astra_oct20_bill_v1/bd_padding_energy_v1/check_import_v1.py](code/astra_oct20_bill_v1/bd_padding_energy_v1/check_import_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6463](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6463)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6475](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6475)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6486](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6486)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [scores_edge8_innovation_stack_v1__edge8_innovation_FULL32_VS_337422.json](scores_edge8_innovation_stack_v1__edge8_innovation_FULL32_VS_337422.json) · result
- [TINY_v1.json](TINY_v1.json) · result
- [PRICE_CURRENT32_v1.json](PRICE_CURRENT32_v1.json) · result
- [CANDIDATES_v1.json](CANDIDATES_v1.json) · result
- [COPY_INVENTORY_v1.json](COPY_INVENTORY_v1.json) · result
- [IMPORT_v1.json](IMPORT_v1.json) · result
- [scores_edge8_innovation_stack_v1__edge8_innovation_FIRST7_VS_337422.json](scores_edge8_innovation_stack_v1__edge8_innovation_FIRST7_VS_337422.json) · result
- [CURRENT_SOURCE7_v1.json](CURRENT_SOURCE7_v1.json) · result
- [VS_BEST_GRADED_v1.json](VS_BEST_GRADED_v1.json) · result
- [PARALLEL32_COMPLETE_v1.json](PARALLEL32_COMPLETE_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-four-network-shared-local-error-readout-mixed-host-results) · [Copy provenance](../../../PROVENANCE.json)

