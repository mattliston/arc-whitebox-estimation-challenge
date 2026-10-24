# Shared third- and fourth-order column weights cannot jointly reproduce the needed covariance correction

2026-09-23T04:19Z · experiment · rejected · measured

A shared per-column fourth-order weight fit was inconsistent across the (3,1) and (2,2) planes. Even free oracle weighting captured too little covariance correction to pay for a new plane read.

Recorded measurement:

    2026-09-22 21:19 physicist: cheap consistent 3rd+4th-order plane supply: NONE (fourth/RESULT_FOURTH.md). Per-column lam4 (same bank columns, free transport), FREE oracle fit r35 L12: K31 51% / K22 44% alone, shared lam4 51%/19% (inconsistent); net covariance change (r=0 read, resp x live): K21 read .037 of net .042; free per-column lam4 27%, K21 re-weight 37%, both 38%. One (3,1) read over L2-14 = 54 dense units young / 108 all (bill ~188). (2,1,1) not read by the pair-map covariance; P1/gain-skew dead.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/fourth`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/col_lambda4_fit.py](code/col_lambda4_fit.py)
- [code/net_read_fit.py](code/net_read_fit.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L47](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L47)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_FOURTH.md](RESULT_FOURTH.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-shared-third-fourth-order-column-weights-cannot-jointly-reproduce) · [Copy provenance](../../../PROVENANCE.json)

