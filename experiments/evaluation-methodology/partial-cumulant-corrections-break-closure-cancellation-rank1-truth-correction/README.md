# Partial cumulant corrections break closure cancellation: rank1 truth correction makes final MSE 6x worse

2026-09-16 · experiment · rejected · measured

Low-rank or scalar corrections to the third-cumulant plane worsened closed-loop accuracy despite better supplied components. A learned per-unit fourth-cumulant correction also regressed, supporting joint adaptation of birth and response coefficients.

Recorded measurement:

    Rank1 truth component final1.15e-7(6x worse);rank8 5.5e-8;truth-slope scaling7.3e-8;learned k4 leave-one-net-out R2.39,error.24 to.17,final1.92e-8(+4%),all-layer1.90e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/oracle_struct.py](code/oracle_struct.py)

## Evidence

- [RECEIPTS.md#L70](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L70)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-16-partial-cumulant-corrections-break-closure-cancellation-rank1-truth-correction) · [Copy provenance](../../../PROVENANCE.json)

