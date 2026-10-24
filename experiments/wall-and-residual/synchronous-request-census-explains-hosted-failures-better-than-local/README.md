# Synchronous request census explains hosted failures better than local compute time

2026-09-13 · experiment · rejected · measured

Each flopscope operation incurred one synchronous request-reply round trip. The direct census corrected earlier approximate call counts and showed that the failed estimator issued far more requests despite a low metered bill.

Recorded measurement:

    Passing #32992637,255 requests; passing #33039044,135; failed #33092460,262; failed #33084087,510; fitted platform wall1.1-1.4ms x requests +~2.0 x local compute.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/callsite_census.py](code/callsite_census.py)

## Evidence

- [RECEIPTS.md#L37](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L37)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-synchronous-request-census-explains-hosted-failures-better-than-local) · [Copy provenance](../../../PROVENANCE.json)

