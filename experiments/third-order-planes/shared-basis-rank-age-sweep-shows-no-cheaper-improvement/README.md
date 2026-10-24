# Shared-basis rank and age sweep shows no cheaper improvement over rank384 at age5

2026-09-16 · experiment · rejected · measured

Native tests on networks35/100 varied old-tier rank and absorption age. Rank512 increased cost without consistent raw gain; rank256 and age4 damaged accuracy.

Recorded measurement:

    rank384/age5 util.201/.197,MSE1.837/1.806e-8;rank512/age5 .228/.223,1.884/1.775e-8;age4 .196/.191,2.07/2.07e-8;age6 .205/.202,1.873/1.775e-8;rank256 .175/.172,2.45/2.45e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/sweep_tier.py](code/sweep_tier.py)
- [code/variantI_rule95.py](code/variantI_rule95.py)

## Evidence

- [RECEIPTS.md#L65](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L65)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-16-shared-basis-rank-age-sweep-shows-no-cheaper-improvement) · [Copy provenance](../../../PROVENANCE.json)

