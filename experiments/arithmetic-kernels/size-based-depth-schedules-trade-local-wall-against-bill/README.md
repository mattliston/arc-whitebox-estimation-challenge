# Size-based depth schedules trade local wall against bill while retaining final MSE near 1.84e-8

2026-09-13 · experiment · rejected · measured

Fixed naive-product FLOP thresholds selected depths without changing estimator coefficients. More conservative schedules remained local alternatives.

Recorded measurement:

    95s rule:r35 MSE1.8374e-8,util0.2010,native16.9s;100s rule1.8363e-8,0.1996,17.4s;120/110/85/80 rules util0.1977/0.1978/0.2091/0.2132.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_variantI.py](code/build_variantI.py)
- [code/variantH_splitframes.py](code/variantH_splitframes.py)

## Evidence

- [RECEIPTS.md#L53](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L53)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-size-based-depth-schedules-trade-local-wall-against-bill) · [Copy provenance](../../../PROVENANCE.json)

