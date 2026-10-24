# Exact-shape frame binding improves native time but repeated rebinding adds 3.6% bill

2026-09-13 · experiment · rejected · measured

Exact-shape frames preserved output bytes and reduced native wall. A persistent cache removed repeated binding charges; free empty-frame allocation produced page-fault churn without wall improvement.

Recorded measurement:

    Per-prediction reshape variant14.7s,+3.6% bill; empty-frame variant7.0GB RSS,no wall gain; persistent cache14.1s,util0.2020 first/0.1972 next/0.1942 repeat,peak5.9GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_wall_repair_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/variantC_contig.py](code/variantC_contig.py)
- [code/variantA2_d4push_sq4.py](code/variantA2_d4push_sq4.py)
- [code/profile_native.py](code/profile_native.py)

## Evidence

- [RECEIPTS.md#L30](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L30)

<details>
<summary>All result files and records (1)</summary>

- [RECEIPTS.md](../../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-13-exact-shape-frame-binding-improves-native-time-but-repeated) · [Copy provenance](../../../PROVENANCE.json)

