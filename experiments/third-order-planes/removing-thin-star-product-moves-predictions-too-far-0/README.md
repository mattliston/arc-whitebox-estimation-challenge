# Removing a thin star product moves predictions too far for a 0.33% cost saving

2026-10-02T09:19Z · experiment · rejected · measured

The parent factors the third-order remainder plane (star) through three thin products; after the second the plane is already a rank-32 factorization and the third could be skipped (-0.32% bill from layer 4). But the star is not low rank (a third of its norm beyond rank 32 at layers 9-14) and rank 32 only works because of the transport through the weights; the shorter form leaves 6-19% more error and moved the final prediction by 9.6 / 11.2e-6 rms (two rounding-level builds differ by 4-5e-6), an expected MSE cost of 0.6-0.9% for 0.33% of bill. Closed; the final error is this sensitive to a 1.2-1.5% change of the transported star plane.

Recorded measurement:

    bill -0.32% / -0.33% from layer 4 (67 Mflop per layer); star beyond rank 32: a third of norm at layers 9-14, nine tenths at layer 1; parent's form 2.2-3.5% relative error at layers 4-14 (best rank-32 2.0-3.1%); shorter form +6-19% at layers 4-14, +36-61% at layers 2-3; measured on 651 / 378: +0.36%, +2.16%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a31_star`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/probe_star.py](code/probe_star.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L225](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L225)
- [TEAM_LOG_2026-10-01_to_10-22.md#L228](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L228)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-removing-thin-star-product-moves-predictions-too-far-0) · [Copy provenance](../../../PROVENANCE.json)

