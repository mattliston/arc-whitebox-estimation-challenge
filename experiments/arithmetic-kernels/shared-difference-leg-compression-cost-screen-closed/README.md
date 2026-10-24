# Shared difference-leg compression: cost screen closed

2026-09-05T17:43Z · experiment · rejected · measured

Existing measured Winograd cost polynomials, current rank35/100 mask/birth shapes and the public float32 eigendecomposition charge were used. Even optimistically reusing each old plane for free, while charging basis acquisition and new repeated-plane restoration, projected total savings are only 4.000% / 4.357% of those networks' current measured first-context FLOPs. It fails the >=15% cost gate before any accuracy work.

Recorded measurement:

    Existing measured Winograd cost polynomials, current rank35/100 mask/birth shapes and the public float32 eigendecomposition charge were used. Even optimistically reusing each old plane for free, while charging basis acquisition and new repeated-plane restoration, projected total savings are only 4.000% / 4.357% of those networks' current measured first-context FLOPs. Raw accuracy is unknown and could deteriorate. It fails the >=15% cost gate before any accuracy work. No SVD execution, rank/cut sweep, coadaptation, port, panel or submission is justified for this proposal. Evidence: difference_leg_cuts_cost_v1.json and the explicit operation model in price_difference_leg_cuts_v1.py. These are analytical projections, not a newly measured utilization or benchmark speedup.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/price_difference_leg_cuts_v1.py](code/price_difference_leg_cuts_v1.py)

## Evidence

- [DIFFERENCE_LEG_CUTS_V1_CLOSED.md#L1](DIFFERENCE_LEG_CUTS_V1_CLOSED.md#L1)
- [DIFFERENCE_LEG_CUTS_V1_CLOSED.md#L13](DIFFERENCE_LEG_CUTS_V1_CLOSED.md#L13)

<details>
<summary>All result files and records (2)</summary>

- [DIFFERENCE_LEG_CUTS_V1_CLOSED.md](DIFFERENCE_LEG_CUTS_V1_CLOSED.md) · record
- [difference_leg_cuts_cost_v1.json](difference_leg_cuts_cost_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-shared-difference-leg-compression-cost-screen-closed) · [Copy provenance](../../../PROVENANCE.json)

