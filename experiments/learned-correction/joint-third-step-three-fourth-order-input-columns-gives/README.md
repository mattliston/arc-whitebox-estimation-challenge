# Joint third step of three fourth-order input columns gives only 0.20% predicted gain, below noise

2026-10-02T20:13Z · experiment · rejected · measured

A joint direction over 384 weights of the fitted diagonal and two new fourth-order inputs is line-searched on 16 ranks. Their aligned gradients ask for the same largely exhausted direction, so the weights are neither evaluated on the 25-rank judge nor exported.

Recorded measurement:

    step0.1 -0.226% (se0.231),9/16; step0.25 +1.267%; step0.5 +7.46%; parabola minimum step0.062,-0.20%,curvature40; earlier single-column curvatures28 and9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cert/head/solve.py](code/cert/head/solve.py)
- [code/otr_common.py](code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L469](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L469)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-joint-third-step-three-fourth-order-input-columns-gives) · [Copy provenance](../../../PROVENANCE.json)

