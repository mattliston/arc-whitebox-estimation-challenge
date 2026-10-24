# Explicit covariance symmetrization: adjusted gain 0.516381% stays below expansion gate

2026-09-27 · experiment · rejected · measured

The covariance was explicitly symmetrized through the paid API while retaining the asymmetric star plane and thin-star routing. The combined score improved but was slightly worse than thin-star routing alone.

Recorded measurement:

    TRAIN2 adjusted-0.516381%,raw+0.083448%,2 wins;increment versus thin routing+0.0796%;fixed expansion gate1%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/explicit_covariance_symmetry_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [RESULTS.md#L1652](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1652)
- [hill_climb_reset_20260928_v1.md#L23](hill_climb_reset_20260928_v1.md#L23)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [hill_climb_reset_20260928_v1.md](hill_climb_reset_20260928_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-explicit-covariance-symmetrization-adjusted-gain-0-516381-stays-below) · [Copy provenance](../../../PROVENANCE.json)

