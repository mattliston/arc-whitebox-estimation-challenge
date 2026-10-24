# Plain matrix multiplication for thin star products removes 4,350 calls but raises charged work by 0.48%

2026-10-02T00:10Z · experiment · adopted · measured

The 84 thin low-rank-star products per network (depth 3 since S32) took about 4,200 kernel calls, a tenth of all requests, for 1.7% of the bill. Running them as plain matmuls (tsr_off) removes those requests at a bill cost; on 20 fresh networks against the parent the final-layer MSE fell 0.238% and the bill rose 0.481%, a net score cost of about a quarter of a percent, accepted in S36 for the residual-time room it buys.

Recorded measurement:

    kernel calls -4,350, bill +0.48%; 20 fresh ranks native vs RRPI: final-layer MSE -0.238% [95% -0.39,-0.09] (13 of 20 lower, per-network mean -0.238%, se 0.080%), bill +0.481%, adjusted +0.245% [+0.09,+0.39], worst network +0.80%; fastC (un-fuse + tsr_off) against S35: bill +0.38%, requests -6.4%, residual instructions -7.0%, projected wall 73.0 / 84.0 s (S35 82.5 / 94.9), residual 0.283 / 0.315 s (S35 0.304 / 0.339)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/variants/rrpi_fastC`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L25](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L25)
- [TEAM_LOG_2026-10-01_to_10-22.md#L31](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L31)
- [TEAM_LOG_2026-10-01_to_10-22.md#L38](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L38)
- [TEAM_LOG_2026-10-01_to_10-22.md#L63](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L63)
- [TEAM_LOG_2026-10-01_to_10-22.md#L64](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L64)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333429: float32 arithmetic, half-zero strips, refreshed setup keys and separate tree levels](../../../submissions/333429/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-plain-matrix-multiplication-thin-star-products-removes-4-350) · [Copy provenance](../../../PROVENANCE.json)

