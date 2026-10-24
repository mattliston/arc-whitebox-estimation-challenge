# Six final read-cancellation operations in float64: small unconfirmed gain and an unsupported-shape exception

2026-10-17T21:51:31Z · experiment · rejected · measured

Promoting six final read-leaf cancellation operations to true float64 improved a seven-network screen. The gain shrank on 28 networks and the fragment raised RuntimeError on an unexpected tree shape, so it was not added to the submitted file.

Recorded measurement:

    7 ranks: raw -0.16% (SE .07, 6/7), bill +0.003%, adjusted -0.157%; 28 ranks: raw -0.060% (SE .054, 14/28), bill -0.02%, adjusted -0.079% (SE .054).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/scout_native`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_astra_final64_v1.py](code/fragment_astra_final64_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4957](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4957)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4959](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4959)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-six-final-read-cancellation-operations-float64-small-unconfirmed-gain) · [Copy provenance](../../../PROVENANCE.json)

