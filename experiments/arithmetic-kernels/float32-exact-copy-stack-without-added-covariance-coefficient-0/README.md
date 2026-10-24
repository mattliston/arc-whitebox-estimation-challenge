# Float32 and exact-copy stack without added covariance coefficient: -0.183168835% versus #337393

2026-10-20T20:47:23Z · experiment · rejected · measured

Removing the added covariance coefficient improved the full-32 result beyond the corresponding submitted stack. The held-out result was -.168640518%, with runtime qualification still outstanding.

Recorded measurement:

    NEW completed micro_copyall_astra f2a08328 (NO added KC/edge) all32-.183168835% SE.111962,21/32 lower; held25-.168640518%; first7-.235055682%. Saved VS_BEST_337393_v1.json. This is better than firedmicroKC's local-.147444%, but NOT release-qualified. micro_k31age4_astra all32-.102122889%,held-.096744110%, no promotion. Other new full32 stacks still pending: noKCmicroedge; GSK+copyall and GSK+K31; noKC K31edge; auxKC K31edge. All new user reports via K/compare_best_v1.py.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/compare_best_v1.py](code/compare_best_v1.py)
- [code/compare_relative_v1.py](code/compare_relative_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6288](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6288)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-float32-exact-copy-stack-without-added-covariance-coefficient-0) · [Copy provenance](../../../PROVENANCE.json)

