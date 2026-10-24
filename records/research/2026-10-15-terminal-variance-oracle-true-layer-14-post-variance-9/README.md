# Terminal variance oracle: true layer-14 post-variance -9.2% final MSE; diagonal part ~0, covariance/transport -10.3%

2026-10-15T01:26:22Z · finding · informational · measured

A capture of the deployed file's terminal state on training networks showed that replacing the layer-14 post-variance by its true value (holding mean, third and fourth cumulants and head fixed) removes about 9% of the final MSE, but decomposing the pre-variance error showed the per-unit diagonal part is worth nothing: the whole oracle sits in the covariance and transport part, whose centered projection is 99.7 to 99.9% of it. A scalar final post-variance fix therefore has no direct headroom.

Recorded measurement:

    TRAIN16 single network: terminal fixed-mu/k3/k4/head variance oracle MSE -9.191% (independent half cross -9.433%); local L12/13/14 -13.39/-8.41/-9.39% (not propagated); three networks 16/259/277: diagonal oracle geometric MSE +0.00529%, remaining covariance/transport oracle -10.3375%, total -10.5978%, centered projection 99.70-99.94%; capture 71.68 s / 242.058 GFLOP per call

## Programs

- [decompose_terminal_variance_v1.py](decompose_terminal_variance_v1.py)
- [analyze_variance_oracle_v1.py](analyze_variance_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3237](../../TEAM_LOG_2026-10-01_to_10-22.md#L3237)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3238](../../TEAM_LOG_2026-10-01_to_10-22.md#L3238)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3241](../../TEAM_LOG_2026-10-01_to_10-22.md#L3241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3247](../../TEAM_LOG_2026-10-01_to_10-22.md#L3247)

<details>
<summary>All result files and records (2)</summary>

- [CURRENT_VARIANCE_ORACLE_TRAIN16_v1.json](CURRENT_VARIANCE_ORACLE_TRAIN16_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Linear (94-feature ridge) and nonlinear (94-16-1) fits of the terminal variance correction fail held-out networks](../../../experiments/learned-correction/linear-94-feature-ridge-nonlinear-94-16-1-fits/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-terminal-variance-oracle-true-layer-14-post-variance-9) · [Copy provenance](../../../PROVENANCE.json)

