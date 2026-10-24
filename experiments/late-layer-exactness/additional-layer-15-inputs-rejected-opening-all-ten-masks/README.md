# Additional layer-15 inputs rejected: opening all ten masks adds 7.5% MSE

2026-10-10T23:24Z · experiment · rejected · measured

Computing the ten masked layer-15 inputs (row stats of C_pre15, plane norms, post-correlation stats) and letting them through the head loses with the existing weights; new layer-15 weights fitted on 24 networks and judged on 32 held-out seeds beat noise on no input; a gradient-boosting ceiling finds the layer-15 residual unpredictable.

Recorded measurement:

    13-15 + 26 +1.60% (7 of 56 lower); all ten +7.5%; best single input 14 -0.017% (se 0.011); groups +0.14..+0.36% held out; ceiling 0.000% / -0.012% (se 0.034); native U15a +0.27% bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/physicist/t83_l15_unmask.py](code/physicist/t83_l15_unmask.py)
- [code/physicist/t84_l15_ceiling.py](code/physicist/t84_l15_ceiling.py)
- [code/engineer/a55_u15/fragment_u15_v1.py](code/engineer/a55_u15/fragment_u15_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L702](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L702)
- [TEAM_LOG_2026-10-01_to_10-22.md#L716](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L716)
- [TEAM_LOG_2026-10-01_to_10-22.md#L717](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L717)
- [TEAM_LOG_2026-10-01_to_10-22.md#L723](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L723)
- [TEAM_LOG_2026-10-01_to_10-22.md#L724](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L724)
- [TEAM_LOG_2026-10-01_to_10-22.md#L733](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L733)
- [TEAM_LOG_2026-10-01_to_10-22.md#L742](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L742)

<details>
<summary>All result files and records (3)</summary>

- [t83_out.txt](t83_out.txt) · result
- [t84_out.txt](t84_out.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [All-layer-loss refit rejected: late-layer residual prediction removes less than 0.1% of error power](../../learned-correction/all-layer-loss-refit-rejected-late-layer-residual-prediction/README.md)
- ← enables: [Layer-15 weight refit rejected: +1.69% held-out MSE despite -1.77% in sample](../../learned-correction/layer-15-weight-refit-rejected-1-69-held-out/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-additional-layer-15-inputs-rejected-opening-all-ten-masks) · [Copy provenance](../../../PROVENANCE.json)

