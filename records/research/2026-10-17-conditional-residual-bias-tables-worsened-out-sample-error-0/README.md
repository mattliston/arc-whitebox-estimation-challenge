# Conditional residual-bias tables worsened out-of-sample error by 0.07–0.39% on 32 networks

2026-10-17T21:39:04Z · finding · informational · measured

Leave-one-rank-out correction tables based on activation ratio, scale, prediction and cumulants, as well as a global scale, found no exploitable conditional final bias. Most final error was on always-positive units and could not be removed by a simple output calibration.

Recorded measurement:

    32 ranks with 2 x 32M truths; tables +0.07..+0.39% out of sample; 32 activation bins explain 0.11% in sample; squared-error share 51% for a>=2, 36% for 0<=a<2, 3% for a<-1; correlation of layer-14 kink fraction and U: -0.10.

## Programs

- [residual_bias_v1.py](residual_bias_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4955](../../TEAM_LOG_2026-10-01_to_10-22.md#L4955)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-conditional-residual-bias-tables-worsened-out-sample-error-0) · [Copy provenance](../../../PROVENANCE.json)

