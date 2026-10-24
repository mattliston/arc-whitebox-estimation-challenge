# Public runtime comparison: submission #336342 had eleven networks above .38 s residual

2026-10-16T21:58:19Z · finding · informational · measured

The lead compared the request profiles and platform residuals of the valid best #336342, the board best #336412 (the estimator submitted as #336412 (S2tr)) and the clean S5: #336342 carried more requests than S2tr and skated under the .4 s cap on a quiet platform with 11 networks above .38 s; S2tr lost one hidden network to a .422 spike; V-A-noB lost its public MLP3 to a .430 spike. Against #336342 on the 50 public networks the S2tr physics (Full384 + GK4 0.6 + table) is MSE -1.644% for bill +1.389%. The zero-failure class stays at median <= .32 s.

Recorded measurement:

    #336342 (2b108): first 48,425 / warm 46,170 requests, residual .336, P(loss) 19%; median .367 / p95 .386 / max .397 (11 nets > .38). S2tr: 47,336 / 45,646, .330, 13.4%; median .356 / max .378 completed, lost hidden net .422. S5: 41,846 / 40,553, .292, 1.9%. #336570: median .355 / max .387, public MLP3 .430. Public 50-net effects vs 336342: b709 table81/R7 MSE -0.02 bill -0.09 score -0.11; S2tr MSE -1.644 bill +1.389 score -0.285.

## Programs

- [../../../submissions/336570/public_pair_v1.py](../../../submissions/336570/public_pair_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4508](../../TEAM_LOG_2026-10-01_to_10-22.md#L4508)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4516](../../TEAM_LOG_2026-10-01_to_10-22.md#L4516)

<details>
<summary>All result files and records (3)</summary>

- [eval_336412.json](../../../submissions/336412/eval_336412.json) · result
- [eval_336570.json](../../../submissions/336570/eval_336570.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-public-runtime-comparison-submission-336342-had-eleven-networks-above) · [Copy provenance](../../../PROVENANCE.json)

