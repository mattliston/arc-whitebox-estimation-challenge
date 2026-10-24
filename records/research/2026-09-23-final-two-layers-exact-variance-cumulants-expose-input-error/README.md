# Final two layers: exact variance and cumulants expose input error rather than a deficient final map

2026-09-23T04:02Z · finding · informational · measured

Replacing last-layer inputs and disabling the learned delta nearly reached the sampling noise floor. Most last-layer variance error came from the preceding mean-orthogonal off-diagonal covariance.

Recorded measurement:

    2026-09-22 21:02 physicist: LAST-TWO-LAYER lane (fable_physicist_v1/late/RESULT_LATE.md). Born@L15 (given exact m14, .232): exact v/k3/k4@15 -> .077, + learned delta off -> .013/.009 (noise floor): final map exact, born error = per-unit input error (v 38-48%, k3 10%, k4 5-8%). v15 error = pushed OFF-DIAGONAL L14 post cov error, mean-orthogonal, from the pair-map output (learned corrections minor). Born@L14: v14 -.07, k4 -.02/-.03, Coff@13 -.08/-.12. Local candidates all dead: per-unit k4 model (0.5 rad v^2 + 0.9 k3^2/v) +15..+61% closed loop (kink units' truth k4 ~ rad v^2; fit halves it); weighted structures -3-5%; class feeds +0 R2; no-frame card +1.7/-4.0% (k3_15 unchanged); k3_15 carriers correctly scaled (coef .97-1.06).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L38](../../TEAM_LOG_2026-09-22_to_09-26.md#L38)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_LATE.md](RESULT_LATE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-final-two-layers-exact-variance-cumulants-expose-input-error) · [Copy provenance](../../../PROVENANCE.json)

