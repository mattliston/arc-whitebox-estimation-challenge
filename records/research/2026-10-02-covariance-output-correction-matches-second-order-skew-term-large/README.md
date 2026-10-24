# Covariance-output correction matches a second-order skew term on large-response rows

2026-10-02T08:28Z · finding · informational · derived

The exact gradient table on 16 newer networks singled out k21n (the knob that scales the third-order plane of rows outside the kink, -0.028 in the parent) with t +5.1 wanting to be more negative; a forward scan on 12 held-out ranks gave -0.19% at -0.02 and +1.75% at -0.08 (parabola optimum -0.019), so the gradient's first-order size was three times too large. Physically the constant is the mean of the K21 covariance term's skew-corrected kink density -(t3/6) He3(alpha): same rows, same sign, -2 to -3% at layers 8-15, varying 5-8% from row to row and growing with depth as t3^2; the per-unit skewness is 0.94 correlated with alpha (the collective mode).

Recorded measurement:

    gradient +0.482 (se 0.095, t +5.1), first order -0.57% per -0.02; scan: -0.02 -0.19% (se 0.16, 9 of 12), -0.04 +0.06% (se 0.32), -0.08 +1.75% (se 0.67); term mean on rows |alpha| >= 1.5: -0.3% (layer 2), -1.1% (5), -2.0..-2.5% (8), -2.4% (11), -3.1..-3.4% (15), row spread 5-8%; corr(t3, alpha) 0.94 from layer 8

## Programs

- [t17_k21n_reading.py](t17_k21n_reading.py)
- [../../../experiments/moment-propagation/second-order-skew-correction-worsens-error-learned-correction-but/code/opus_reexam_20261001_v1/physicist/pcard_light.py](../../../experiments/moment-propagation/second-order-skew-correction-worsens-error-learned-correction-but/code/opus_reexam_20261001_v1/physicist/pcard_light.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L126](../../TEAM_LOG_2026-10-01_to_10-22.md#L126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L140](../../TEAM_LOG_2026-10-01_to_10-22.md#L140)
- [TEAM_LOG_2026-10-01_to_10-22.md#L197](../../TEAM_LOG_2026-10-01_to_10-22.md#L197)
- [RESULT.md#L1284](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md#L1284)

<details>
<summary>All result files and records (3)</summary>

- [scan_k21n_heldout.jsonl](scan_k21n_heldout.jsonl) · result
- [RESULT.md](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-covariance-output-correction-matches-second-order-skew-term-large) · [Copy provenance](../../../PROVENANCE.json)

