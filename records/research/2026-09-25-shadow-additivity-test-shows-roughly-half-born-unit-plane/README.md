# Shadow additivity test shows roughly half of the born-unit plane correction is already learned

2026-09-25T00:31Z · finding · informational · measured

Adding the learned-off correction effect to the learned-on predictor yielded only partial improvement. This explained why a refit was needed and blocked the proposed covariance-only experiment.

Recorded measurement:

    2026-09-24 17:31 covhunter: SHADOW (additivity) TEST (RESULT sec 23; VALID16, B2 ep1 ON 1.74611e-8 @3 thr): S = m(ON,0) + [m(OFF,X) - m(OFF,0)]. Controls reproduce (OFF +24.1%; j1 OFF X.5 -6.33% 15/16, X.3 -3.45% 13/16). Shadow beta=1: X.5 -0.80% (10/16, se 1.0%), X.3 -1.34% (11/16, se 0.8%); fitted pooled beta* .54/.62 -> -2.89%/-2.18% (in-sample). = PARTIALLY ADDITIVE: ~half of j1's learned-off effect is already inside the learned parts; additive half ~2-3%. L2 gate (<= -2% at beta=1) FAILS as pre-registered. Next: L3 read-aware frame (rank 320, VALID16 no refit).
    - 2026-09-24 17:55 lead: COVHUNTER SHADOW TEST (§23): j1 is PARTIALLY additive. m(ON,0) + [m(OFF,X) - m(OFF,0)] on VALID16 vs B2 ep1: X=.5 -0.80% (10/16), X=.3 -1.34% (11/16); a fitted pooled scale beta ~.54-.62 gives -2.2..-2.9%. About half of j1's learned-OFF effect already sits inside the learned parts; the other half (~2-3%) adds. This matches the j05 refit's -1.94%. L2 (covariance-read-only fix) FAILS its pre-registered gate and is not run. Covhunter moves to L3: read-aware old-tier frame (Omega = top-rank units by min(gate, 1-gate)), rank 320, bit-exact certificate at 384 first.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L515](../../TEAM_LOG_2026-09-22_to_09-26.md#L515)
- [TEAM_LOG_2026-09-22_to_09-26.md#L516](../../TEAM_LOG_2026-09-22_to_09-26.md#L516)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-shadow-additivity-test-shows-roughly-half-born-unit-plane) · [Copy provenance](../../../PROVENANCE.json)

