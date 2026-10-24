# True layer-14 pre-activation variance on kink units removes 9.29% of final MSE; no cheap per-unit predictor

2026-10-16T16:42:44Z · finding · informational · measured

On 8 store ranks with 2 x 32M per-unit truths, substituting the true layer-14 pre-activation variance on the kink units only (about 220 units with |a14| < 1) lowers the noise-free final cross-half MSE of the estimator submitted as #336412 (S2tr) by 9.29%; on all units the gain is smaller (-6.6%) because the non-kink units are co-adapted. The kink units relative variance error has sd 0.108% with mean about 0; the part propagated from the layer-13 diagonal explains about 1%, so over 95% of it is layer-13 cross-covariance error projected on the unit weights. A leave-one-rank-out regression on 7 runtime features gives R^2 -0.01 (0.8% of the sensitivity-weighted share), below the 20% rule, so no per-unit correction can be built.

Recorded measurement:

    True L14 variance on |a14| < 1: final xMSE -9.29% (SE 1.46); per rank 16 -12.8, 35 -2.9, 150 -7.5, 200 -4.8, 250 -14.2, 259 -7.8, 277 -12.0, 287 -12.3; |a| < 0.5: -3.85% (n 4); all units -6.6% (n 4; kink-only on the same ranks -7.0). Kink v error sd 0.108%; diagonal-propagated part sd ~0.008%, corr ~0.1. LORO regression on 7 features (a, |a|, log v, s, mu^2/v, a s, |a| s; 1,939 kink units): R^2 -0.01, share removed 0.8%.

## Programs

- [cand_s2tr_kinkoracle_v1.py](cand_s2tr_kinkoracle_v1.py)
- [oracle_judge_v2.py](oracle_judge_v2.py)
- [analyze_oracle_v1.py](analyze_oracle_v1.py)
- [predictability_v1.py](predictability_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4336](../../TEAM_LOG_2026-10-01_to_10-22.md#L4336)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4343](../../TEAM_LOG_2026-10-01_to_10-22.md#L4343)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4347](../../TEAM_LOG_2026-10-01_to_10-22.md#L4347)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4353](../../TEAM_LOG_2026-10-01_to_10-22.md#L4353)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4358](../../TEAM_LOG_2026-10-01_to_10-22.md#L4358)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Layer-13 source decomposition of the kink variance error: no switchable source over 2.4%; true L13 means -78%](../2026-10-16-layer-13-source-decomposition-kink-variance-error-no-switchable/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-true-layer-14-pre-activation-variance-kink-units-removes) · [Copy provenance](../../../PROVENANCE.json)

