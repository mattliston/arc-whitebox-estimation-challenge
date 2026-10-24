# Native old-plane ranks 352, 320 and 288 show limited work savings and increasing raw loss

2026-09-24T17:49Z · experiment · rejected · measured

Native rank cuts reduced runtime but did not provide a material adjusted-score gain without recovering raw error. Rank 320 was retained only for a refit test.

Recorded measurement:

    2026-09-24 10:49 porter: native tier-rank option built (build_tier_rank.py; R384 bit-identical to S11 on 6 nets). No-refit vs S11: R352 bill -1.67% raw +1.64%; R320 bill -3.89% raw +4.26% (r37 +8.12% vs torch +7.93%); R288 bill -5.93% raw +10.58%. Census (server CPU, requests) running.
    - 2026-09-24 10:55 lead: porter's native rank option (build_tier_rank.py), no refit, vs S11 (bill / raw / adjusted): R352 -1.67% / +1.64% / -0.06%; R320 -3.89% / +4.26% / +0.20%; R288 -5.93% / +10.6% / +4.0%. R320 needs the refit to absorb more than ~1.2 points of raw.
    2026-09-24 11:00 porter: rank census vs S11 (paired): R352 req -2.0% CPU -6.5%; R320 req -2.9% CPU -6.3%; R288 req -5.1% CPU -8.7%. Rank cuts lower wall. Waiting for covhunter's r320 refit .pt.
    - 2026-09-24 11:12 lead: porter rank census (paired, same window): every R cut lowers requests, round trips, server CPU and residual. R320 is -2.9% requests and -6.3% server CPU, so a rank cut adds no wall risk.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_tier_rank.py](code/build_tier_rank.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L324](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L324)
- [TEAM_LOG_2026-09-22_to_09-26.md#L327](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L327)
- [TEAM_LOG_2026-09-22_to_09-26.md#L330](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L330)
- [TEAM_LOG_2026-09-22_to_09-26.md#L332](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L332)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-native-old-plane-ranks-352-320-288-show-limited) · [Copy provenance](../../../PROVENANCE.json)

