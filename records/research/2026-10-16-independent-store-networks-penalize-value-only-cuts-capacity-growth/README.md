# Independent store networks penalize value-only cuts; capacity growth does not overcome coefficient co-adaptation

2026-10-16T19:37:34Z · finding · informational · measured

Every cut looked cheaper on the generated seeds than on the store networks (UVP79 -0.09 -> +0.91%, r7rank320 +0.75 -> +1.7%, edgel1012 +0.18 -> +0.62%, L15RC30 +0.05 -> +0.40%, STK4 -0.38 -> +0.37%). The lead tested whether grows closed on seeds would pay on the store: the old-tier tail grows show real accuracy on store but at rate 0.6-0.7 (under 1:1), and rank grows do not pay. So no value-only lever exists without a refit, which the user had excluded.

Recorded measurement:

    Store-32: ott30 (both sides 2.5 -> 3.0) xMSE -0.677% (SE .18, 25/32 lower), bill +1.007 -> xnet +0.33 (rate 0.67); ottp30 (positive side) xMSE -0.295 (SE .13, 23/32), bill +0.488 -> +0.19 (rate 0.6); ott30_1314 n16 MSE -0.25 (SE .11) / bill +0.36 -> +0.11 (rate 0.7), n31 xnet +0.18. Grow prices (warm judge first/warm): r10rank448 +2.42/+2.42% bill, r13rank448 +0.78/+0.61%, ottp30 +0.60/+0.44%, ott30 +1.22/+1.14%.

## Programs

- [cand_ott30.py](cand_ott30.py)
- [cand_ottp30.py](cand_ottp30.py)
- [cand_decomp_ott30_v1.py](cand_decomp_ott30_v1.py)
- [cand_decomp_ott30_1314_v1.py](cand_decomp_ott30_1314_v1.py)
- [../2026-10-16-store-32-validation-every-value-only-part-picked-generated/pool_solo.py](../2026-10-16-store-32-validation-every-value-only-part-picked-generated/pool_solo.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4467](../../TEAM_LOG_2026-10-01_to_10-22.md#L4467)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4469](../../TEAM_LOG_2026-10-01_to_10-22.md#L4469)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4471](../../TEAM_LOG_2026-10-01_to_10-22.md#L4471)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4475](../../TEAM_LOG_2026-10-01_to_10-22.md#L4475)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4502](../../TEAM_LOG_2026-10-01_to_10-22.md#L4502)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-independent-store-networks-penalize-value-only-cuts-capacity-growth) · [Copy provenance](../../../PROVENANCE.json)

