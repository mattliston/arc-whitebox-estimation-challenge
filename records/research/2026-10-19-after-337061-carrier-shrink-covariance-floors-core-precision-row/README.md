# After #337061, carrier shrink, covariance floors, core precision and row-selection changes are rejected

2026-10-19T06:50:00Z · lesson · informational · measured

The record of what not to redo after #337061: the half carrier shrink (a split, -0.01%), the dead-unit floor 0.002 (bill -0.13% / raw +0.15%), C-side tree boxes (garbage on the first network: never box the matmul output tree), the Nystrom core numerics, task-aware Nystrom rows (+0.21% on 32 store ranks on 10-16, arms built but not run) and every cheap dial (young scale a sharp V, old scale at layers 8-10 flat, star beta 0.85 optimal). Seven-rank raw reads mislead below 0.5%: use the 32-rank paired judge, count a result only if held-out 25 is negative.

Recorded measurement:

    #337061 local -0.102% (se .029, 25/32 lower; bill -0.092, raw -0.009); HALF all32 -0.007% (se .136); dcv002 +0.015%; task-aware rows +0.213% on 32 store ranks, +0.256% on 8 judge ranks

## Programs

- [candidate_tjB1013_sci_v1.py](candidate_tjB1013_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5909](../../TEAM_LOG_2026-10-01_to_10-22.md#L5909)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5910](../../TEAM_LOG_2026-10-01_to_10-22.md#L5910)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5911](../../TEAM_LOG_2026-10-01_to_10-22.md#L5911)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5913](../../TEAM_LOG_2026-10-01_to_10-22.md#L5913)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5917](../../TEAM_LOG_2026-10-01_to_10-22.md#L5917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5919](../../TEAM_LOG_2026-10-01_to_10-22.md#L5919)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-after-337061-carrier-shrink-covariance-floors-core-precision-row) · [Copy provenance](../../../PROVENANCE.json)

