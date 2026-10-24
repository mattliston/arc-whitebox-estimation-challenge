# Plane-read column cut a >= -2.0 adopted: about -0.50% score; row cuts corrupt birth updates

2026-10-11T16:39Z · decision · adopted · derived

Restricting the plane read's columns to the live roster's prefix at a >= -2.0 nets about -0.5% at 1.25 se; restricting its rows is catastrophic because the birth update reads every hub's whole third-cumulant plane with index multiplicities (2,1) (K21) row; thresholds 2.2 / 2.4 / 1.8 lose to 2.0 on score.

Recorded measurement:

    rows |a| <= 2.0 +13.75% (0 of 8) for -3.25% bill; columns -2.0 real build MSE +0.38% (se 0.40), bill -0.88%; one-sided rows a > +2 +16.8%, a < -2 +7.1%; rdc 2.2 -> +0.28% score, 1.8 +0.42%

## Programs

- [rows_probe_v1.py](rows_probe_v1.py)
- [fragment_readcol_v1.py](fragment_readcol_v1.py)
- [make_rdz_v3.py](make_rdz_v3.py)
- [judge_rdz3_v1.py](judge_rdz3_v1.py)
- [t122_table.py](t122_table.py)
- [candidate.py](candidate.py)
- [table_v1.py](table_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1102](../../TEAM_LOG_2026-10-01_to_10-22.md#L1102)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1103](../../TEAM_LOG_2026-10-01_to_10-22.md#L1103)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1107](../../TEAM_LOG_2026-10-01_to_10-22.md#L1107)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1108](../../TEAM_LOG_2026-10-01_to_10-22.md#L1108)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1110](../../TEAM_LOG_2026-10-01_to_10-22.md#L1110)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1112](../../TEAM_LOG_2026-10-01_to_10-22.md#L1112)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1233](../../TEAM_LOG_2026-10-01_to_10-22.md#L1233)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1257](../../TEAM_LOG_2026-10-01_to_10-22.md#L1257)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1268](../../TEAM_LOG_2026-10-01_to_10-22.md#L1268)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1273](../../TEAM_LOG_2026-10-01_to_10-22.md#L1273)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Module audit: learned modules use 5.4% of FLOPs and every removal costs more error than FLOPs saved](../2026-10-11-module-audit-learned-modules-use-5-4-flops-every/README.md)
- shipped in: [Submission #335411: carrier-strength refit, one-direction covariance star and plane-read column cut](../../../submissions/335411/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-plane-read-column-cut-2-0-adopted-about-0) · [Copy provenance](../../../PROVENANCE.json)

