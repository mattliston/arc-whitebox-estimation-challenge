# Residual time is approximately seven microseconds per charged request

2026-10-01T23:00Z · finding · informational · derived

A client profile of the bank estimator on network 35 shows 43,438 flopscope calls for 42,527 requests and no hot spot in the estimator's own Python; the residual (client-side time not covered by metered kernels) is paid per flopscope call, two thirds of it in flopscope's own wrapper. On the platform the residual per request measured 7.04 (S35), 6.93 (S36) and 7.07 (S37) microseconds, so residual = requests x 7 microseconds to 1.5%.

Recorded measurement:

    43,438 flopscope calls for 42,527 requests (network 35); platform residual per request S35 7.04, S36 6.93, S37 7.07 microseconds

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L52](../../TEAM_LOG_2026-10-01_to_10-22.md#L52)
- [TEAM_LOG_2026-10-01_to_10-22.md#L91](../../TEAM_LOG_2026-10-01_to_10-22.md#L91)
- [TEAM_LOG_2026-10-01_to_10-22.md#L105](../../TEAM_LOG_2026-10-01_to_10-22.md#L105)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-residual-time-approximately-seven-microseconds-per-charged-request) · [Copy provenance](../../../PROVENANCE.json)

