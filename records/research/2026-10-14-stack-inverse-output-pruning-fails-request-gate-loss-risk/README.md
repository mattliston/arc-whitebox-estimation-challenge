# Stack with inverse output pruning fails the request gate (loss risk 88.33%, 57,693/52,643 requests); pruning dropped

2026-10-14T17:43:57Z · incident · failed · measured

The combined feasible-head-mean + dense reuse + Gaussian T22 reuse + inverse-output-pruning file failed the normal production boundary: its per-network request count projected a 100-network loss risk far above the 20% limit, because the pruning adds about two thousand charged requests per network. A per-plan view memoisation (IVD) that deduplicates the pruning's views also failed the request-risk gate. The fallback with pruning switched off passed and was fired as #336075.

Recorded measurement:

    combined (6f) normal boundary: modeled 100-network loss risk 88.33% (limit 20%), first 57,693 / warm 52,643 requests; IOP-off fallback (4e): normal production PASS, risk 15.04%

## Programs

- [fragment_iop_view_dedup_HELD_v1.py](fragment_iop_view_dedup_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2921](../../TEAM_LOG_2026-10-01_to_10-22.md#L2921)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2928](../../TEAM_LOG_2026-10-01_to_10-22.md#L2928)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2930](../../TEAM_LOG_2026-10-01_to_10-22.md#L2930)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2939](../../TEAM_LOG_2026-10-01_to_10-22.md#L2939)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2943](../../TEAM_LOG_2026-10-01_to_10-22.md#L2943)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2953](../../TEAM_LOG_2026-10-01_to_10-22.md#L2953)

<details>
<summary>All result files and records (4)</summary>

- [BOUNDARY_STOPPED_v1.json](BOUNDARY_STOPPED_v1.json) · result
- [FIXED6_VIEW_COUNT_SUMMARY_v1.json](FIXED6_VIEW_COUNT_SUMMARY_v1.json) · result
- [BOUNDARY_COMPLETE_HELD_v1.json](BOUNDARY_COMPLETE_HELD_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- rejects: [Inverse output pruning: exact, bill -0.044%/-0.035% with 1,979/2,312 additional requests](../../../experiments/flop-accounting/inverse-output-pruning-exact-bill-0-044-0-035/README.md)
- enables: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../../../submissions/336075/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-stack-inverse-output-pruning-fails-request-gate-loss-risk) · [Copy provenance](../../../PROVENANCE.json)

