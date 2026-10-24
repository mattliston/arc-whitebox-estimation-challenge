# Residual time is dominated by client requests: estimator Python accounts for only 3-5%

2026-10-11T07:43Z · finding · informational · derived

A stub estimator at the boundary measured flopscope's own client work at 42.9k instructions per request; our files run 44.6-45.5k, so only 3-5% of the residual is our Python and there are no 30 ms of Python to cut; a prebind-list refit on census networks overfits.

Recorded measurement:

    flopscope 42.9k instructions per view, 43.5k per in-place add; S46 + XRQ2 + HZ7R 44.6k / 44.8k; lazy binds 0.17% (fit) / 0.21% (held out) of bill

## Programs

- [floor_candidate.py](floor_candidate.py)
- [floor_table.py](floor_table.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1042](../../TEAM_LOG_2026-10-01_to_10-22.md#L1042)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Residual-time model has a heavy tail; retain free request reductions without buying costly margin](../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-residual-time-dominated-client-requests-estimator-python-accounts-only) · [Copy provenance](../../../PROVENANCE.json)

