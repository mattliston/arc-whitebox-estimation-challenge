# Local network ranks are not hosted public indices; the selected slow-network panel was discarded

2026-10-21T15:14:48Z · incident · failed · measured

The intended slow-public panel actually selected local cached networks with matching ordinal numbers. Different FLOP counts proved those identities did not match, so margin decisions required a uniform mechanism instead.

Recorded measurement:

    - 2026-10-21T15:14:48Z [fable-lead] CORRECTION: local store ranks are NOT the hosted evaluation MLPs (hosted idx16/35 FLOPs differ from local ranks 16/35 by -2.0%/+0.8%). The slowpub9 plan (lead/runs/fable_oct21_final_v1/slowpub9_v1) used local networks 28/21/1/42/17/23/7/4/3, not the hosted slow public nets; dropped. Margin must be a uniform request/push cut judged on the worst case.

## Programs

- [../2026-10-21-hosted-wall-time-correlates-requests-tensor-writes-more-strongly/wall_anatomy_v1.py](../2026-10-21-hosted-wall-time-correlates-requests-tensor-writes-more-strongly/wall_anatomy_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7442](../../TEAM_LOG_2026-10-01_to_10-22.md#L7442)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [PLAN_v1.json](PLAN_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-local-network-ranks-not-hosted-public-indices-selected-slow) · [Copy provenance](../../../PROVENANCE.json)

