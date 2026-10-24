# Loose worker-trigger pattern kills two refit workers prematurely; exact marker matching replaces it

2026-10-12T06:04:22Z · incident · failed · measured

The trainer's add_workers_v3 keyed the launch of candidate generation 51 (S51) workers 2 / 3 on loose patterns and matched release's 05:21Z 'candidate generation 50 (S50) ORDER NOTED ... s50s3 ... gate' line as if the s50s3 gate had already passed; both workers were started and killed again at 06:03:56Z before their gradient peak, their claims (r7100286 / 287) requeued. The trigger was re-armed as add_workers_v4 on exact release strings ('TWIN s50tw (' with 'VERDICT:' and 's50s3 SAFE GATE PASS' / 's50s3 SAFE: DO NOT FIRE') plus 5 minutes at >= 60% memory free; later rounds' add-on scripts used lane-anchored fixed strings.

Recorded measurement:

    2 workers killed at 06:03:56Z, claims r7100286 / 287 requeued, swap flat at 2.54 GB

## Programs

- [add_workers_v3.sh](add_workers_v3.sh)
- [add_workers_v4.sh](add_workers_v4.sh)
- [add_workers_s52.sh](add_workers_s52.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1375](../../TEAM_LOG_2026-10-01_to_10-22.md#L1375)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1389](../../TEAM_LOG_2026-10-01_to_10-22.md#L1389)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1391](../../TEAM_LOG_2026-10-01_to_10-22.md#L1391)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-loose-worker-trigger-pattern-kills-two-refit-workers-prematurely) · [Copy provenance](../../../PROVENANCE.json)

