# Shorter worker hold rejected because memory-log criterion fails during +-753.5 MiB swap movements

2026-10-12T21:00:54Z · decision · rejected · measured

To decide whether refit workers could ramp from 2 to 6 as soon as a round's files were ready and its stage-2 table posted (rule 'ready_table') rather than after release's gates ('release'), the trainer logged memory_pressure, swap and per-lane RSS once a minute from 13:28Z to 20:59Z (442 samples) and applied a rule written in advance: switch iff >= 10 release-only minutes all had free >= 60% and swap rise <= 64 MiB. The release-only phases showed free >= 77-80% but swap excursions of +-753.5 MiB, so HOLD_RULE stayed 'release' for candidate generation 53 (S53)-candidate generation 66 (S66).

Recorded measurement:

    phases (minutes, min free %, max swap rise MiB, min free+inactive GiB): idle 174, 53%, +0.0, 12.7; port 71, 46%, +0.0, 10.9; port+release 51, 77%, +-753.5, 18.5; port+stage2 99, 48%, +-48.0, 11.5; release 1, 80%, +-753.5, 21.2; stage2 45, 55%, +-48.0, 13.2

## Programs

- [memlog_v1.sh](memlog_v1.sh)
- [hold_decide_v1.sh](hold_decide_v1.sh)
- [hold_decide_v1.py](hold_decide_v1.py)
- [hold_s53_v2.sh](hold_s53_v2.sh)
- [../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/gate.sh](../2026-10-01-concurrent-workers-exhaust-memory-compression-capacity-require-reduced/gate.sh)
- [memlog_v1.py](memlog_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1490](../../TEAM_LOG_2026-10-01_to_10-22.md#L1490)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1493](../../TEAM_LOG_2026-10-01_to_10-22.md#L1493)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1494](../../TEAM_LOG_2026-10-01_to_10-22.md#L1494)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1679](../../TEAM_LOG_2026-10-01_to_10-22.md#L1679)

<details>
<summary>All result files and records (3)</summary>

- [memlog_v1.jsonl](memlog_v1.jsonl) · result
- [HOLD_RULE](HOLD_RULE) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-shorter-worker-hold-rejected-because-memory-log-criterion-fails) · [Copy provenance](../../../PROVENANCE.json)

