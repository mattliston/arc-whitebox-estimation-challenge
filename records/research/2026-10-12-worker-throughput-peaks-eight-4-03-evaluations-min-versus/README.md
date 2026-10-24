# Worker throughput peaks at eight: 4.03 evaluations/min versus 2.40 at twelve

2026-10-12T18:30Z · lesson · informational · measured

An A/B inside candidate generation 51 (S51)'s phase 2 showed throughput collapsing above 8 workers on the 16-core box (about 140k zero-fill page faults a second at 12); with macOS scheduling only ~8 cores on the 96 W adapter the cap went to 4, and side jobs must pin BLAS to one thread (an unpinned solve took 12+ min instead of 14 s).

Recorded measurement:

    8 workers 121 evals / 30 min = 4.03/min (forward 118.8 s); 12 workers 72 = 2.40/min (297.5 s); 4 workers quiet 2.9/min > 8 overloaded 2.1

## Programs

- [../../../instruments/worker-maintainer-restarts-indices-requeues-abandoned-claims-scales-concurrency/code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_analyze_v1.py](../../../instruments/worker-maintainer-restarts-indices-requeues-abandoned-claims-scales-concurrency/code/opus_reexam_20261001_v1/trainer/cert/RAD/S51/ab_analyze_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1465](../../TEAM_LOG_2026-10-01_to_10-22.md#L1465)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1504](../../TEAM_LOG_2026-10-01_to_10-22.md#L1504)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1506](../../TEAM_LOG_2026-10-01_to_10-22.md#L1506)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1518](../../TEAM_LOG_2026-10-01_to_10-22.md#L1518)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1582](../../TEAM_LOG_2026-10-01_to_10-22.md#L1582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1596](../../TEAM_LOG_2026-10-01_to_10-22.md#L1596)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1617](../../TEAM_LOG_2026-10-01_to_10-22.md#L1617)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1620](../../TEAM_LOG_2026-10-01_to_10-22.md#L1620)

<details>
<summary>All result files and records (2)</summary>

- [AB_v1.json](AB_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-worker-throughput-peaks-eight-4-03-evaluations-min-versus) · [Copy provenance](../../../PROVENANCE.json)

