# Worker guard tolerates process-list timeouts and conditions swap retirement on memory pressure

2026-10-12T23:46Z · tool · informational · derived

The lifecycle guard retired every worker on one slow ps call (0.05 s polling, ~80 ps calls a second) and later on a 256 MiB swap growth at 74% free memory in every guarded job at once; v2 tolerates failed readings and polls once a second, v3 kills on swap only with memory_pressure free < 35% or growth > 1 GiB.

Recorded measurement:

    10-12 08:06Z all 8 S51 workers retired by ps timeout; 23:39Z cascade swap 1717M -> 1986M at 74% free; v2 self-test 6/6, v3 self-test all pass

## Programs

- [code/guard_footprint_v2.py](code/guard_footprint_v2.py)
- [code/guard_footprint_v3.py](code/guard_footprint_v3.py)
- [code/rusage.py](code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1346](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1346)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1347](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1347)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1439](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1439)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1440](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1440)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1802](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1802)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1804](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1804)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1805](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1805)

<details>
<summary>All result files and records (3)</summary>

- [SELFTEST.json](SELFTEST.json) · result
- [SELFTEST-3ccf5b46.json](SELFTEST-3ccf5b46.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-12-worker-guard-tolerates-process-list-timeouts-conditions-swap-retirement) · [Copy provenance](../../PROVENANCE.json)

