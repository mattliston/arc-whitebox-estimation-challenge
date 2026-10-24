# A concurrent grading rescan triggered the native-judge guard and orphaned four groups

2026-10-21T17:45:24Z · incident · failed · measured

A Python grading rescan during a Mac native panel triggered the wrapper ban on external heavy processes. The judge terminated and left its queue stopped; cleanup restored the queue and the policy moved grading polls away from the Mac numerical window.

Recorded measurement:

    - 2026-10-21T17:45:24Z [fable-lead] INCIDENT: the a11 grading rescan (python) running on the Mac tripped the Mac32 wrapper's deny_external_heavy mid-run (17:43Z): judge died, 4 orphan census groups, queue 77740 left SIGSTOPped. Recovered: SIGCONT 77740, TERM orphans, archived partial dir (science32_iter2_s10_cap3_parallel5_v1_killed_*). LESSON: no rescans/extra Mac python while a Mac32 runs. Mac32 reruns started 17:47Z: iter2_s10_cap3, extgate_s15_small, iter2_s10_cap4 (sequential).
    - Mac rule: no rescans/extra python while a Mac32 runs (wrapper deny kills the judge; queue 77740 must then be SIGCONTed). Grading/J2W polls now run on the VM (python3 /home/arcworker/arc-runs/fetch_telemetry_v1.py SID OUT).
    - 2026-10-21T20:46:32Z [fable-lead] hgfh_c4s8_k4lc105_clean: gates CLEAR (archive b6cc634bc9a6, tag fable_hgfh_c4s8_k4lc105_clean_05ae2dfc_v1); its first Mac32 died (one of nine groups exit 1 at 20:44:59Z, likely the fire_once Mac python at 20:39Z tripping the judge guard) -> partial moved aside, Mac32 re-running (k4lc_mac32_rerun.sh); VM32 18/32. #337617 grading clean (5+/100); #337619 queued. Matt away until ~21:35Z; last two shots HELD.

## Programs

- [../2026-10-21-hosted-wall-time-correlates-requests-tensor-writes-more-strongly/fetch_telemetry_v1.py](../2026-10-21-hosted-wall-time-correlates-requests-tensor-writes-more-strongly/fetch_telemetry_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7509](../../TEAM_LOG_2026-10-01_to_10-22.md#L7509)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7517](../../TEAM_LOG_2026-10-01_to_10-22.md#L7517)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7552](../../TEAM_LOG_2026-10-01_to_10-22.md#L7552)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-concurrent-grading-rescan-triggered-native-judge-guard-orphaned-four) · [Copy provenance](../../../PROVENANCE.json)

