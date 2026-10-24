# Hosted grading monitor refreshes already-scanned submissions using new raw snapshots

2026-10-21T20:57:50Z · tool · adopted · measured

The prior identifier scan skipped already-seen submissions and returned stale grades. A fresh-file poll supplied updated per-network and final state for the submitted candidates.

Recorded measurement:

    - 2026-10-21T20:57:50Z [fable-lead] BOTH PASSED SETUP (5 workers each 4.13-4.25 s): #337617 grading 42/50 public, raw final MSE -0.29% vs #337565 on the same 42 nets (better on 27), util .10870 = same -> ~-0.3% adjusted so far; #337619 grading 23/50 public, raw -0.30% on the same 23 nets (16 better), util .10820 (-0.47% bill) -> ~-0.75% adjusted so far (~1.393e-9 if it holds). Walls max 107.3 / 106.7 s, residual max .363/.362. Old per-id polls were stale (scan_submissions skips already-scanned ids) -> replaced by a fresh-file poll (b0g5ejrvf); refire watcher no longer needed.

## Programs

- [code/fetch_telemetry_v1.py](code/fetch_telemetry_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7555](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7555)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-21-hosted-grading-monitor-refreshes-already-scanned-submissions-using-new) · [Copy provenance](../../PROVENANCE.json)

