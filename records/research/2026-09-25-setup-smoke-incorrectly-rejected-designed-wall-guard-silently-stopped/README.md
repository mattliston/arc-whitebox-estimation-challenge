# Setup smoke incorrectly rejected its designed wall guard and silently stopped the build chain

2026-09-25T05:13Z · incident · failed · measured

Under load, setup correctly stopped prebinding at its own guard. The smoke test was corrected to accept guarded finite outputs without a setup error, and the chain gained failure logging.

Recorded measurement:

    2026-09-24 22:13 porter: S22s1 (r2) SMOKE FAILED
    - 2026-09-24 22:22 lead: S22 CHAIN DIED SILENTLY (~22:15) at the setup smoke. AssertionError: setup_wall_s 3.004 with a 'wall guard' stop (284/362 bound) under load (4 heavy jobs); the guard is the designed safety vs the 5 s platform limit, and platform setup for this code path runs 1.5-2.05 s (10/10 SUCCESS). S21jn then took the lock (22:16) for its VALID40. Porter: relaunch S22 behind it; the smoke accepts a wall-guard stop when setup had no error, every net scored OK, and wall <= 3.1 s; chains must alert on death. Lock order after S21jn: S22 > s1=0 VALID40.
    2026-09-24 22:24 porter: S22 RELAUNCHED as fire_chain_S22s1_v2.sh (log fire_chain_S22_v2.out): the 22:12 run died at the smoke assertion because the prebind hit its 3.0 s wall guard under load (284/362 bound; 4 heavy jobs). v2 smoke accepts a guarded stop (no restore, bound present, wall <= 3.1 s, 16384 finite outputs per net; bound count recorded), census v3 records finiteness, and an EXIT trap logs any chain death here. S22 takes the lock after S21jn's VALID40 (~22:27).
    2026-09-24 22:45 porter: S22 STAGED fable_fire_v1/fire_candS22/candS22_S21s1_r2_mem.py (33d4f0d0): VALID40 vs S21+s1 adj -1.259% [-1.708, -0.816] 34/40 PASS (-0.5% bar); RESIDUAL PASS cold -0.003 s (recomputed; in-chain analysis had a tag typo, addendum in FIRE_CHECKS). S23 relaunched (engineer2 renamed their queue run_after_S22_v2.sh; pattern fixed).

## Programs

- [fire_chain_S22s1_v2.sh](fire_chain_S22s1_v2.sh)
- [../../../submissions/332241/estimator.py](../../../submissions/332241/estimator.py)
- [run_after_S22_v2.sh](run_after_S22_v2.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L698](../../TEAM_LOG_2026-09-22_to_09-26.md#L698)
- [TEAM_LOG_2026-09-22_to_09-26.md#L706](../../TEAM_LOG_2026-09-22_to_09-26.md#L706)
- [TEAM_LOG_2026-09-22_to_09-26.md#L708](../../TEAM_LOG_2026-09-22_to_09-26.md#L708)
- [TEAM_LOG_2026-09-22_to_09-26.md#L721](../../TEAM_LOG_2026-09-22_to_09-26.md#L721)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-setup-smoke-incorrectly-rejected-designed-wall-guard-silently-stopped) · [Copy provenance](../../../PROVENANCE.json)

