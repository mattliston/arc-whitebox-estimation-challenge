# Performance projections require the latest graded parent and fault-free instruction counts

2026-10-02T09:14Z · lesson · informational · derived

Three fires taught the release lane's projection rule: the residual is requests x 7 microseconds (exact to 1.5% on every fire); the kernel projection scales memory-bound and CPU-bound work separately with the memory-bound share re-set from 0.72 to 0.65 (the fires implied 0.62 / 0.72 / 0.55); kernel instruction counts include page-fault work (14,200 instructions per minor fault), which caused every window-dependent miss (S37's wall projected 3.7 s low) and is now subtracted; and every stage projects from the newest graded file (its own platform record and stored legs) rather than from S35, whose overhead per round trip differs. S39's projection then held to +0.3 s of wall.

Recorded measurement:

    residual per request 7.04 / 6.93 / 7.07 / 7.03 / 7.08 microseconds (S35-S39); memory-bound share implied 0.62 (S36), 0.72 (S37), 0.55 (S38), set 0.65; instructions 349.8 G at 25,664 faults vs 381.1 G at 2,371,499 faults (S35, network 100); fit 14,200 instructions per fault, 0.27% rms left (raw 2.2%); re-projected fires S36 +1.4..+2.1%, S37 -0.6..+0.1%, S38 -1.9..-1.1% from any window (raw -9.2..+9.4%); overhead per round trip S35 0.1833 ms, S36-S38 0.1777-0.1794; S39: wall 77.6 projected / 77.3 measured, kernel 68.75 / 68.44 s, residual 0.2998 / 0.3018 s

## Programs

- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [set_bank.py](set_bank.py)
- [new_bank.sh](new_bank.sh)
- [check_projection.py](check_projection.py)
- [compare_parent_legs.py](compare_parent_legs.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/stage_c_boundary.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L91](../../TEAM_LOG_2026-10-01_to_10-22.md#L91)
- [TEAM_LOG_2026-10-01_to_10-22.md#L105](../../TEAM_LOG_2026-10-01_to_10-22.md#L105)
- [TEAM_LOG_2026-10-01_to_10-22.md#L202](../../TEAM_LOG_2026-10-01_to_10-22.md#L202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L213](../../TEAM_LOG_2026-10-01_to_10-22.md#L213)
- [TEAM_LOG_2026-10-01_to_10-22.md#L226](../../TEAM_LOG_2026-10-01_to_10-22.md#L226)
- [TEAM_LOG_2026-10-01_to_10-22.md#L227](../../TEAM_LOG_2026-10-01_to_10-22.md#L227)
- [TEAM_LOG_2026-10-01_to_10-22.md#L277](../../TEAM_LOG_2026-10-01_to_10-22.md#L277)
- [TEAM_LOG_2026-10-01_to_10-22.md#L292](../../TEAM_LOG_2026-10-01_to_10-22.md#L292)
- [RESULT.md#L244](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md#L244)

<details>
<summary>All result files and records (4)</summary>

- [bank.json](bank.json) · config
- [bank_stage_d_paced.json](bank_stage_d_paced.json) · config
- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-performance-projections-require-latest-graded-parent-fault-free-instruction) · [Copy provenance](../../../PROVENANCE.json)

