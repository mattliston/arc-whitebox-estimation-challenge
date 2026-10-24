# Qualification memory gates use physical footprint with resident memory as a backstop

2026-10-02T08:52Z · decision · adopted · measured

On the shared machine group RSS counts freed pages the system has not taken back, so quiet sessions showed 6.0-6.8 GB of RSS at footprints of 2.1-2.7 GB and S38's worker came within 0.7 GB of the guard's 7.5 GB RSS kill; the release stages and the fast path now run under a guard copy that kills at a 7.5 GB physical footprint (server plus client peaks) with RSS as a hard stop at 10 GB. The change was needed the same day: S39's worker reached 7.23 GB of RSS at a footprint of 2.47 GB.

Recorded measurement:

    quiet sessions group RSS 6.0-6.8 GB vs loaded 2.9-3.9 GB at footprints 2.1-2.7 GB; S38 worker RSS 6.81 GB, footprint 2.20 GB; S37 worker footprint 3.34 GB; S39 worker RSS 7.23 GB, footprint 2.47 GB

## Programs

- [guard_footprint.py](guard_footprint.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L187](../../TEAM_LOG_2026-10-01_to_10-22.md#L187)
- [TEAM_LOG_2026-10-01_to_10-22.md#L213](../../TEAM_LOG_2026-10-01_to_10-22.md#L213)
- [TEAM_LOG_2026-10-01_to_10-22.md#L277](../../TEAM_LOG_2026-10-01_to_10-22.md#L277)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-qualification-memory-gates-use-physical-footprint-resident-memory-backstop) · [Copy provenance](../../../PROVENANCE.json)

