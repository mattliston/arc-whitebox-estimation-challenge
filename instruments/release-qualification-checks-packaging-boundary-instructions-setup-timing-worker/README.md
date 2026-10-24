# Release qualification checks packaging, boundary instructions, setup timing and worker execution

2026-10-01T19:16Z · tool · informational · measured

A release lane was set up with stage scripts (A packaging, B native error, C paired client-server boundary census, D 20-network worker) and a qualify script that stops before upload. Because single-thread wall time swings 2-2.7x within seconds on the shared machine, release numbers are taken in kernel instructions and cycles (rusage) with seconds as a cross-check; boundary stages later run the estimator's setup under the platform's clock (--short-setup) so that how far the setup gets no longer depends on local load.

Recorded measurement:

    wall swing 2-2.7x within seconds; calibration on banks S32-S35 (12 ordered pairs): kernel projected to 1.5% mean abs error (worst 3.0%), overhead 0.7% (worst 1.2%), residual 2.5% (worst 5.0%), mean wall 1.1 s (worst 2.3 s)

## Programs

- [code/rusage.py](code/rusage.py)
- [code/stage_c_boundary.py](code/stage_c_boundary.py)
- [code/qualify.sh](code/qualify.sh)
- [code/common.py](code/common.py)
- [code/guard_footprint_v3.py](code/guard_footprint_v3.py)
- [code/project.py](code/project.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L9](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L9)
- [TEAM_LOG_2026-10-01_to_10-22.md#L22](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L22)
- [TEAM_LOG_2026-10-01_to_10-22.md#L38](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L38)

<details>
<summary>All result files and records (3)</summary>

- [RUNBOOK.md](RUNBOOK.md) · record
- [RESULT.md](RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-01-release-qualification-checks-packaging-boundary-instructions-setup-timing-worker) · [Copy provenance](../../PROVENANCE.json)

