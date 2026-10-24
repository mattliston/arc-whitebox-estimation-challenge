# Setup guard retained at 4.0 s: observed maximum setup 4.336 s under the 5 s cap

2026-10-10T03:14Z · lesson · informational · measured

Our setups are the board's longest (max 4.336 s) but the tail beyond the 4.0 s wall-clock guard is only 0.28-0.34 s; a breach needs a 3.3x slowdown; the judge gained a line for projected setup wall <= 4.5 s (4.6 s in the 1.3x session).

Recorded measurement:

    86 setups in 15 runs: max 4.336 s, median 4.282 s; board setup slowness p99 1.11x, p99.9 1.22x, max 1.235x; lower guards 3.5 / 3.0 s cost +520 / +1,050 requests on a first network

## Programs

- [../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py](../2026-10-01-four-second-setup-budget-binds-approximately-19-000-requests/setup_clock.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L520](../../TEAM_LOG_2026-10-01_to_10-22.md#L520)
- [TEAM_LOG_2026-10-01_to_10-22.md#L525](../../TEAM_LOG_2026-10-01_to_10-22.md#L525)
- [TEAM_LOG_2026-10-01_to_10-22.md#L533](../../TEAM_LOG_2026-10-01_to_10-22.md#L533)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-setup-guard-retained-4-0-s-observed-maximum-setup) · [Copy provenance](../../../PROVENANCE.json)

