# Designation judge replaces mean-second thresholds with request counts and worst-network limits

2026-10-02T18:25Z · decision · informational · measured

Submission #333587 ran 3-4% slower in kernel, overhead and residual time together, with the same file accelerating during the run. Because platform-hour variation makes fixed mean-second caps incomparable, the judge retains request and slow-setup limits and requires worst-network residual at most 0.36 s and roundtrip at most 105 s.

Recorded measurement:

    kernel projected / observed 71.97 / 74.28 s; overhead 8.648 / 9.027 s; mean residual 0.3047 / 0.3182 s; wall 80.9 / 83.6 s; kernel ms/Gflop by quarter 296.2 / 293.9 / 286.9 / 283.5; 0.36 s residual and 105 s roundtrip limits

## Programs

- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L417](../../TEAM_LOG_2026-10-01_to_10-22.md#L417)
- [TEAM_LOG_2026-10-01_to_10-22.md#L418](../../TEAM_LOG_2026-10-01_to_10-22.md#L418)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-designation-judge-replaces-mean-second-thresholds-request-counts-worst) · [Copy provenance](../../../PROVENANCE.json)

