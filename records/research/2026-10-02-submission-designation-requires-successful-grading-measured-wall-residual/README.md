# Submission designation requires successful grading and measured wall and residual margins

2026-10-02T03:50Z · decision · adopted · measured

The release lane proposed, and the lead adopted, a rule for the file designated for the private re-run: the best-scoring graded file that lost no network in its public run, leaves at least 0.095 s between its mean residual and the 0.4 s cap and 35 s between its mean wall and 120 s by the platform's own telemetry, and holds no error lever judged only on the networks it was tuned on. A one-command designation test (packaging, the pair S35 / file / file / S35 under the platform clock, a 1.3x-slow setup session, the 20-network worker, a judge printing one line per criterion) implements it; S37 and later S38 passed it.

Recorded measurement:

    S37: requests 42,864 (limit 43,300), binding phase 18,004 requests (limit 18,100), projected wall 78.2 s (limit 80), residual 0.302 s (limit 0.305), handles 56,441 (limit 70,000), group RSS 6.20 GB (limit 7.5), native parity true; S38 designation test PASS 08:46-08:59Z

## Programs

- [designation_test.sh](designation_test.sh)
- [designation_judge.py](designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L108](../../TEAM_LOG_2026-10-01_to_10-22.md#L108)
- [TEAM_LOG_2026-10-01_to_10-22.md#L115](../../TEAM_LOG_2026-10-01_to_10-22.md#L115)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L388](../../SESSION_NOTES_2026-10-01_to_10-03.md#L388)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../../../submissions/333435/README.md)
- shipped in: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../../../submissions/333481/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual) · [Copy provenance](../../../PROVENANCE.json)

