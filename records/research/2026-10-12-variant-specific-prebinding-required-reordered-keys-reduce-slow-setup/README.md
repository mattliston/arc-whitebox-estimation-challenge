# Variant-specific prebinding required: reordered keys reduce slow-setup request growth from +2.20% to +0.47%

2026-10-12T00:54Z · incident · failed · derived

With the fused trees dropped, S48b's prebind list no longer covered the twin's keys, so a slow worker's setup bound too few and the gate refused +2.20%; a prebind list fit on the twin's own 20-network census (keys reordered by use) brought it to +0.47% and the twin submitted.

Recorded measurement:

    twin 44,989 / 43,306 normal, 45,120 slow (+2.20%, limit 2%); with its own list 44,476 / 42,846 normal, 44,839 slow (+0.47%); S50 safe needed the same (+2.29% -> +0.53%)

## Programs

- [../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py](../../../instruments/targeted-prebinding-replaces-unused-keys-first-network-requests-fall/code/opus_reexam_20261001_v1/engineer/a62_modules/prebind_targeted_v2.py)
- [fragment_prebind_s48tw_v1.py](fragment_prebind_s48tw_v1.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1230](../../TEAM_LOG_2026-10-01_to_10-22.md#L1230)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1232](../../TEAM_LOG_2026-10-01_to_10-22.md#L1232)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1235](../../TEAM_LOG_2026-10-01_to_10-22.md#L1235)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1236](../../TEAM_LOG_2026-10-01_to_10-22.md#L1236)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1254](../../TEAM_LOG_2026-10-01_to_10-22.md#L1254)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1357](../../TEAM_LOG_2026-10-01_to_10-22.md#L1357)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1359](../../TEAM_LOG_2026-10-01_to_10-22.md#L1359)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1362](../../TEAM_LOG_2026-10-01_to_10-22.md#L1362)

<details>
<summary>All result files and records (3)</summary>

- [gate_s48tw_safe.json](gate_s48tw_safe.json) · record
- [gate_s50s2_safe.json](gate_s50s2_safe.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #335451: unfused-tree variant with its own measured prebind list](../../../submissions/335451/README.md)
- enables: [Submission #335508: lower-request fallback for the same carrier-strength refit](../../../submissions/335508/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-variant-specific-prebinding-required-reordered-keys-reduce-slow-setup) · [Copy provenance](../../../PROVENANCE.json)

