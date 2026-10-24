# Duplicate release fragments break setup binding and invalidate the stage-C run

2026-10-11T19:22Z · incident · failed · derived

The lead fed a file already carrying the three release fragments to a script that appends them; the doubled l3_bind128 broke the setup binding (592 requests bound, first network 49,873) and the FAIL line was void; later scripts take FRAGS=none and the engineer's file byte for byte.

Recorded measurement:

    s48s: setup 592 requests, first network 49,873 vs 44,222; s48t stopped on a script bug; s48u PASS

## Programs

- [../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/s48_safe_v2.sh](../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/s48_safe_v2.sh)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/designation_judge_before_median_7p35_20261011T2057Z.py](../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/designation_judge_before_median_7p35_20261011T2057Z.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1168](../../TEAM_LOG_2026-10-01_to_10-22.md#L1168)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1169](../../TEAM_LOG_2026-10-01_to_10-22.md#L1169)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1170](../../TEAM_LOG_2026-10-01_to_10-22.md#L1170)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1172](../../TEAM_LOG_2026-10-01_to_10-22.md#L1172)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1175](../../TEAM_LOG_2026-10-01_to_10-22.md#L1175)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1186](../../TEAM_LOG_2026-10-01_to_10-22.md#L1186)

<details>
<summary>All result files and records (2)</summary>

- [gate_s48s_safe.json](../2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10/gate_s48s_safe.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-duplicate-release-fragments-break-setup-binding-invalidate-stage-c) · [Copy provenance](../../../PROVENANCE.json)

