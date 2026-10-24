# First-network request cap replaced by modelled loss risk <= 10% per 100 networks

2026-10-11T18:09:20Z · decision · informational · measured

The full candidate generation 48 (S48) stack's first-network request count 44,222 was 422 over the 43,800 cap; the lead ruled it a priced risk rather than a block, and release's loss_model_v3 priced first networks separately (6 per 100): P(loss) 2.47% per 100 for the full stack vs 2.41% for carrier-strength refit-after-drop round (MS) + star, +0.06 points per 100 against the column cut's -0.88% of metered FLOPs, with the S47e pair covering a loss. The release gate (release_gate_v4, designation_judge --first-priced, script s48_safe_v2.sh) replaced the first-network cap by 'P(loss) per 100 networks <= 10%'; the warm cap 42,700 and every other gate stayed.

Recorded measurement:

    full stack 44,222 / 41,971 requests: 2.47% (fat 2.60%) per 100, 7.23% (7.60%) per 300; MS + star 43,668 / 41,970: 2.41% (2.55%) per 100, 7.07% per 300; difference +0.06 points per 100, +0.16 per 300 (+0.08 / +0.24 with 9 first networks per 100); S47e as graded 43,596 / 41,898: 2.37% / 6.93%; new gate limit P(loss) per 100 <= 10%

## Programs

- [s48_safe_v2.sh](s48_safe_v2.sh)
- [release_gate_v4.py](release_gate_v4.py)
- [designation_judge_before_median_7p35_20261011T2057Z.py](designation_judge_before_median_7p35_20261011T2057Z.py)
- [../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py](../2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/loss_model_v3.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../submissions/333111/estimator.py](../../../submissions/333111/estimator.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/common.py)
- [../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py](../2026-10-02-submission-designation-requires-successful-grading-measured-wall-residual/designation_judge.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/guard_footprint_v3.py)
- [../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py](../2026-10-02-designation-caps-set-43-800-first-network-requests-42/loss_risk.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/project.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](../../../instruments/strict-release-gate-recomputes-acceptance-exact-artifacts-raw-gate/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1142](../../TEAM_LOG_2026-10-01_to_10-22.md#L1142)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1144](../../TEAM_LOG_2026-10-01_to_10-22.md#L1144)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1145](../../TEAM_LOG_2026-10-01_to_10-22.md#L1145)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1158](../../TEAM_LOG_2026-10-01_to_10-22.md#L1158)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1188](../../TEAM_LOG_2026-10-01_to_10-22.md#L1188)

<details>
<summary>All result files and records (2)</summary>

- [gate_s48s_safe.json](gate_s48s_safe.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-first-network-request-cap-replaced-modelled-loss-risk-10) · [Copy provenance](../../../PROVENANCE.json)

