# Submission #332241: one-epoch learned refit at the fitted physics scales

2026-09-25T05:46:28Z · submission · adopted · measured

Replaced learned literals with one additional refit epoch at the first-round physics scales; the second-round scale fit was rejected. The official public score is 2.5054244494808718e-09 with 100 networks and 0 failures.

Recorded measurement:

    {"submission_id": 332241, "date": "2026-09-25T05:46:28Z", "status": "GRADED", "score_public": 2.5054244494808718e-09, "n": 100, "failed_networks": 0, "utilization": 0.14814569571712127, "wall_mean_s": 65.151, "wall_max_s": 74.286, "resid_mean_s": 0.238, "resid_max_s": 0.281, "source_sha256": "33d4f0d0f968b07e0494dd00b8e0c3b79cc0f3b9a9f037984f0a4ed2c349c019"}
    - 2026-09-24 22:46 PDT FIRE #332241 (Fable): fable_fire_v1/fire_candS22/candS22_S21s1_r2_mem.py (sha256 33d4f0d0...) = Fable S22 = S21s1 #332236 + the physicist's loopR1 refit literals (one refit epoch on the round-1-dial card, warm from j05 ep2; VALID16 -0.95%) + round-1 dials unchanged (the round-2 dial fit converged: LOO +0.44%).
    - 2026-09-24 23:11 PDT GRADED #332241 (S22): adjusted 2.5054244494808718e-9, raw 1.6916191079019428e-8, util 0.14815. NEW PHASE-2 BANK, -1.10% vs #332236 (2.5332e-9); expected ~2.501e-9 (VALID40 -1.26%).
    - 2026-09-24 23:45 PDT GRADED #332246 (S23): adjusted 2.395096202270728e-9, raw 1.6912490359288768e-8, util 0.14166. NEW PHASE-2 BANK, -4.40% vs #332241 (2.5054e-9), -5.45% vs S21s1; expected ~2.384e-9 (VALID40 vs S21s1 -5.90%).

Public score: 2.5054244494808718e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L854](../../records/SUBMISSION_LEDGER.md#L854)
- [SUBMISSION_LEDGER.md#L860](../../records/SUBMISSION_LEDGER.md#L860)
- [SUBMISSION_LEDGER.md#L871](../../records/SUBMISSION_LEDGER.md#L871)
- [TEAM_LOG_2026-09-22_to_09-26.md#L722](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L722)
- [TEAM_LOG_2026-09-22_to_09-26.md#L750](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L750)

<details>
<summary>All result files and records (6)</summary>

- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [submit_output.txt](submit_output.txt) · record
- [FIRE_CHECKS.txt](FIRE_CHECKS.txt) · record
- [poll.log](poll.log) · log
- [eval_332241.json](eval_332241.json) · result

</details>

## Related work

- ← shipped in: [Alternating learned-on dial fit and refit improves held-out error 0.89% in the first round](../../experiments/dials/alternating-learned-dial-fit-refit-improves-held-out-error/README.md)
- ← shipped in: [Refit on the new trajectory transfers to native validation: adjusted error improves 1.259%](../../experiments/learned-correction/refit-new-trajectory-transfers-native-validation-adjusted-error-improves/README.md)
- supersedes: [Submission #332236: rank-32 evaluation of the residual-plane cross term](../332236/README.md)
- ← supersedes: [Submission #332246: cross-term removal and Nyström old-cohort joins](../332246/README.md)

[Research graph](../../graph/index.html#node=submission-332241) · [Copy provenance](../../PROVENANCE.json)

